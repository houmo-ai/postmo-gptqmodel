"""QuaRot-style offline Hadamard rotation for Laguna causal language models.

The transform changes the residual-stream basis from ``x`` to ``x @ Q`` while
preserving the floating-point model function.  AutoRound must receive the same
mutated model object afterwards; reloading the original checkpoint would discard
the rotation.

Laguna needs a dedicated implementation because it combines:

* packed routed-expert tensors;
* a BF16 sigmoid router and a shared expert;
* a per-head attention gate before ``o_proj``; and
* layers whose attention width can be larger than ``hidden_size``.

All large matrix products are chunked and copied back to the source device.  In
the intended pipeline the model remains on CPU while one chunk at a time uses a
GPU for the rotation calculation.
"""

from __future__ import annotations

import math
from typing import Iterable

import torch


QUAROT_CONFIG_KEY = "quarot_config"


def _get_model_parts(model):
    if not hasattr(model, "model") or not hasattr(model.model, "layers"):
        raise ValueError(f"Unsupported Laguna model layout: {type(model).__name__}")
    if not hasattr(model, "lm_head"):
        raise ValueError("Laguna QuaRot requires an lm_head module")
    return model.model.layers, model.model.embed_tokens, model.model.norm, model.lm_head


def validate_laguna_quarot(model, *, headwise_v_o: bool) -> None:
    """Fail before mutation when the model does not satisfy rotation invariants."""
    layers, embedding, final_norm, lm_head = _get_model_parts(model)
    hidden_size = int(model.config.hidden_size)
    if embedding.weight.shape[-1] != hidden_size or lm_head.in_features != hidden_size:
        raise ValueError("Laguna embedding/lm_head dimensions do not match config.hidden_size")
    if final_norm.weight.numel() != hidden_size:
        raise ValueError("Laguna final RMSNorm dimension does not match config.hidden_size")

    for index, layer in enumerate(layers):
        for name in ("input_layernorm", "post_attention_layernorm", "self_attn", "mlp"):
            if not hasattr(layer, name):
                raise ValueError(f"Laguna layer {index} is missing required module {name!r}")
        attention = layer.self_attn
        for name in ("q_proj", "k_proj", "v_proj", "o_proj", "head_dim"):
            if not hasattr(attention, name):
                raise ValueError(f"Laguna attention layer {index} is missing {name!r}")
        if headwise_v_o and hasattr(attention, "g_proj") and not bool(
            getattr(attention, "gate_per_head", False)
        ):
            raise ValueError(
                "Head-wise V/O rotation is only equivalent for Laguna per-head gating; "
                f"layer {index} uses per-element gating"
            )

        mlp = layer.mlp
        if hasattr(mlp, "experts"):
            experts = mlp.experts
            packed_gate_up = getattr(experts, "gate_up_proj", None)
            if isinstance(packed_gate_up, torch.nn.Parameter) and not isinstance(
                getattr(experts, "down_proj", None), torch.nn.Parameter
            ):
                raise ValueError(f"Laguna packed experts at layer {index} are missing down_proj")


def _get_norm_scale(layernorm: torch.nn.Module) -> torch.Tensor:
    return layernorm.weight.detach().to(dtype=torch.float64)


def _reset_norm(layernorm: torch.nn.Module) -> None:
    layernorm.weight.data.fill_(1.0)
    bias = getattr(layernorm, "bias", None)
    if bias is not None:
        bias.data.zero_()


def _scale_last_dim_(
    tensor: torch.Tensor,
    scale: torch.Tensor,
    *,
    chunk_rows: int,
) -> None:
    """Multiply a tensor's last dimension by ``scale`` without a full FP64 copy."""
    if tensor.shape[-1] != scale.numel():
        raise ValueError(
            f"Norm scale has {scale.numel()} values but weight shape is {tuple(tensor.shape)}"
        )

    flat = tensor.data.reshape(-1, tensor.shape[-1])
    scale_on_device = scale.to(device=flat.device)
    for start in range(0, flat.shape[0], chunk_rows):
        stop = min(start + chunk_rows, flat.shape[0])
        source = flat[start:stop].to(dtype=torch.float64)
        flat[start:stop].copy_((source * scale_on_device).to(dtype=flat.dtype))


def _fuse_norm_into_linears(
    layernorm: torch.nn.Module,
    linears: Iterable[torch.nn.Linear],
    *,
    chunk_rows: int,
) -> None:
    scale = _get_norm_scale(layernorm)
    norm_bias = getattr(layernorm, "bias", None)

    for linear in linears:
        if linear is None:
            continue
        if norm_bias is not None:
            original = linear.weight.detach().to(dtype=torch.float64)
            correction = original @ norm_bias.detach().to(device=original.device, dtype=torch.float64)
            if linear.bias is None:
                linear.bias = torch.nn.Parameter(
                    torch.zeros(linear.out_features, device=linear.weight.device, dtype=linear.weight.dtype)
                )
            linear.bias.data.add_(correction.to(device=linear.bias.device, dtype=linear.bias.dtype))
        _scale_last_dim_(linear.weight, scale, chunk_rows=chunk_rows)


def _fuse_norm_into_parameter(
    layernorm: torch.nn.Module,
    parameter: torch.nn.Parameter,
    *,
    chunk_rows: int,
) -> None:
    if getattr(layernorm, "bias", None) is not None:
        raise ValueError("Packed Laguna expert/router fusion does not support a norm bias")
    _scale_last_dim_(parameter, _get_norm_scale(layernorm), chunk_rows=chunk_rows)


def _expert_containers(experts) -> list[torch.nn.Module]:
    num_experts = int(getattr(experts, "num_experts", 0) or 0)
    containers = []
    for index in range(num_experts):
        try:
            container = experts[index]
        except (IndexError, KeyError, TypeError):
            container = getattr(experts, str(index), None)
        if container is not None:
            containers.append(container)
    return containers


@torch.no_grad()
def fuse_laguna_norms(model, *, chunk_rows: int = 1024) -> None:
    """Fuse Laguna residual RMSNorm scales into every direct consumer."""
    layers, _, final_norm, lm_head = _get_model_parts(model)

    _fuse_norm_into_linears(final_norm, [lm_head], chunk_rows=chunk_rows)
    _reset_norm(final_norm)

    for layer in layers:
        input_norm = layer.input_layernorm
        post_norm = layer.post_attention_layernorm
        attention = layer.self_attn

        attention_inputs = [attention.q_proj, attention.k_proj, attention.v_proj]
        if hasattr(attention, "g_proj"):
            attention_inputs.append(attention.g_proj)
        _fuse_norm_into_linears(input_norm, attention_inputs, chunk_rows=chunk_rows)

        mlp = layer.mlp
        if hasattr(mlp, "experts"):
            if hasattr(mlp, "gate") and hasattr(mlp.gate, "weight"):
                _fuse_norm_into_parameter(post_norm, mlp.gate.weight, chunk_rows=chunk_rows)

            shared_expert = getattr(mlp, "shared_expert", None)
            if shared_expert is not None:
                _fuse_norm_into_linears(
                    post_norm,
                    [shared_expert.gate_proj, shared_expert.up_proj],
                    chunk_rows=chunk_rows,
                )

            experts = mlp.experts
            packed_gate_up = getattr(experts, "gate_up_proj", None)
            if isinstance(packed_gate_up, torch.nn.Parameter):
                _fuse_norm_into_parameter(post_norm, packed_gate_up, chunk_rows=chunk_rows)
            else:
                for expert in _expert_containers(experts):
                    _fuse_norm_into_linears(
                        post_norm,
                        [expert.gate_proj, expert.up_proj],
                        chunk_rows=chunk_rows,
                    )
        else:
            _fuse_norm_into_linears(
                post_norm,
                [mlp.gate_proj, mlp.up_proj],
                chunk_rows=chunk_rows,
            )

        _reset_norm(input_norm)
        _reset_norm(post_norm)


def _right_multiply_last_dim(
    tensor: torch.Tensor,
    matrix: torch.Tensor,
    *,
    device: torch.device,
    chunk_rows: int,
    compute_dtype: torch.dtype,
) -> torch.Tensor:
    """Return ``tensor @ matrix`` along the last dimension, chunked by rows."""
    if tensor.device.type == "meta":
        raise RuntimeError("Cannot rotate a meta-device tensor")
    if tensor.shape[-1] != matrix.shape[0]:
        raise ValueError(f"Cannot right-rotate {tuple(tensor.shape)} with {tuple(matrix.shape)}")

    source_device = tensor.device
    source_dtype = tensor.dtype
    original_shape = tuple(tensor.shape)
    flat = tensor.detach().reshape(-1, original_shape[-1])
    result = torch.empty_like(flat, device=source_device)
    matrix_device = matrix.to(device=device, dtype=compute_dtype)

    for start in range(0, flat.shape[0], chunk_rows):
        stop = min(start + chunk_rows, flat.shape[0])
        chunk = flat[start:stop].to(device=device, dtype=compute_dtype)
        rotated = chunk @ matrix_device
        result[start:stop].copy_(rotated.to(device=source_device, dtype=source_dtype))

    return result.reshape(original_shape).contiguous()


def _left_multiply_output_dim(
    tensor: torch.Tensor,
    matrix: torch.Tensor,
    *,
    device: torch.device,
    chunk_columns: int,
    matrix_batch: int,
    compute_dtype: torch.dtype,
) -> torch.Tensor:
    """Return ``matrix @ tensor`` where tensor is ``[..., output_dim, input_dim]``."""
    if tensor.device.type == "meta":
        raise RuntimeError("Cannot rotate a meta-device tensor")
    if tensor.ndim < 2 or tensor.shape[-2] != matrix.shape[-1]:
        raise ValueError(f"Cannot left-rotate {tuple(tensor.shape)} with {tuple(matrix.shape)}")

    source_device = tensor.device
    source_dtype = tensor.dtype
    original_shape = tuple(tensor.shape)
    output_dim, input_dim = original_shape[-2:]
    batch = math.prod(original_shape[:-2]) or 1
    flat = tensor.detach().reshape(batch, output_dim, input_dim)
    result = torch.empty_like(flat, device=source_device)
    matrix_device = matrix.to(device=device, dtype=compute_dtype)

    for batch_start in range(0, batch, matrix_batch):
        batch_stop = min(batch_start + matrix_batch, batch)
        for column_start in range(0, input_dim, chunk_columns):
            column_stop = min(column_start + chunk_columns, input_dim)
            chunk = flat[batch_start:batch_stop, :, column_start:column_stop].to(
                device=device,
                dtype=compute_dtype,
            )
            rotated = torch.matmul(matrix_device, chunk)
            result[batch_start:batch_stop, :, column_start:column_stop].copy_(
                rotated.to(device=source_device, dtype=source_dtype)
            )

    return result.reshape(original_shape).contiguous()


def _rotate_linear_input(
    linear: torch.nn.Linear,
    matrix: torch.Tensor,
    **kwargs,
) -> None:
    linear.weight.data = _right_multiply_last_dim(linear.weight, matrix, **kwargs)


def _rotate_linear_output(
    linear: torch.nn.Linear,
    matrix: torch.Tensor,
    *,
    device: torch.device,
    chunk_columns: int,
    matrix_batch: int,
    compute_dtype: torch.dtype,
) -> None:
    linear.weight.data = _left_multiply_output_dim(
        linear.weight,
        matrix.T,
        device=device,
        chunk_columns=chunk_columns,
        matrix_batch=matrix_batch,
        compute_dtype=compute_dtype,
    )
    if linear.bias is not None:
        bias = linear.bias.detach().reshape(-1, 1)
        rotated = _left_multiply_output_dim(
            bias,
            matrix.T,
            device=device,
            chunk_columns=1,
            matrix_batch=1,
            compute_dtype=compute_dtype,
        )
        linear.bias.data = rotated.reshape_as(linear.bias)


def _normalized_hadamard(size: int, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    if size <= 0 or size & (size - 1):
        raise ValueError(f"Head-wise Hadamard size must be a power of two, got {size}")
    matrix = torch.ones((1, 1), device=device, dtype=dtype)
    while matrix.shape[0] < size:
        matrix = torch.cat(
            (
                torch.cat((matrix, matrix), dim=1),
                torch.cat((matrix, -matrix), dim=1),
            ),
            dim=0,
        ) / math.sqrt(2.0)
    return matrix


def _hadamard_12(*, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    # Paley order-12 matrix used by GPTQModel's QuaRot implementation.  Keeping
    # this small basis local avoids importing the full GPTQModel runtime merely
    # to construct Laguna's 3072-dimensional residual rotation.
    matrix = torch.tensor(
        [
            [+1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            [+1, +1, -1, +1, -1, -1, -1, +1, +1, +1, -1, +1],
            [+1, +1, +1, -1, +1, -1, -1, -1, +1, +1, +1, -1],
            [+1, -1, +1, +1, -1, +1, -1, -1, -1, +1, +1, +1],
            [+1, +1, -1, +1, +1, -1, +1, -1, -1, -1, +1, +1],
            [+1, +1, +1, -1, +1, +1, -1, +1, -1, -1, -1, +1],
            [+1, +1, +1, +1, -1, +1, +1, -1, +1, -1, -1, -1],
            [+1, -1, +1, +1, +1, -1, +1, +1, -1, +1, -1, -1],
            [+1, -1, -1, +1, +1, +1, -1, +1, +1, -1, +1, -1],
            [+1, -1, -1, -1, +1, +1, +1, -1, +1, +1, -1, +1],
            [+1, +1, -1, -1, -1, +1, +1, +1, -1, +1, +1, -1],
            [+1, -1, +1, -1, -1, -1, +1, +1, +1, -1, +1, +1],
        ],
        device=device,
        dtype=dtype,
    )
    return matrix


def _sylvester_hadamard(size: int, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    if size <= 0 or size & (size - 1):
        raise ValueError(f"Sylvester Hadamard size must be a power of two, got {size}")
    matrix = torch.ones((1, 1), device=device, dtype=dtype)
    while matrix.shape[0] < size:
        matrix = torch.cat(
            (
                torch.cat((matrix, matrix), dim=1),
                torch.cat((matrix, -matrix), dim=1),
            ),
            dim=0,
        )
    return matrix


def _random_residual_hadamard(
    size: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    signs = torch.randint(0, 2, (size,), dtype=torch.int64).mul_(2).sub_(1)
    if size > 0 and not size & (size - 1):
        hadamard = _sylvester_hadamard(size, device=device, dtype=dtype)
    elif size % 12 == 0 and not (size // 12) & (size // 12 - 1):
        sylvester = _sylvester_hadamard(size // 12, device=device, dtype=dtype)
        hadamard = torch.kron(
            _hadamard_12(device=device, dtype=dtype),
            sylvester,
        ).T.contiguous()
    else:
        raise ValueError(
            "Laguna residual Hadamard requires a power-of-two hidden size or "
            f"12 times a power of two, got {size}"
        )
    # Match GPTQModel's default-dtype normalization after the integer
    # butterfly so a given seed constructs the same matrix.
    normalization = torch.tensor(size).sqrt().to(device=device, dtype=dtype)
    return signs.to(device=device, dtype=dtype).unsqueeze(1) * hadamard / normalization


def _rotate_attention_v_o_headwise(
    attention: torch.nn.Module,
    *,
    device: torch.device,
    chunk_rows: int,
    chunk_columns: int,
    matrix_batch: int,
    compute_dtype: torch.dtype,
) -> None:
    """Apply the exact per-head V/O Hadamard used by the repository QuaRot path."""
    head_dim = int(attention.head_dim)
    value_width = attention.v_proj.out_features
    output_width = attention.o_proj.in_features
    if value_width % head_dim or output_width % head_dim:
        raise ValueError(
            f"Laguna attention widths must be divisible by head_dim={head_dim}: "
            f"v={value_width}, o={output_width}"
        )

    if hasattr(attention, "g_proj") and not bool(getattr(attention, "gate_per_head", False)):
        raise ValueError(
            "Head-wise V/O rotation is only equivalent for Laguna per-head gating; "
            "disable head-wise rotation for per-element gating"
        )

    hadamard = _normalized_hadamard(head_dim, device=device, dtype=compute_dtype)
    value_weight = attention.v_proj.weight.detach().reshape(-1, head_dim, attention.v_proj.in_features)
    value_weight = _left_multiply_output_dim(
        value_weight,
        hadamard,
        device=device,
        chunk_columns=chunk_columns,
        matrix_batch=matrix_batch,
        compute_dtype=compute_dtype,
    )
    attention.v_proj.weight.data = value_weight.reshape_as(attention.v_proj.weight)

    if attention.v_proj.bias is not None:
        value_bias = attention.v_proj.bias.detach().reshape(-1, head_dim, 1)
        value_bias = _left_multiply_output_dim(
            value_bias,
            hadamard,
            device=device,
            chunk_columns=1,
            matrix_batch=matrix_batch,
            compute_dtype=compute_dtype,
        )
        attention.v_proj.bias.data = value_bias.reshape_as(attention.v_proj.bias)

    output_weight = attention.o_proj.weight.detach().reshape(
        attention.o_proj.out_features,
        -1,
        head_dim,
    )
    output_weight = _right_multiply_last_dim(
        output_weight,
        hadamard,
        device=device,
        chunk_rows=chunk_rows,
        compute_dtype=compute_dtype,
    )
    attention.o_proj.weight.data = output_weight.reshape_as(attention.o_proj.weight)


def _generate_residual_rotation(
    hidden_size: int,
    *,
    mode: str,
    device: torch.device,
    compute_dtype: torch.dtype,
) -> torch.Tensor:
    if mode != "hadamard":
        raise ValueError(f"Unsupported Laguna rotation mode: {mode}")
    return _random_residual_hadamard(hidden_size, device=device, dtype=compute_dtype)


@torch.no_grad()
def rotate_laguna_weights(
    model,
    matrix: torch.Tensor,
    *,
    device: str | torch.device,
    headwise_v_o: bool = True,
    chunk_rows: int = 1024,
    chunk_columns: int = 1024,
    matrix_batch: int = 4,
    compute_dtype: torch.dtype = torch.float64,
) -> None:
    """Rotate all Laguna weights after norm fusion."""
    device = torch.device(device)
    layers, embedding, _, lm_head = _get_model_parts(model)

    input_kwargs = {
        "device": device,
        "chunk_rows": chunk_rows,
        "compute_dtype": compute_dtype,
    }
    output_kwargs = {
        "device": device,
        "chunk_columns": chunk_columns,
        "matrix_batch": matrix_batch,
        "compute_dtype": compute_dtype,
    }

    embedding.weight.data = _right_multiply_last_dim(embedding.weight, matrix, **input_kwargs)
    _rotate_linear_input(lm_head, matrix, **input_kwargs)

    for index, layer in enumerate(layers):
        print(f"[QuaRot] Rotating Laguna layer {index + 1}/{len(layers)}", flush=True)
        attention = layer.self_attn
        for projection in (attention.q_proj, attention.k_proj, attention.v_proj):
            _rotate_linear_input(projection, matrix, **input_kwargs)
        if hasattr(attention, "g_proj"):
            _rotate_linear_input(attention.g_proj, matrix, **input_kwargs)
        _rotate_linear_output(attention.o_proj, matrix, **output_kwargs)

        if headwise_v_o:
            _rotate_attention_v_o_headwise(
                attention,
                device=device,
                chunk_rows=chunk_rows,
                chunk_columns=chunk_columns,
                matrix_batch=matrix_batch,
                compute_dtype=compute_dtype,
            )

        mlp = layer.mlp
        if hasattr(mlp, "experts"):
            if hasattr(mlp, "gate") and hasattr(mlp.gate, "weight"):
                mlp.gate.weight.data = _right_multiply_last_dim(
                    mlp.gate.weight,
                    matrix,
                    **input_kwargs,
                )

            shared_expert = getattr(mlp, "shared_expert", None)
            if shared_expert is not None:
                _rotate_linear_input(shared_expert.gate_proj, matrix, **input_kwargs)
                _rotate_linear_input(shared_expert.up_proj, matrix, **input_kwargs)
                _rotate_linear_output(shared_expert.down_proj, matrix, **output_kwargs)

            experts = mlp.experts
            packed_gate_up = getattr(experts, "gate_up_proj", None)
            packed_down = getattr(experts, "down_proj", None)
            if isinstance(packed_gate_up, torch.nn.Parameter):
                experts.gate_up_proj.data = _right_multiply_last_dim(
                    packed_gate_up,
                    matrix,
                    **input_kwargs,
                )
                experts.down_proj.data = _left_multiply_output_dim(
                    packed_down,
                    matrix.T,
                    **output_kwargs,
                )
            else:
                for expert in _expert_containers(experts):
                    _rotate_linear_input(expert.gate_proj, matrix, **input_kwargs)
                    _rotate_linear_input(expert.up_proj, matrix, **input_kwargs)
                    _rotate_linear_output(expert.down_proj, matrix, **output_kwargs)
        else:
            _rotate_linear_input(mlp.gate_proj, matrix, **input_kwargs)
            _rotate_linear_input(mlp.up_proj, matrix, **input_kwargs)
            _rotate_linear_output(mlp.down_proj, matrix, **output_kwargs)


@torch.no_grad()
def apply_quarot(
    model,
    *,
    mode: str = "hadamard",
    device: str | torch.device = "cuda:0",
    seed: int = 42,
    headwise_v_o: bool = True,
    chunk_rows: int = 1024,
    chunk_columns: int = 1024,
    matrix_batch: int = 4,
    compute_dtype: torch.dtype = torch.float64,
):
    """Fuse Laguna norms and apply the complete repository-style QuaRot transform."""
    existing = getattr(model.config, QUAROT_CONFIG_KEY, None)
    if isinstance(existing, dict) and existing.get("applied"):
        raise RuntimeError("This Laguna model already advertises an applied QuaRot transform")

    validate_laguna_quarot(model, headwise_v_o=headwise_v_o)

    device = torch.device(device)
    hidden_size = int(model.config.hidden_size)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    print(f"[QuaRot] Building {mode} residual rotation: hidden_size={hidden_size}, device={device}")
    matrix = _generate_residual_rotation(
        hidden_size,
        mode=mode,
        device=device,
        compute_dtype=compute_dtype,
    )

    print("[QuaRot] Fusing Laguna RMSNorm scales")
    fuse_laguna_norms(model, chunk_rows=chunk_rows)
    rotate_laguna_weights(
        model,
        matrix,
        device=device,
        headwise_v_o=headwise_v_o,
        chunk_rows=chunk_rows,
        chunk_columns=chunk_columns,
        matrix_batch=matrix_batch,
        compute_dtype=compute_dtype,
    )

    metadata = {
        "applied": True,
        "method": "quarot",
        "mode": mode,
        "seed": int(seed),
        "headwise_v_o": bool(headwise_v_o),
        "compute_dtype": str(compute_dtype).removeprefix("torch."),
        "implementation": "scripts_laguna.rotation_utils",
    }
    setattr(model.config, QUAROT_CONFIG_KEY, metadata)
    setattr(model, "_quarot_applied", True)
    print(f"[QuaRot] Done: {metadata}")
    return model, matrix
