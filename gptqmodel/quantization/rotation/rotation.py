# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

# Adapted from https://github.com/spcl/QuaRot/blob/main/fake_quant/rotation_utils.py
import typing

import torch
from transformers import PreTrainedModel

from ...utils.logger import setup_logger
from ...utils.model import get_module_by_name_prefix
from ...utils.torch import torch_empty_cache
from .hadamard_utils import apply_exact_had_to_linear, random_hadamard_matrix, random_block_hadamard_matrix


log = setup_logger()

_ADDITIVE_NORM_CLASSES = {
    "Qwen3_5MoeRMSNorm",
    "Qwen3_5RMSNorm",
    "Qwen3NextRMSNorm",
}


def fuse_ln_linear(
        layernorm: torch.nn.Module, linear_layers: typing.Iterable[torch.nn.Linear]
) -> None:
    """
    fuse the linear operations in Layernorm into the adjacent linear blocks.
    """
    for linear in linear_layers:
        linear_dtype = linear.weight.dtype

        # Calculating new weight and bias
        W_ = linear.weight.data.double()
        linear.weight.data = (W_ * layernorm.weight.double()).to(linear_dtype)

        if hasattr(layernorm, "bias"):
            if linear.bias is None:
                linear.bias = torch.nn.Parameter(
                    torch.zeros(linear.out_features, dtype=torch.float64)
                )
            linear.bias.data = linear.bias.data.double() + torch.matmul(
                W_, layernorm.bias.double()
            )
            linear.bias.data = linear.bias.data.to(linear_dtype)


def reset_ln(ln):
    W_norm = ln.weight.data
    ln.weight.data = torch.ones_like(W_norm)
    if hasattr(ln, "bias"):
        ln.bias.data.zero_()


def _get_norm_scale(layernorm: torch.nn.Module) -> torch.Tensor:
    """
    Return the multiplicative scale applied by a norm module.

    Most RMSNorm/LayerNorm variants use `weight * x`, but Qwen3.5-MoE RMSNorm
    uses `(1 + weight) * x`.
    """
    weight = layernorm.weight.data.double()
    if layernorm.__class__.__name__ in _ADDITIVE_NORM_CLASSES:
        return 1.0 + weight
    return weight


def _reset_norm_for_fusion(layernorm: torch.nn.Module) -> None:
    """
    Reset a norm module to the identity transform after fusing into linear ops.
    """
    if layernorm.__class__.__name__ in _ADDITIVE_NORM_CLASSES:
        layernorm.weight.data.zero_()
        if hasattr(layernorm, "bias"):
            layernorm.bias.data.zero_()
        return
    reset_ln(layernorm)


def _fuse_norm_into_linears(layernorm: torch.nn.Module, linear_layers: typing.Iterable[torch.nn.Linear]) -> None:
    scale = _get_norm_scale(layernorm)

    for linear in linear_layers:
        linear_dtype = linear.weight.dtype
        W_ = linear.weight.data.double()
        linear.weight.data = (W_ * scale).to(linear_dtype)

        if hasattr(layernorm, "bias"):
            if linear.bias is None:
                linear.bias = torch.nn.Parameter(torch.zeros(linear.out_features, dtype=torch.float64))
            linear.bias.data = linear.bias.data.double() + torch.matmul(W_, layernorm.bias.double())
            linear.bias.data = linear.bias.data.to(linear_dtype)


def _fuse_norm_into_tensor_last_dim(layernorm: torch.nn.Module, tensor: torch.nn.Parameter) -> None:
    scale = _get_norm_scale(layernorm)
    dtype = tensor.dtype
    tensor.data = (tensor.data.double() * scale).to(dtype)


def _fuse_pre_head_norm_if_present(
    model: PreTrainedModel,
    pre_lm_head_norm_module_name: str,
    lm_head_name: str,
) -> None:
    if not pre_lm_head_norm_module_name:
        return

    pre_head_layernorm, _ = get_module_by_name_prefix(model, pre_lm_head_norm_module_name)
    if pre_head_layernorm is None:
        log.warn(
            f"Rotation: skip pre-lm-head norm fusion because `{pre_lm_head_norm_module_name}` was not found."
        )
        return

    lm_head, _ = get_module_by_name_prefix(model, lm_head_name)
    _fuse_norm_into_linears(pre_head_layernorm, [lm_head])
    _reset_norm_for_fusion(pre_head_layernorm)


def fuse_merger_linear(
    layernorm: torch.nn.Module, linear_layers: typing.Iterable[torch.nn.Linear]
) -> None:
    """
    fuse the linear operations in Layernorm into the adjacent linear blocks.
    """
    for linear in linear_layers:
        linear_dtype = linear.weight.dtype

        # Calculating new weight and bias
        W_ = linear.weight.data.double()
        w_o, w_i = W_.shape
        size = layernorm.weight.shape[0]
        linear.weight.data = (
            (W_.view(w_o, -1, size) * layernorm.weight.double())
            .to(linear_dtype)
            .view(w_o, w_i)
        )

        if hasattr(layernorm, "bias"):
            if linear.bias is None:
                linear.bias = torch.nn.Parameter(
                    torch.zeros(linear.out_features,
                                dtype=torch.float64).to(W_)
                )
            linear.bias.data = linear.bias.data.double() + torch.matmul(
                W_.view(w_o, -1, size), layernorm.bias.double()
            ).sum(dim=-1)
            linear.bias.data = linear.bias.data.to(linear_dtype)

    layernorm.weight.data = torch.ones_like(layernorm.weight.data)
    if hasattr(layernorm, "bias"):
        layernorm.bias.data = torch.zeros_like(layernorm.bias.data)


def fuse_layer_norms(model: PreTrainedModel, pre_lm_head_norm_module_name: str, layers_node: str, lm_head_name: str):
    layers , _ = get_module_by_name_prefix(model, layers_node)

    # Fuse the linear operations in Layernorm into the adjacent linear blocks.
    for layer in layers:
        # fuse the input layernorms into the linear layers
        fuse_ln_linear(
            layer.post_attention_layernorm, [layer.mlp.up_proj, layer.mlp.gate_proj]
        )
        fuse_ln_linear(
            layer.input_layernorm,
            [
                layer.self_attn.q_proj,
                layer.self_attn.k_proj,
                layer.self_attn.v_proj,
            ],
        )
        reset_ln(layer.post_attention_layernorm)
        reset_ln(layer.input_layernorm)

    pre_head_layernorm, _ = get_module_by_name_prefix(model, pre_lm_head_norm_module_name)
    lm_head, _ = get_module_by_name_prefix(model, lm_head_name)

    fuse_ln_linear(pre_head_layernorm, [lm_head])
    reset_ln(pre_head_layernorm)
    return model


def fuse_layer_norms_moe(model: PreTrainedModel, pre_lm_head_norm_module_name: str, layers_node: str, lm_head_name: str):
    layers , _ = get_module_by_name_prefix(model, layers_node)

    # Fuse the linear operations in Layernorm into the adjacent linear blocks.
    for layer in layers:
        # fuse the input layernorms into the linear layers
        fuse_ln_linear(
                layer.post_attention_layernorm, [layer.mlp.gate]
            )
        for i in range(layer.mlp.n_routed_experts):
            fuse_ln_linear(
                layer.post_attention_layernorm, [layer.mlp.experts[i].gate_proj, layer.mlp.experts[i].up_proj]
            )
        if hasattr(layer.mlp, 'shared_experts'):
            fuse_ln_linear(
                layer.post_attention_layernorm, [layer.mlp.shared_experts.gate_proj, layer.mlp.shared_experts.up_proj]
            )
        fuse_ln_linear(
            layer.input_layernorm,
            [
                layer.self_attn.q_proj,
                layer.self_attn.k_proj,
                layer.self_attn.v_proj,
            ],
        )
        reset_ln(layer.post_attention_layernorm)
        reset_ln(layer.input_layernorm)

    pre_head_layernorm, _ = get_module_by_name_prefix(model, pre_lm_head_norm_module_name)
    lm_head, _ = get_module_by_name_prefix(model, lm_head_name)

    fuse_ln_linear(pre_head_layernorm, [lm_head])
    reset_ln(pre_head_layernorm)
    return model


def fuse_layer_norms_qwen3_5_moe(
    model: PreTrainedModel,
    pre_lm_head_norm_module_name: str,
    layers_node: str,
    lm_head_name: str,
):
    """
    Qwen3.5-MoE specific LN fusion.

    Differences vs generic MoE fusion:
    - RMSNorm uses `(1 + weight)` scaling.
    - Experts may be stored as packed tensors (`gate_up_proj`, `down_proj`) rather
      than per-expert Linear modules.
    - Token mixer may be either `self_attn` or `linear_attn`.
    """
    layers, _ = get_module_by_name_prefix(model, layers_node)

    for layer in layers:
        post_ln = layer.post_attention_layernorm
        input_ln = layer.input_layernorm

        # MLP / router fusion (post-attention norm)
        mlp = layer.mlp

        if hasattr(mlp, "gate") and hasattr(mlp.gate, "weight"):
            dtype = mlp.gate.weight.dtype
            W_ = mlp.gate.weight.data.double()
            mlp.gate.weight.data = (W_ * _get_norm_scale(post_ln)).to(dtype)

        if hasattr(mlp, "shared_expert"):
            _fuse_norm_into_linears(
                post_ln,
                [mlp.shared_expert.gate_proj, mlp.shared_expert.up_proj],
            )
        if hasattr(mlp, "shared_expert_gate"):
            _fuse_norm_into_linears(post_ln, [mlp.shared_expert_gate])

        experts = getattr(mlp, "experts", None)
        if experts is not None:
            if hasattr(experts, "gate_up_proj"):
                _fuse_norm_into_tensor_last_dim(post_ln, experts.gate_up_proj)
            elif hasattr(experts, "gate_proj") and hasattr(experts, "up_proj"):
                for i in range(len(experts.gate_proj)):
                    _fuse_norm_into_linears(post_ln, [experts.gate_proj[i], experts.up_proj[i]])
            else:
                # defuser per-expert container format: experts[i].gate_proj / up_proj
                num_experts = getattr(experts, "num_experts", None)
                if num_experts is not None:
                    for i in range(num_experts):
                        expert = experts[i]
                        linears = []
                        if hasattr(expert, "gate_proj"):
                            linears.append(expert.gate_proj)
                        if hasattr(expert, "up_proj"):
                            linears.append(expert.up_proj)
                        if linears:
                            _fuse_norm_into_linears(post_ln, linears)

        # Token-mixer input fusion (input norm)
        attn_inputs = []
        if hasattr(layer, "self_attn"):
            attn_inputs.extend([layer.self_attn.q_proj, layer.self_attn.k_proj, layer.self_attn.v_proj])
        if hasattr(layer, "linear_attn"):
            attn_inputs.extend(
                [
                    layer.linear_attn.in_proj_qkv,
                    layer.linear_attn.in_proj_z,
                    layer.linear_attn.in_proj_b,
                    layer.linear_attn.in_proj_a,
                ]
            )
        if attn_inputs:
            _fuse_norm_into_linears(input_ln, attn_inputs)

        _reset_norm_for_fusion(post_ln)
        _reset_norm_for_fusion(input_ln)

    _fuse_pre_head_norm_if_present(model, pre_lm_head_norm_module_name, lm_head_name)

    return model


def fuse_layer_norms_qwen3_5(
    model: PreTrainedModel,
    pre_lm_head_norm_module_name: str,
    layers_node: str,
    lm_head_name: str,
):
    """
    Qwen3.5 dense specific LN fusion.

    Qwen3.5 RMSNorm uses `(1 + weight)` scaling, and decoder layers may be either
    full-attention (`self_attn`) or linear-attention (`linear_attn`).
    """
    layers, _ = get_module_by_name_prefix(model, layers_node)

    for layer in layers:
        post_ln = layer.post_attention_layernorm
        input_ln = layer.input_layernorm

        _fuse_norm_into_linears(
            post_ln,
            [layer.mlp.gate_proj, layer.mlp.up_proj],
        )

        attn_inputs = []
        if hasattr(layer, "self_attn"):
            attn_inputs.extend([layer.self_attn.q_proj, layer.self_attn.k_proj, layer.self_attn.v_proj])
        if hasattr(layer, "linear_attn"):
            attn_inputs.extend(
                [
                    layer.linear_attn.in_proj_qkv,
                    layer.linear_attn.in_proj_z,
                    layer.linear_attn.in_proj_b,
                    layer.linear_attn.in_proj_a,
                ]
            )
        if attn_inputs:
            _fuse_norm_into_linears(input_ln, attn_inputs)

        _reset_norm_for_fusion(post_ln)
        _reset_norm_for_fusion(input_ln)

    _fuse_pre_head_norm_if_present(model, pre_lm_head_norm_module_name, lm_head_name)

    return model


def fuse_layer_norms_qwen3_next(
    model: PreTrainedModel,
    pre_lm_head_norm_module_name: str,
    layers_node: str,
    lm_head_name: str,
):
    """
    Qwen3-Next specific LN fusion.

    Qwen3-Next uses `(1 + weight)` RMSNorm scaling, mixes full and linear token
    mixers by layer, and alternates between dense and sparse-MoE MLP blocks.
    """
    layers, _ = get_module_by_name_prefix(model, layers_node)

    for layer in layers:
        post_ln = layer.post_attention_layernorm
        input_ln = layer.input_layernorm
        mlp = layer.mlp

        mlp_inputs = []
        if hasattr(mlp, "gate_proj") and hasattr(mlp, "up_proj"):
            mlp_inputs.extend([mlp.gate_proj, mlp.up_proj])
        else:
            if hasattr(mlp, "gate") and hasattr(mlp.gate, "weight"):
                gate_dtype = mlp.gate.weight.dtype
                gate_w = mlp.gate.weight.data.double()
                mlp.gate.weight.data = (gate_w * _get_norm_scale(post_ln)).to(gate_dtype)
            if hasattr(mlp, "shared_expert"):
                mlp_inputs.extend([mlp.shared_expert.gate_proj, mlp.shared_expert.up_proj])
            if hasattr(mlp, "shared_expert_gate"):
                mlp_inputs.append(mlp.shared_expert_gate)
            experts = getattr(mlp, "experts", None)
            if experts is not None:
                if hasattr(experts, "gate_up_proj"):
                    _fuse_norm_into_tensor_last_dim(post_ln, experts.gate_up_proj)
                else:
                    for expert in experts:
                        if hasattr(expert, "gate_proj") and hasattr(expert, "up_proj"):
                            mlp_inputs.extend([expert.gate_proj, expert.up_proj])

        if mlp_inputs:
            _fuse_norm_into_linears(post_ln, mlp_inputs)

        attn_inputs = []
        if hasattr(layer, "self_attn"):
            attn_inputs.extend([layer.self_attn.q_proj, layer.self_attn.k_proj, layer.self_attn.v_proj])
        if hasattr(layer, "linear_attn"):
            attn_inputs.extend([layer.linear_attn.in_proj_qkvz, layer.linear_attn.in_proj_ba])
        if attn_inputs:
            _fuse_norm_into_linears(input_ln, attn_inputs)

        _reset_norm_for_fusion(post_ln)
        _reset_norm_for_fusion(input_ln)

    _fuse_pre_head_norm_if_present(model, pre_lm_head_norm_module_name, lm_head_name)

    return model


def bake_mean_into_linear_conv(linear: torch.nn.Linear, embed_dims) -> None:
    """
    This function takes a linear layer and subtracts the means from the
    weights and biases. This will result in the linear layer performing
    the mean substitution which is usually done inside layernorm.
    """
    linear_dtype = linear.weight.dtype
    weight_shape = linear.weight.data.shape
    W_ = linear.weight.data.double().view(embed_dims, -1)
    linear.weight.data = W_ - W_.mean(dim=0, keepdim=True)
    linear.weight.data = linear.weight.data.to(linear_dtype).view(weight_shape)
    if linear.bias is not None:
        b_ = linear.bias.data.double()
        linear.bias.data = b_ - b_.mean()
        linear.bias.data = linear.bias.data.to(linear_dtype)


def bake_mean_into_linear(linear: torch.nn.Linear) -> None:
    """
    This function takes a linear layer and subtracts the means from the
    weights and biases. This will result in the linear layer performing
    the mean substitution which is usually done inside layernorm.
    """
    linear_dtype = linear.weight.dtype
    W_ = linear.weight.data.double()
    linear.weight.data = W_ - W_.mean(dim=-2, keepdim=True)
    linear.weight.data = linear.weight.data.to(linear_dtype)
    if linear.bias is not None:
        b_ = linear.bias.data.double()
        linear.bias.data = b_ - b_.mean()
        linear.bias.data = linear.bias.data.to(linear_dtype)


def fuse_deepstack_linear(model):
    for layer in model.visual.deepstack_merger_list:
        fuse_merger_linear(layer.norm, [layer.linear_fc1])


def fuse_vision_layer_norms_qwen2_5_vl(model: PreTrainedModel, pre_lm_head_norm_module_name: str, layers_node: str, lm_head_name: str):
    model = model.eval()
    for layer in model.visual.blocks:
        fuse_ln_linear(layer.norm1, [layer.attn.qkv])
        reset_ln(layer.norm1)
        fuse_ln_linear(
            layer.norm2, [layer.mlp.gate_proj, layer.mlp.up_proj])
        reset_ln(layer.norm2)
    fuse_merger_linear(model.visual.merger.ln_q, [
                    model.visual.merger.mlp[0]])
    reset_ln(model.visual.merger.ln_q)
    return model


def fuse_vision_layer_norms_qwen3_vl(model: PreTrainedModel, pre_lm_head_norm_module_name: str, layers_node: str, lm_head_name: str):
    model = model.eval()
    bake_mean_into_linear_conv(model.visual.patch_embed.proj, model.visual.blocks[0].attn.qkv.in_features)
    for W in [model.visual.pos_embed]:
        W_ = W.weight.data.double()
        W.weight.data = (W_ - W_.mean(dim=-1, keepdim=True)
                        ).to(W.weight.data.dtype)
    for layer in model.visual.blocks:
        fuse_ln_linear(layer.norm1, [layer.attn.qkv])
        reset_ln(layer.norm1)
        bake_mean_into_linear(layer.attn.proj)
        fuse_ln_linear(layer.norm2, [layer.mlp.linear_fc1])
        reset_ln(layer.norm2)
        bake_mean_into_linear(layer.mlp.linear_fc2)
    fuse_merger_linear(model.visual.merger.norm, [
                    model.visual.merger.linear_fc1])
    fuse_deepstack_linear(model)
    reset_ln(model.visual.merger.norm)
    return model


def _get_qwen3_5_multimodal_root(model: PreTrainedModel):
    root = getattr(model, "model", model)
    if not hasattr(root, "visual") or not hasattr(root, "language_model"):
        raise AttributeError(
            f"Expected Qwen3.5 multimodal root to expose `visual` and `language_model`, got {type(root).__name__}"
        )
    return root


def _get_qwen3_5_visual(model: PreTrainedModel):
    return _get_qwen3_5_multimodal_root(model).visual


def _get_qwen3_5_language_model(model: PreTrainedModel):
    return _get_qwen3_5_multimodal_root(model).language_model


def fuse_vision_layer_norms_qwen3_5_vl(model: PreTrainedModel, pre_lm_head_norm_module_name: str, layers_node: str, lm_head_name: str):
    model = model.eval()
    visual = _get_qwen3_5_visual(model)

    bake_mean_into_linear_conv(visual.patch_embed.proj, visual.blocks[0].attn.qkv.in_features)
    for W in [visual.pos_embed]:
        W_ = W.weight.data.double()
        W.weight.data = (W_ - W_.mean(dim=-1, keepdim=True)).to(W.weight.data.dtype)

    for layer in visual.blocks:
        fuse_ln_linear(layer.norm1, [layer.attn.qkv])
        reset_ln(layer.norm1)
        bake_mean_into_linear(layer.attn.proj)
        fuse_ln_linear(layer.norm2, [layer.mlp.linear_fc1])
        reset_ln(layer.norm2)
        bake_mean_into_linear(layer.mlp.linear_fc2)

    fuse_merger_linear(visual.merger.norm, [visual.merger.linear_fc1])
    reset_ln(visual.merger.norm)

    for layer in getattr(visual, "deepstack_merger_list", []):
        if hasattr(layer, "norm") and hasattr(layer, "linear_fc1"):
            fuse_merger_linear(layer.norm, [layer.linear_fc1])
            reset_ln(layer.norm)

    return model


def random_orthogonal_matrix(size, device):
    """
    Generate a random orthogonal matrix of the specified size.
    First, we generate a random matrix with entries from a standard distribution.
    Then, we use QR decomposition to obtain an orthogonal matrix.
    Finally, we multiply by a diagonal matrix with diag r to adjust the signs.

    Args:
    size (int): The size of the matrix (size x size).

    Returns:
    torch.Tensor: An orthogonal matrix of the specified size.
    """
    torch.cuda.empty_cache()
    random_matrix = torch.randn(size, size, dtype=torch.float64).to(device)
    q, r = torch.linalg.qr(random_matrix)
    q *= torch.sign(torch.diag(r)).unsqueeze(0)
    return q


def get_orthogonal_matrix(size, mode, device):
    if mode == "random":
        return random_orthogonal_matrix(size, device)
    elif mode == "hadamard":
        return random_hadamard_matrix(size, device)
    elif mode == "block_hadamard":
        # Use default block size; falls back to full hadamard if not compatible
        return random_block_hadamard_matrix(size, device)
    else:
        raise ValueError(f"Unknown mode {mode}")


def _resolve_attention_head_dim(config, hidden_size: int, num_heads: int) -> int:
    head_dim = getattr(config, "head_dim", None)
    if head_dim is not None:
        return int(head_dim)
    return hidden_size // num_heads


def rotate_conv(layer, Q_v, embed_dims, device):
    dtype = layer.weight.dtype
    weight_shape = layer.weight.data.shape
    layer = layer.to(device)
    layer.weight.data = (
        torch.matmul(Q_v.T, layer.weight.data.double().view(
            embed_dims, -1)).to(dtype=dtype, device=device).view(weight_shape)
    )
    if layer.bias is not None:
        layer.bias.data = torch.matmul(layer.bias.data.double(), Q_v).to(dtype=dtype, device=device)


def rotate_visual_merger(model, Q: torch.Tensor, device) -> None:
    # Rotate the head.
    dtype = model.visual.merger.mlp[0].weight.dtype

    q_shape = Q.shape[0]
    o_shape, i_shape = model.visual.merger.mlp[0].weight.shape

    W_ = model.visual.merger.mlp[0].weight.to(
        dtype=torch.float64, device=device).reshape(o_shape, -1, q_shape)
    model.visual.merger.mlp[0].weight.data = torch.matmul(W_, Q).to(
        dtype=dtype, device=device).reshape(o_shape, i_shape).contiguous()


def rotate_visual_merger_qwen3_vl(model, Q: torch.Tensor, device) -> None:
    # Rotate the head.
    dtype = model.visual.merger.linear_fc1.weight.dtype

    q_shape = Q.shape[0]
    o_shape, i_shape = model.visual.merger.linear_fc1.weight.shape

    W_ = model.visual.merger.linear_fc1.weight.to(
        dtype=torch.float64, device=device).reshape(o_shape, -1, q_shape)
    model.visual.merger.linear_fc1.weight.data = torch.matmul(W_, Q).to(
        dtype=dtype, device=device).reshape(o_shape, i_shape).contiguous()


def rotate_visual_merger_qwen3_5_vl(model, Q: torch.Tensor, device) -> None:
    visual = _get_qwen3_5_visual(model)

    dtype = visual.merger.linear_fc1.weight.dtype
    q_shape = Q.shape[0]
    o_shape, i_shape = visual.merger.linear_fc1.weight.shape

    W_ = visual.merger.linear_fc1.weight.to(
        dtype=torch.float64, device=device).reshape(o_shape, -1, q_shape)
    visual.merger.linear_fc1.weight.data = torch.matmul(W_, Q).to(
        dtype=dtype, device=device).reshape(o_shape, i_shape).contiguous()

    for layer in getattr(visual, "deepstack_merger_list", []):
        if not hasattr(layer, "linear_fc1"):
            continue
        layer_dtype = layer.linear_fc1.weight.dtype
        layer_o_shape, layer_i_shape = layer.linear_fc1.weight.shape
        layer_w = layer.linear_fc1.weight.to(
            dtype=torch.float64, device=device).reshape(layer_o_shape, -1, q_shape)
        layer.linear_fc1.weight.data = torch.matmul(layer_w, Q).to(
            dtype=layer_dtype, device=device).reshape(layer_o_shape, layer_i_shape).contiguous()


def rotate_qwen3_vl_embeddings(model, Q, device) -> None:
    for W in [model.language_model.embed_tokens]:
        dtype  = W.weight.data.dtype
        W_ = W.weight.data.to(device=device, dtype=torch.float64)
        W.weight.data = torch.matmul(W_, Q).to(device="cpu", dtype=dtype)

    W_ = model.visual.merger.linear_fc2.weight.data.to(dtype=torch.float64, device=device)
    model.visual.merger.linear_fc2.weight.data = torch.matmul(Q.T, W_).to(dtype=dtype, device="cpu")
    if model.visual.merger.linear_fc2.bias is not None:
        b = model.visual.merger.linear_fc2.bias.data.to(dtype=torch.float64, device=device)
        model.visual.merger.linear_fc2.bias.data = torch.matmul(b, Q).to(dtype=dtype, device="cpu")

    for layer in model.visual.deepstack_merger_list:
        W_ = layer.linear_fc2.weight.data.to(dtype=torch.float64, device=device)
        layer.linear_fc2.weight.data = torch.matmul(Q.T, W_).to(dtype=dtype, device="cpu")
        if layer.linear_fc2.bias is not None:
            b = layer.linear_fc2.bias.data.to(dtype=torch.float64, device=device)
            layer.linear_fc2.bias.data = torch.matmul(b, Q).to(dtype=dtype, device="cpu")


def rotate_qwen3_5_vl_embeddings(model, Q, device) -> None:
    language_model = _get_qwen3_5_language_model(model)
    visual = _get_qwen3_5_visual(model)

    for W in [language_model.embed_tokens]:
        dtype = W.weight.data.dtype
        W_ = W.weight.data.to(device=device, dtype=torch.float64)
        W.weight.data = torch.matmul(W_, Q).to(device="cpu", dtype=dtype)

    merger_dtype = visual.merger.linear_fc2.weight.data.dtype
    W_ = visual.merger.linear_fc2.weight.data.to(dtype=torch.float64, device=device)
    visual.merger.linear_fc2.weight.data = torch.matmul(Q.T, W_).to(dtype=merger_dtype, device="cpu")
    if visual.merger.linear_fc2.bias is not None:
        b = visual.merger.linear_fc2.bias.data.to(dtype=torch.float64, device=device)
        visual.merger.linear_fc2.bias.data = torch.matmul(b, Q).to(dtype=merger_dtype, device="cpu")

    for layer in getattr(visual, "deepstack_merger_list", []):
        layer_dtype = layer.linear_fc2.weight.data.dtype
        W_ = layer.linear_fc2.weight.data.to(dtype=torch.float64, device=device)
        layer.linear_fc2.weight.data = torch.matmul(Q.T, W_).to(dtype=layer_dtype, device="cpu")
        if layer.linear_fc2.bias is not None:
            b = layer.linear_fc2.bias.data.to(dtype=torch.float64, device=device)
            layer.linear_fc2.bias.data = torch.matmul(b, Q).to(dtype=layer_dtype, device="cpu")


def rotate_qwen2_5_vl_embeddings(model, Q, device) -> None:
    for W in [model.language_model.embed_tokens]:
        dtype  = W.weight.data.dtype
        W_ = W.weight.data.to(device=device, dtype=torch.float64)
        W.weight.data = torch.matmul(W_, Q).to(device="cpu", dtype=dtype)

    W_ = model.visual.merger.mlp[2].weight.data.to(dtype=torch.float64, device=device)
    model.visual.merger.mlp[2].weight.data = torch.matmul(Q.T, W_).to(dtype=dtype, device="cpu")
    if model.visual.merger.mlp[2].bias is not None:
        b = model.visual.merger.mlp[2].bias.data.to(dtype=torch.float64, device=device)
        model.visual.merger.mlp[2].bias.data = torch.matmul(b, Q).to(dtype=dtype, device="cpu")


def rotate_qwen3_vl_vision_embeddings(model, Q, device) -> None:
    for W in [model.visual.pos_embed]:
        dtype  = W.weight.data.dtype
        W_ = W.weight.data.to(device=device, dtype=torch.float64)
        W.weight.data = torch.matmul(W_, Q).to(device="cpu", dtype=dtype)


def rotate_qwen3_5_vl_vision_embeddings(model, Q, device) -> None:
    visual = _get_qwen3_5_visual(model)
    for W in [visual.pos_embed]:
        dtype = W.weight.data.dtype
        W_ = W.weight.data.to(device=device, dtype=torch.float64)
        W.weight.data = torch.matmul(W_, Q).to(device="cpu", dtype=dtype)


def get_embeddings(model) -> list[torch.nn.Module]:
    candidates = [
        ("model", "embed_tokens"),
        ("language_model", "embed_tokens"),
        ("model", "language_model", "embed_tokens"),
    ]

    for path in candidates:
        current = model
        for attr in path:
            current = getattr(current, attr, None)
            if current is None:
                break
        if current is not None:
            return [current]

    raise AttributeError(f"Unable to locate embed_tokens module for rotation on model type {type(model).__name__}")


def rotate_embeddings(model, Q, device) -> None:
    # Rotate the embeddings.
    for W in get_embeddings(model):
        dtype = W.weight.data.dtype
        W_ = W.weight.data.to(device=device, dtype=torch.float64)
        W.weight.data = torch.matmul(W_, Q).to(device="cpu", dtype=dtype)


def rotate_attention_inputs(layer, Q, device) -> None:
    # Rotate the WQ, WK and WV matrices of the self-attention layer.
    # Matrix multiplication order: W@R (W[oc,ic] @ R[ic,ic] -> W_new[oc,ic])
    # This matches weight training mode where we train R using W@R.
    # For activation training: A@R is equivalent to W@R when applied to weights.
    try:
        for W in [layer.self_attn.q_proj, layer.self_attn.k_proj, layer.self_attn.v_proj]:
            dtype = W.weight.dtype
            W_ = W.weight.to(device=device, dtype=torch.float64)
            W.weight.data = torch.matmul(W_, Q).to(device="cpu", dtype=dtype)
    except:
        for W in [layer.attn.qkv]:
            dtype = W.weight.dtype
            W_ = W.weight.to(device=device, dtype=torch.float64)
            W.weight.data = torch.matmul(W_, Q).to(device="cpu", dtype=dtype)

def rotate_attention_output(layer, Q, device) -> None:
    # Rotate output matrix of the self-attention layer.
    # Matrix multiplication order: R.T@W (R.T[ic,ic] @ W[ic,oc] -> W_new[ic,oc])
    # Note: For output layers, we use R.T@W instead of W@R.
    # If using weight training for output layers, need special handling.
    try:
        W = layer.self_attn.o_proj
    except:
        W = layer.attn.proj
    dtype = W.weight.data.dtype
    W_ = W.weight.data.to(device=device, dtype=torch.float64)
    W.weight.data = torch.matmul(Q.T, W_).to(device="cpu", dtype=dtype)
    if W.bias is not None:
        b = W.bias.data.to(device=device, dtype=torch.float64)
        W.bias.data = torch.matmul(Q.T, b).to(device="cpu", dtype=dtype)


def rotate_mlp_input(layer, Q, device):
    # Rotate the MLP input weights.
    # Matrix multiplication order: W@R (W[oc,ic] @ R[ic,ic] -> W_new[oc,ic])
    # This matches weight training mode where we train R using W@R.
    mlp_inputs = [layer.mlp.up_proj, layer.mlp.gate_proj]
    for W in mlp_inputs:
        dtype = W.weight.dtype
        W_ = W.weight.data.to(device=device, dtype=torch.float64)
        W.weight.data = torch.matmul(W_, Q).to(device="cpu", dtype=dtype)


def rotate_mlp_input_qwen3_vl(layer, Q, device):
    # Rotate the MLP input weights.
    mlp_inputs = [layer.mlp.linear_fc1]
    for W in mlp_inputs:
        dtype = W.weight.dtype
        W_ = W.weight.data.to(device=device, dtype=torch.float64)
        W.weight.data = torch.matmul(W_, Q).to(device="cpu", dtype=dtype)


def rotate_mlp_input_moe(layer, Q, device):
    # Rotate the MLP input weights.
    mlp_inputs = [layer.mlp.gate]
    for i in range(layer.mlp.n_routed_experts):
        mlp_inputs.append(layer.mlp.experts[i].gate_proj)
        mlp_inputs.append(layer.mlp.experts[i].up_proj)
    if hasattr(layer.mlp, 'shared_experts'):
        mlp_inputs.append(layer.mlp.shared_experts.gate_proj)
        mlp_inputs.append(layer.mlp.shared_experts.up_proj)
    for W in mlp_inputs:
        dtype = W.weight.dtype
        W_ = W.weight.data.to(device=device, dtype=torch.float64)
        W.weight.data = torch.matmul(W_, Q).to(device="cpu", dtype=dtype)


def rotate_mlp_output(layer, Q, device):
    # Rotate the MLP output weights and bias.
    # Matrix multiplication order: R.T@W (R.T[ic,ic] @ W[ic,oc] -> W_new[ic,oc])
    # Note: For output layers, we use R.T@W instead of W@R.
    # If using weight training for output layers, need special handling.
    W = layer.mlp.down_proj
    dtype = W.weight.data.dtype
    W_ = W.weight.data.to(device=device, dtype=torch.float64)
    W.weight.data = torch.matmul(Q.T, W_).to(device="cpu", dtype=dtype)
    # apply_exact_had_to_linear(W, had_dim=-1, output=False) #apply exact (inverse) hadamard on the weights of mlp output
    if W.bias is not None:
        b = W.bias.data.to(device=device, dtype=torch.float64)
        W.bias.data = torch.matmul(Q.T, b).to(device="cpu", dtype=dtype)


def rotate_mlp_output_qwen3_vl(layer, Q, device):
    # Rotate the MLP output weights and bias.
    W = layer.mlp.linear_fc2
    dtype = W.weight.data.dtype
    W_ = W.weight.data.to(device=device, dtype=torch.float64)
    W.weight.data = torch.matmul(Q.T, W_).to(device="cpu", dtype=dtype)
    # apply_exact_had_to_linear(W, had_dim=-1, output=False) #apply exact (inverse) hadamard on the weights of mlp output
    if W.bias is not None:
        b = W.bias.data.to(device=device, dtype=torch.float64)
        W.bias.data = torch.matmul(Q.T, b).to(device="cpu", dtype=dtype)


def rotate_mlp_output_moe(layer, Q, device):
    # Rotate the MLP output weights and bias.
    for i in range(layer.mlp.n_routed_experts):
        W = layer.mlp.experts[i].down_proj
        dtype = W.weight.data.dtype
        W_ = W.weight.data.to(device=device, dtype=torch.float64)
        W.weight.data = torch.matmul(Q.T, W_).to(device="cpu", dtype=dtype)
        # apply_exact_had_to_linear(W, had_dim=-1, output=False) #apply exact (inverse) hadamard on the weights of mlp output
        if W.bias is not None:
            b = W.bias.data.to(device=device, dtype=torch.float64)
            W.bias.data = torch.matmul(Q.T, b).to(device="cpu", dtype=dtype)
    if hasattr(layer.mlp, 'shared_experts'):
        W = layer.mlp.shared_experts.down_proj
        dtype = W.weight.data.dtype
        W_ = W.weight.data.to(device=device, dtype=torch.float64)
        W.weight.data = torch.matmul(Q.T, W_).to(device="cpu", dtype=dtype)
        # apply_exact_had_to_linear(W, had_dim=-1, output=False) #apply exact (inverse) hadamard on the weights of mlp output
        if W.bias is not None:
            b = W.bias.data.to(device=device, dtype=torch.float64)
            W.bias.data = torch.matmul(Q.T, b).to(device="cpu", dtype=dtype)

def rotate_head(model, Q, device, lm_head_name) -> None:
    # Rotate the head.
    W, _ = get_module_by_name_prefix(model, lm_head_name)
    dtype = W.weight.data.dtype
    W_ = W.weight.data.to(device=device, dtype=torch.float64)
    W.weight.data = torch.matmul(W_, Q).to(device="cpu", dtype=dtype)


def rotate_ov_proj(layer, head_num, head_dim):
    v_proj = layer.self_attn.v_proj
    o_proj = layer.self_attn.o_proj
    apply_exact_had_to_linear(v_proj, had_dim=head_dim, output=True)
    # apply_exact_had_to_linear(o_proj, had_dim=-1, output=False)
    apply_exact_had_to_linear(o_proj, had_dim=head_dim, output=False)


def rotate_attention_inputs_qwen3_5_moe(layer, Q, device) -> None:
    projections = []
    if hasattr(layer, "self_attn"):
        projections.extend([layer.self_attn.q_proj, layer.self_attn.k_proj, layer.self_attn.v_proj])
    if hasattr(layer, "linear_attn"):
        projections.extend(
            [
                layer.linear_attn.in_proj_qkv,
                layer.linear_attn.in_proj_z,
                layer.linear_attn.in_proj_b,
                layer.linear_attn.in_proj_a,
            ]
        )

    for W in projections:
        dtype = W.weight.dtype
        W_ = W.weight.to(device=device, dtype=torch.float64)
        W.weight.data = torch.matmul(W_, Q).to(device="cpu", dtype=dtype)


def rotate_attention_inputs_qwen3_5(layer, Q, device) -> None:
    projections = []
    if hasattr(layer, "self_attn"):
        projections.extend([layer.self_attn.q_proj, layer.self_attn.k_proj, layer.self_attn.v_proj])
    if hasattr(layer, "linear_attn"):
        projections.extend(
            [
                layer.linear_attn.in_proj_qkv,
                layer.linear_attn.in_proj_z,
                layer.linear_attn.in_proj_b,
                layer.linear_attn.in_proj_a,
            ]
        )

    for W in projections:
        dtype = W.weight.dtype
        W_ = W.weight.to(device=device, dtype=torch.float64)
        W.weight.data = torch.matmul(W_, Q).to(device="cpu", dtype=dtype)


def rotate_attention_output_qwen3_5_moe(layer, Q, device) -> None:
    if hasattr(layer, "self_attn"):
        W = layer.self_attn.o_proj
    elif hasattr(layer, "linear_attn"):
        W = layer.linear_attn.out_proj
    else:
        return

    dtype = W.weight.data.dtype
    W_ = W.weight.data.to(device=device, dtype=torch.float64)
    W.weight.data = torch.matmul(Q.T, W_).to(device="cpu", dtype=dtype)
    if W.bias is not None:
        b = W.bias.data.to(device=device, dtype=torch.float64)
        W.bias.data = torch.matmul(Q.T, b).to(device="cpu", dtype=dtype)


def rotate_attention_output_qwen3_5(layer, Q, device) -> None:
    if hasattr(layer, "self_attn"):
        W = layer.self_attn.o_proj
    elif hasattr(layer, "linear_attn"):
        W = layer.linear_attn.out_proj
    else:
        return

    dtype = W.weight.data.dtype
    W_ = W.weight.data.to(device=device, dtype=torch.float64)
    W.weight.data = torch.matmul(Q.T, W_).to(device="cpu", dtype=dtype)
    if W.bias is not None:
        b = W.bias.data.to(device=device, dtype=torch.float64)
        W.bias.data = torch.matmul(Q.T, b).to(device="cpu", dtype=dtype)


def rotate_mlp_input_qwen3_5_moe(layer, Q, device) -> None:
    mlp = layer.mlp
    mlp_inputs = []

    if hasattr(mlp, "gate") and hasattr(mlp.gate, "weight"):
        gate_dtype = mlp.gate.weight.dtype
        gate_w = mlp.gate.weight.data.to(device=device, dtype=torch.float64)
        mlp.gate.weight.data = torch.matmul(gate_w, Q).to(device="cpu", dtype=gate_dtype)

    if hasattr(mlp, "shared_expert"):
        mlp_inputs.extend([mlp.shared_expert.gate_proj, mlp.shared_expert.up_proj])
    if hasattr(mlp, "shared_expert_gate"):
        mlp_inputs.append(mlp.shared_expert_gate)

    experts = getattr(mlp, "experts", None)
    if experts is not None:
        if hasattr(experts, "gate_up_proj"):
            dtype = experts.gate_up_proj.dtype
            gate_up = experts.gate_up_proj.data.to(device=device, dtype=torch.float64)
            experts.gate_up_proj.data = torch.matmul(gate_up, Q).to(device="cpu", dtype=dtype)
        elif hasattr(experts, "gate_proj") and hasattr(experts, "up_proj"):
            for i in range(len(experts.gate_proj)):
                mlp_inputs.append(experts.gate_proj[i])
                mlp_inputs.append(experts.up_proj[i])
        else:
            # defuser per-expert container format: experts[i].gate_proj / up_proj
            num_experts = getattr(experts, "num_experts", None)
            if num_experts is not None:
                for i in range(num_experts):
                    expert = experts[i]
                    if hasattr(expert, "gate_proj"):
                        mlp_inputs.append(expert.gate_proj)
                    if hasattr(expert, "up_proj"):
                        mlp_inputs.append(expert.up_proj)

    for W in mlp_inputs:
        dtype = W.weight.dtype
        W_ = W.weight.data.to(device=device, dtype=torch.float64)
        W.weight.data = torch.matmul(W_, Q).to(device="cpu", dtype=dtype)


def rotate_mlp_input_qwen3_5(layer, Q, device) -> None:
    for W in [layer.mlp.gate_proj, layer.mlp.up_proj]:
        dtype = W.weight.dtype
        W_ = W.weight.data.to(device=device, dtype=torch.float64)
        W.weight.data = torch.matmul(W_, Q).to(device="cpu", dtype=dtype)


def rotate_mlp_output_qwen3_5_moe(layer, Q, device) -> None:
    mlp = layer.mlp
    output_linears = []

    if hasattr(mlp, "shared_expert"):
        output_linears.append(mlp.shared_expert.down_proj)

    experts = getattr(mlp, "experts", None)
    if experts is not None:
        if hasattr(experts, "down_proj") and isinstance(experts.down_proj, torch.nn.Parameter):
            dtype = experts.down_proj.dtype
            down_proj = experts.down_proj.data.to(device=device, dtype=torch.float64)
            experts.down_proj.data = torch.matmul(Q.T, down_proj).to(device="cpu", dtype=dtype)
        elif hasattr(experts, "down_proj"):
            for i in range(len(experts.down_proj)):
                output_linears.append(experts.down_proj[i])
        else:
            # defuser per-expert container format: experts[i].down_proj
            num_experts = getattr(experts, "num_experts", None)
            if num_experts is not None:
                for i in range(num_experts):
                    expert = experts[i]
                    if hasattr(expert, "down_proj"):
                        output_linears.append(expert.down_proj)

    for W in output_linears:
        dtype = W.weight.data.dtype
        W_ = W.weight.data.to(device=device, dtype=torch.float64)
        W.weight.data = torch.matmul(Q.T, W_).to(device="cpu", dtype=dtype)
        if W.bias is not None:
            b = W.bias.data.to(device=device, dtype=torch.float64)
            W.bias.data = torch.matmul(Q.T, b).to(device="cpu", dtype=dtype)


def rotate_mlp_output_qwen3_5(layer, Q, device) -> None:
    W = layer.mlp.down_proj
    dtype = W.weight.data.dtype
    W_ = W.weight.data.to(device=device, dtype=torch.float64)
    W.weight.data = torch.matmul(Q.T, W_).to(device="cpu", dtype=dtype)
    if W.bias is not None:
        b = W.bias.data.to(device=device, dtype=torch.float64)
        W.bias.data = torch.matmul(Q.T, b).to(device="cpu", dtype=dtype)


def rotate_attention_inputs_qwen3_next(layer, Q, device) -> None:
    projections = []
    if hasattr(layer, "self_attn"):
        projections.extend([layer.self_attn.q_proj, layer.self_attn.k_proj, layer.self_attn.v_proj])
    if hasattr(layer, "linear_attn"):
        projections.extend([layer.linear_attn.in_proj_qkvz, layer.linear_attn.in_proj_ba])

    for W in projections:
        dtype = W.weight.dtype
        W_ = W.weight.to(device=device, dtype=torch.float64)
        W.weight.data = torch.matmul(W_, Q).to(device="cpu", dtype=dtype)


def rotate_attention_output_qwen3_next(layer, Q, device) -> None:
    if hasattr(layer, "self_attn"):
        W = layer.self_attn.o_proj
    elif hasattr(layer, "linear_attn"):
        W = layer.linear_attn.out_proj
    else:
        return

    dtype = W.weight.data.dtype
    W_ = W.weight.data.to(device=device, dtype=torch.float64)
    W.weight.data = torch.matmul(Q.T, W_).to(device="cpu", dtype=dtype)
    if W.bias is not None:
        b = W.bias.data.to(device=device, dtype=torch.float64)
        W.bias.data = torch.matmul(Q.T, b).to(device="cpu", dtype=dtype)


def rotate_mlp_input_qwen3_next(layer, Q, device) -> None:
    mlp = layer.mlp
    mlp_inputs = []

    if hasattr(mlp, "gate_proj") and hasattr(mlp, "up_proj"):
        mlp_inputs.extend([mlp.gate_proj, mlp.up_proj])
    else:
        if hasattr(mlp, "gate") and hasattr(mlp.gate, "weight"):
            gate_dtype = mlp.gate.weight.dtype
            gate_w = mlp.gate.weight.data.to(device=device, dtype=torch.float64)
            mlp.gate.weight.data = torch.matmul(gate_w, Q).to(device="cpu", dtype=gate_dtype)
        if hasattr(mlp, "shared_expert"):
            mlp_inputs.extend([mlp.shared_expert.gate_proj, mlp.shared_expert.up_proj])
        if hasattr(mlp, "shared_expert_gate"):
            mlp_inputs.append(mlp.shared_expert_gate)
        experts = getattr(mlp, "experts", None)
        if experts is not None:
            if hasattr(experts, "gate_up_proj"):
                dtype = experts.gate_up_proj.dtype
                gate_up = experts.gate_up_proj.data.to(device=device, dtype=torch.float64)
                experts.gate_up_proj.data = torch.matmul(gate_up, Q).to(device="cpu", dtype=dtype)
            else:
                for expert in experts:
                    if hasattr(expert, "gate_proj") and hasattr(expert, "up_proj"):
                        mlp_inputs.extend([expert.gate_proj, expert.up_proj])

    for W in mlp_inputs:
        dtype = W.weight.dtype
        W_ = W.weight.data.to(device=device, dtype=torch.float64)
        W.weight.data = torch.matmul(W_, Q).to(device="cpu", dtype=dtype)


def rotate_mlp_output_qwen3_next(layer, Q, device) -> None:
    mlp = layer.mlp
    output_linears = []

    if hasattr(mlp, "down_proj") and hasattr(mlp, "gate_proj"):
        output_linears.append(mlp.down_proj)
    if hasattr(mlp, "shared_expert"):
        output_linears.append(mlp.shared_expert.down_proj)
    experts = getattr(mlp, "experts", None)
    if experts is not None:
        if hasattr(experts, "down_proj") and isinstance(experts.down_proj, torch.nn.Parameter):
            dtype = experts.down_proj.dtype
            down_proj = experts.down_proj.data.to(device=device, dtype=torch.float64)
            experts.down_proj.data = torch.matmul(Q.T, down_proj).to(device="cpu", dtype=dtype)
        else:
            for expert in experts:
                if hasattr(expert, "down_proj"):
                    output_linears.append(expert.down_proj)

    for W in output_linears:
        dtype = W.weight.data.dtype
        W_ = W.weight.data.to(device=device, dtype=torch.float64)
        W.weight.data = torch.matmul(Q.T, W_).to(device="cpu", dtype=dtype)
        if W.bias is not None:
            b = W.bias.data.to(device=device, dtype=torch.float64)
            W.bias.data = torch.matmul(Q.T, b).to(device="cpu", dtype=dtype)


@torch.inference_mode()
def rotate_model_qwen3_5_moe(
    model: PreTrainedModel,
    rotate_mode: str,
    device: torch.device,
    lm_head_name: str,
    layers_node: str,
    Q=None,
):
    q_block_spec = Q
    if isinstance(Q, dict):
        q_block_spec = Q.get("Q_block", Q.get("Q"))

    text_config = getattr(model.config, "text_config", model.config)
    hidden_size = getattr(text_config, "hidden_size", None)
    if hidden_size is None:
        raise ValueError("Qwen3.5-MoE rotation requires `hidden_size` in text config.")

    Q_block = _resolve_rotation_matrix(q_block_spec, 0, "Q_block")
    if Q_block is None:
        Q_block = get_orthogonal_matrix(hidden_size, rotate_mode, device)

    rotate_embeddings(model, Q_block, device)
    rotate_head(model, Q_block, device, lm_head_name)
    torch_empty_cache()

    layers, _ = get_module_by_name_prefix(model, layers_node)
    for idx, layer in enumerate(log.pb(layers).title("Rotating Qwen3.5-MoE")):
        q_layer = _resolve_rotation_matrix(q_block_spec, idx, "Q_block")
        if q_layer is None:
            q_layer = Q_block

        rotate_attention_inputs_qwen3_5_moe(layer, q_layer, device)
        rotate_attention_output_qwen3_5_moe(layer, q_layer, device)
        rotate_mlp_input_qwen3_5_moe(layer, q_layer, device)
        rotate_mlp_output_qwen3_5_moe(layer, q_layer, device)

        # NOTE:
        # Qwen3.5-MoE already mixes full-attention and linear-attention blocks; we
        # keep block-rotation only and skip extra V/O hadamard projection rewrite
        # here to avoid additional backend dependency constraints.

    return model, Q_block


@torch.inference_mode()
def rotate_model_qwen3_5(
    model: PreTrainedModel,
    rotate_mode: str,
    device: torch.device,
    lm_head_name: str,
    layers_node: str,
    Q=None,
):
    q_block_spec = Q
    if isinstance(Q, dict):
        q_block_spec = Q.get("Q_block", Q.get("Q"))

    text_config = getattr(model.config, "text_config", model.config)
    hidden_size = getattr(text_config, "hidden_size", None)
    if hidden_size is None:
        raise ValueError("Qwen3.5 rotation requires `hidden_size` in text config.")

    Q_block = _resolve_rotation_matrix(q_block_spec, 0, "Q_block")
    if Q_block is None:
        Q_block = get_orthogonal_matrix(hidden_size, rotate_mode, device)

    rotate_embeddings(model, Q_block, device)
    rotate_head(model, Q_block, device, lm_head_name)
    torch_empty_cache()

    layers, _ = get_module_by_name_prefix(model, layers_node)
    for idx, layer in enumerate(log.pb(layers).title("Rotating Qwen3.5")):
        q_layer = _resolve_rotation_matrix(q_block_spec, idx, "Q_block")
        if q_layer is None:
            q_layer = Q_block

        rotate_attention_inputs_qwen3_5(layer, q_layer, device)
        rotate_attention_output_qwen3_5(layer, q_layer, device)
        rotate_mlp_input_qwen3_5(layer, q_layer, device)
        rotate_mlp_output_qwen3_5(layer, q_layer, device)

    return model, Q_block


@torch.inference_mode()
def rotate_model_qwen3_next(
    model: PreTrainedModel,
    rotate_mode: str,
    device: torch.device,
    lm_head_name: str,
    layers_node: str,
    Q=None,
):
    q_block_spec = Q
    if isinstance(Q, dict):
        q_block_spec = Q.get("Q_block", Q.get("Q"))

    text_config = getattr(model.config, "text_config", model.config)
    hidden_size = getattr(text_config, "hidden_size", None)
    if hidden_size is None:
        raise ValueError("Qwen3-Next rotation requires `hidden_size` in model config.")

    Q_block = _resolve_rotation_matrix(q_block_spec, 0, "Q_block")
    if Q_block is None:
        Q_block = get_orthogonal_matrix(hidden_size, rotate_mode, device)

    rotate_embeddings(model, Q_block, device)
    rotate_head(model, Q_block, device, lm_head_name)
    torch_empty_cache()

    layers, _ = get_module_by_name_prefix(model, layers_node)
    for idx, layer in enumerate(log.pb(layers).title("Rotating Qwen3-Next")):
        q_layer = _resolve_rotation_matrix(q_block_spec, idx, "Q_block")
        if q_layer is None:
            q_layer = Q_block

        rotate_attention_inputs_qwen3_next(layer, q_layer, device)
        rotate_attention_output_qwen3_next(layer, q_layer, device)
        rotate_mlp_input_qwen3_next(layer, q_layer, device)
        rotate_mlp_output_qwen3_next(layer, q_layer, device)

    return model, Q_block


@torch.inference_mode()
def rotate_model(model: PreTrainedModel, rotate_mode: str, device: torch.device, lm_head_name: str,
                 layers_node: str, Q=None):
    q_block_spec = Q
    if isinstance(Q, dict):
        q_block_spec = Q.get("Q_block", Q.get("Q"))

    Q_block = _resolve_rotation_matrix(q_block_spec, 0, "Q_block")
    if Q_block is None:
        Q_block = get_orthogonal_matrix(model.config.hidden_size, rotate_mode, device)

    config = model.config
    num_heads = config.num_attention_heads
    model_dim = config.hidden_size
    head_dim = _resolve_attention_head_dim(config, model_dim, num_heads)

    rotate_embeddings(model, Q_block, device)
    rotate_head(model, Q_block, device, lm_head_name)
    torch_empty_cache()
    layers , _ = get_module_by_name_prefix(model, layers_node)
    for idx, layer in enumerate(log.pb(layers).title("Rotating")):
        q_layer = _resolve_rotation_matrix(q_block_spec, idx, "Q_block")
        if q_layer is None:
            q_layer = Q_block
        rotate_attention_inputs(layer, q_layer, device)
        rotate_attention_output(layer, q_layer, device)
        rotate_mlp_input(layer, q_layer, device)
        rotate_mlp_output(layer, q_layer, device)
        rotate_ov_proj(layer, num_heads, head_dim)
    return model, Q_block


@torch.inference_mode()
def rotate_model_qwen2_5_vl(model: PreTrainedModel, rotate_mode: str, device: torch.device, lm_head_name: str,
                          layers_node: str, Q=None):
    q_block_spec = Q
    if isinstance(Q, dict):
        q_block_spec = Q.get("Q_block", Q.get("Q"))

    Q_block = _resolve_rotation_matrix(q_block_spec, 0, "Q_block")
    if Q_block is None:
        Q_block = get_orthogonal_matrix(model.config.text_config.hidden_size, rotate_mode, device)

    config = model.config
    num_heads = config.text_config.num_attention_heads
    model_dim = config.text_config.hidden_size
    head_dim = _resolve_attention_head_dim(config.text_config, model_dim, num_heads)

    rotate_qwen2_5_vl_embeddings(model, Q_block, device)
    rotate_head(model, Q_block, device, lm_head_name)
    torch_empty_cache()
    layers , _ = get_module_by_name_prefix(model, layers_node)
    for idx, layer in enumerate(log.pb(layers).title("Rotating")):
        q_layer = _resolve_rotation_matrix(q_block_spec, idx, "Q_block")
        if q_layer is None:
            q_layer = Q_block
        rotate_attention_inputs(layer, q_layer, device)
        rotate_attention_output(layer, q_layer, device)
        rotate_mlp_input(layer, q_layer, device)
        rotate_mlp_output(layer, q_layer, device)
    return model, Q_block


@torch.inference_mode()
def rotate_model_qwen3_vl(model: PreTrainedModel, rotate_mode: str, device: torch.device, lm_head_name: str,
                          layers_node: str, Q=None):
    q_block_spec = Q
    if isinstance(Q, dict):
        q_block_spec = Q.get("Q_block", Q.get("Q"))

    Q_block = _resolve_rotation_matrix(q_block_spec, 0, "Q_block")
    if Q_block is None:
        Q_block = get_orthogonal_matrix(model.config.text_config.hidden_size, rotate_mode, device)

    config = model.config
    num_heads = config.text_config.num_attention_heads
    model_dim = config.text_config.hidden_size
    head_dim = _resolve_attention_head_dim(config.text_config, model_dim, num_heads)

    rotate_qwen3_vl_embeddings(model, Q_block, device)
    rotate_head(model, Q_block, device, lm_head_name)
    torch_empty_cache()
    layers , _ = get_module_by_name_prefix(model, layers_node)
    for idx, layer in enumerate(log.pb(layers).title("Rotating")):
        q_layer = _resolve_rotation_matrix(q_block_spec, idx, "Q_block")
        if q_layer is None:
            q_layer = Q_block
        rotate_attention_inputs(layer, q_layer, device)
        rotate_attention_output(layer, q_layer, device)
        rotate_mlp_input(layer, q_layer, device)
        rotate_mlp_output(layer, q_layer, device)
    return model, Q_block


@torch.inference_mode()
def rotate_model_qwen3_5_vl(model: PreTrainedModel, rotate_mode: str, device: torch.device, lm_head_name: str,
                            layers_node: str, Q=None):
    q_block_spec = Q
    if isinstance(Q, dict):
        q_block_spec = Q.get("Q_block", Q.get("Q"))

    text_config = getattr(model.config, "text_config", model.config)
    hidden_size = getattr(text_config, "hidden_size", None)
    if hidden_size is None:
        raise ValueError("Qwen3.5-VL rotation requires `hidden_size` in text config.")

    Q_block = _resolve_rotation_matrix(q_block_spec, 0, "Q_block")
    if Q_block is None:
        Q_block = get_orthogonal_matrix(hidden_size, rotate_mode, device)

    rotate_qwen3_5_vl_embeddings(model, Q_block, device)
    rotate_head(model, Q_block, device, lm_head_name)
    torch_empty_cache()

    layers, _ = get_module_by_name_prefix(model, layers_node)
    for idx, layer in enumerate(log.pb(layers).title("Rotating Qwen3.5-VL")):
        q_layer = _resolve_rotation_matrix(q_block_spec, idx, "Q_block")
        if q_layer is None:
            q_layer = Q_block

        rotate_attention_inputs_qwen3_5(layer, q_layer, device)
        rotate_attention_output_qwen3_5(layer, q_layer, device)
        rotate_mlp_input_qwen3_5(layer, q_layer, device)
        rotate_mlp_output_qwen3_5(layer, q_layer, device)

    return model, Q_block


@torch.inference_mode()
def rotate_model_moe(model: PreTrainedModel, rotate_mode: str, device: torch.device, lm_head_name: str,
                          layers_node: str, Q=None):
    q_block_spec = Q
    if isinstance(Q, dict):
        q_block_spec = Q.get("Q_block", Q.get("Q"))

    Q_block = _resolve_rotation_matrix(q_block_spec, 0, "Q_block")
    if Q_block is None:
        Q_block = get_orthogonal_matrix(model.config.hidden_size, rotate_mode, device)

    config = model.config
    num_heads = config.num_attention_heads
    model_dim = config.hidden_size
    head_dim = _resolve_attention_head_dim(config, model_dim, num_heads)

    rotate_embeddings(model, Q_block, device)
    rotate_head(model, Q_block, device, lm_head_name)
    torch_empty_cache()
    layers , _ = get_module_by_name_prefix(model, layers_node)
    for idx, layer in enumerate(log.pb(layers).title("Rotating")):
        q_layer = _resolve_rotation_matrix(q_block_spec, idx, "Q_block")
        if q_layer is None:
            q_layer = Q_block
        rotate_attention_inputs(layer, q_layer, device)
        rotate_attention_output(layer, q_layer, device)
        rotate_mlp_input_moe(layer, q_layer, device)
        rotate_mlp_output_moe(layer, q_layer, device)
        rotate_ov_proj(layer, num_heads, head_dim)
    return model, Q_block


@torch.inference_mode()
def rotate_vision_model(model: PreTrainedModel, rotate_mode: str, device: torch.device, lm_head_name: str,
                        layers_node: str, Q=None):
    num_heads = model.visual.blocks[0].attn.num_heads
    Q = (
        get_orthogonal_matrix(
            model.visual.blocks[0].attn.qkv.in_features, rotate_mode, device
        )
        if Q is None
        else Q
    )
    rotate_conv(
            model.visual.patch_embed.proj,
            Q,
            model.visual.blocks[0].attn.qkv.in_features,
            device,
    )
    torch_empty_cache()
    for idx, layer in enumerate(log.pb(model.visual.blocks).title("Rotating")):
        rotate_attention_inputs(layer, Q, device)
        rotate_attention_output(layer, Q, device)
        rotate_mlp_input(layer, Q, device)
        rotate_mlp_output(layer, Q, device)
    rotate_visual_merger(model, Q, device)
    return model, Q


@torch.inference_mode()
def rotate_vision_model_qwen3_vl(model: PreTrainedModel, rotate_mode: str, device: torch.device, lm_head_name: str,
                        layers_node: str, Q=None):
    num_heads = model.visual.blocks[0].attn.num_heads
    Q = (
        get_orthogonal_matrix(
            model.visual.blocks[0].attn.qkv.in_features, rotate_mode, device
        )
        if Q is None
        else Q
    )
    rotate_conv(
            model.visual.patch_embed.proj,
            Q,
            model.visual.blocks[0].attn.qkv.in_features,
            device,
    )
    rotate_qwen3_vl_vision_embeddings(model, 
                                      Q, 
                                      device) 
    torch_empty_cache()
    for idx, layer in enumerate(log.pb(model.visual.blocks).title("Rotating")):
        rotate_attention_inputs(layer, Q, device)
        rotate_attention_output(layer, Q, device)
        rotate_mlp_input_qwen3_vl(layer, Q, device)
        rotate_mlp_output_qwen3_vl(layer, Q, device)
    rotate_visual_merger_qwen3_vl(model, Q, device)
    rotate_qwen3_vl_deepstack_merger(model, Q, device)
    return model, Q


def rotate_qwen3_vl_deepstack_merger(model, Q: torch.Tensor, device) -> None:
    for layer in model.visual.deepstack_merger_list:
        dtype = layer.linear_fc1.weight.dtype
        q_shape = Q.shape[0]
        o_shape, i_shape = layer.linear_fc1.weight.shape
        W_ = layer.linear_fc1.weight.to(device=device, dtype=torch.float64).reshape(o_shape, -1, q_shape)
        layer.linear_fc1.weight.data = torch.matmul(W_, Q).to(device=device, dtype=dtype).reshape(o_shape, i_shape).contiguous()
@torch.inference_mode()
def rotate_vision_model_qwen3_5_vl(model: PreTrainedModel, rotate_mode: str, device: torch.device, lm_head_name: str,
                                   layers_node: str, Q=None):
    visual = _get_qwen3_5_visual(model)
    Q = (
        get_orthogonal_matrix(
            visual.blocks[0].attn.qkv.in_features, rotate_mode, device
        )
        if Q is None
        else Q
    )
    rotate_conv(
        visual.patch_embed.proj,
        Q,
        visual.blocks[0].attn.qkv.in_features,
        device,
    )
    rotate_qwen3_5_vl_vision_embeddings(model, Q, device)
    torch_empty_cache()
    for idx, layer in enumerate(log.pb(visual.blocks).title("Rotating Qwen3.5-Vision")):
        rotate_attention_inputs(layer, Q, device)
        rotate_attention_output(layer, Q, device)
        rotate_mlp_input_qwen3_vl(layer, Q, device)
        rotate_mlp_output_qwen3_vl(layer, Q, device)
    rotate_visual_merger_qwen3_5_vl(model, Q, device)
    return model, Q


# ============================================================================
# GLM4 MoE Lite (MLA attention) rotation functions
# ============================================================================

def fuse_layer_norms_mla_with_moe(model: PreTrainedModel, pre_lm_head_norm_module_name: str, layers_node: str, lm_head_name: str):
    """
    Fuse layer norms for GLM4 MoE Lite model.
    
    GLM4 MoE Lite uses MLA (Multi-head Latent Attention) with two-stage projections:
    - First stage (takes hidden states): q_a_proj, kv_a_proj_with_mqa (or kv_a_proj_latent, kv_a_proj_rope after fusion)
    - Second stage (takes intermediate): q_b_proj, kv_b_proj (or q_rope_proj, q_absorbed_proj after fusion)
    
    We only fuse layernorm into first-stage projections that take hidden states directly.
    MoE: shared_experts (gate_proj, up_proj, down_proj) 
    """
    layers, _ = get_module_by_name_prefix(model, layers_node)
    
    for layer in layers:
        if hasattr(layer.mlp, 'experts'):
            # Fuse post_attention_layernorm into MLP inputs (shared_experts)
            fuse_ln_linear(
                    layer.post_attention_layernorm, [layer.mlp.gate]
                )
            for i in range(layer.mlp.n_routed_experts):
                fuse_ln_linear(
                    layer.post_attention_layernorm, [layer.mlp.experts.gate_proj[i], layer.mlp.experts.up_proj[i]]
                )
            if hasattr(layer.mlp, 'shared_experts'):
                fuse_ln_linear(
                    layer.post_attention_layernorm, [layer.mlp.shared_experts.gate_proj, layer.mlp.shared_experts.up_proj]
                )
        else:
            # Fuse post_attention_layernorm into MLP inputs (shared_experts)
            fuse_ln_linear(
                    layer.post_attention_layernorm, [layer.mlp.up_proj, layer.mlp.gate_proj]
                )
        # Fuse input_layernorm into first-stage attention MLA projections ONLY
        # These are the projections that take hidden_states (dim=2048) as input
        attn = layer.self_attn
        attn_inputs = []
        
        # First-stage Q projection (takes hidden_states -> q_lora_rank)
        if hasattr(attn, 'q_a_proj'):
            attn_inputs.append(attn.q_a_proj)
        
        # First-stage KV projections (takes hidden_states)
        # if hasattr(attn, 'kv_a_proj_latent'):
        #     attn_inputs.append(attn.kv_a_proj_latent)
        # if hasattr(attn, 'kv_a_proj_rope'):
        #     attn_inputs.append(attn.kv_a_proj_rope)
        # Before MLA fusion: kv_a_proj_with_mqa takes hidden_states
        if hasattr(attn, 'kv_a_proj_with_mqa'):
            attn_inputs.append(attn.kv_a_proj_with_mqa)
        
        # NOTE: Do NOT include q_b_proj, q_rope_proj, q_absorbed_proj, kv_b_proj
        # These take intermediate representations (from q_a_proj or kv_a_proj), not hidden_states
            
        if attn_inputs:
            fuse_ln_linear(layer.input_layernorm, attn_inputs)

        # Fuse q_a_layernorm into q_b_proj
        if hasattr(attn, 'q_a_layernorm'):
            fuse_ln_linear(attn.q_a_layernorm, [attn.q_b_proj])
        
        # Fuse kv_a_layernorm into kv_b_proj
        if hasattr(attn, 'kv_a_layernorm'):
            fuse_ln_linear(attn.kv_a_layernorm, [attn.kv_b_proj])
        
        reset_ln(layer.post_attention_layernorm)
        reset_ln(layer.input_layernorm)
        reset_ln(attn.q_a_layernorm)
        reset_ln(attn.kv_a_layernorm)
    
    # Fuse pre-head layernorm into lm_head
    pre_head_layernorm, _ = get_module_by_name_prefix(model, pre_lm_head_norm_module_name)
    lm_head, _ = get_module_by_name_prefix(model, lm_head_name)
    
    fuse_ln_linear(pre_head_layernorm, [lm_head])
    reset_ln(pre_head_layernorm)
    return model


def fuse_layer_norms_deepseekv3_mla_moe(
    model: PreTrainedModel,
    pre_lm_head_norm_module_name: str,
    layers_node: str,
    lm_head_name: str,
):
    """
    Alias for DeepSeek V3 MLA+MoE layer-norm fusion.

    DeepSeek V3 的文本部分结构与 GLM4 MoE Lite 的 MLA+MoE 结构一致，这里直接复用
    `fuse_layer_norms_mla_with_moe` 的实现，仅在命名上区分，便于阅读和注册。
    """
    return fuse_layer_norms_mla_with_moe(
        model=model,
        pre_lm_head_norm_module_name=pre_lm_head_norm_module_name,
        layers_node=layers_node,
        lm_head_name=lm_head_name,
    )

def fuse_layer_norms_deepseekv2_ocr(
    model: PreTrainedModel,
    pre_lm_head_norm_module_name: str,
    layers_node: str,
    lm_head_name: str,
):
    """
    Fuse layer norms for DeepSeek V2 OCR (MHA + MoE) model.

    当前 gptqmodel 中的 DeepSeekV2 实现基于 DeepSeek-OCR 文本模型：
      - 注意力部分是标准 MHA（LlamaAttention），只有 q_proj/k_proj/v_proj/o_proj，
        没有 MLA 的 q_a_proj/kv_a_proj_with_mqa 等低秩结构。
      - MLP 仍然是 DeepseekV2 风格的 MoE：experts 为 DeepseekV2MLP(gate_proj, up_proj, down_proj)，
        可选 shared_experts 结构相同。

    因此这里只对：
      - MLP（dense 或 MoE）的 gate_proj / up_proj 融合 post_attention_layernorm
      - MHA 的 q_proj / k_proj / v_proj 融合 input_layernorm
      - pre_lm_head_norm 融合到 lm_head
    """
    layers, _ = get_module_by_name_prefix(model, layers_node)

    for layer in layers:
        mlp = layer.mlp
        # Fuse post_attention_layernorm into MLP inputs
        if hasattr(mlp, "experts"):
            # MoE layers: experts is a ModuleList of DeepseekV2MLP
            for expert in mlp.experts:
                if expert is None:
                    continue
                expert_linears = []
                if hasattr(expert, "gate_proj"):
                    expert_linears.append(expert.gate_proj)
                if hasattr(expert, "up_proj"):
                    expert_linears.append(expert.up_proj)
                if expert_linears:
                    fuse_ln_linear(layer.post_attention_layernorm, expert_linears)

            if hasattr(mlp, "shared_experts"):
                shared_linears = []
                if hasattr(mlp.shared_experts, "gate_proj"):
                    shared_linears.append(mlp.shared_experts.gate_proj)
                if hasattr(mlp.shared_experts, "up_proj"):
                    shared_linears.append(mlp.shared_experts.up_proj)
                if shared_linears:
                    fuse_ln_linear(layer.post_attention_layernorm, shared_linears)
        else:
            # Dense MLP layers
            fuse_ln_linear(
                layer.post_attention_layernorm,
                [mlp.up_proj, mlp.gate_proj],
            )

        # Fuse input_layernorm into MHA attention input projections
        attn = layer.self_attn
        attn_inputs = []
        # DeepSeek-OCR 文本模型使用 LlamaAttention（MHA），只有 q_proj/k_proj/v_proj
        if hasattr(attn, "q_proj"):
            attn_inputs.append(attn.q_proj)
        if hasattr(attn, "k_proj"):
            attn_inputs.append(attn.k_proj)
        if hasattr(attn, "v_proj"):
            attn_inputs.append(attn.v_proj)

        if attn_inputs:
            fuse_ln_linear(layer.input_layernorm, attn_inputs)

        reset_ln(layer.post_attention_layernorm)
        reset_ln(layer.input_layernorm)

    # Fuse pre-head layernorm into lm_head
    pre_head_layernorm, _ = get_module_by_name_prefix(model, pre_lm_head_norm_module_name)
    lm_head, _ = get_module_by_name_prefix(model, lm_head_name)

    fuse_ln_linear(pre_head_layernorm, [lm_head])
    reset_ln(pre_head_layernorm)
    return model


def fuse_layer_norms_deepseekv2_mla_moe(
    model: PreTrainedModel,
    pre_lm_head_norm_module_name: str,
    layers_node: str,
    lm_head_name: str,
):
    """
    Fuse layer norms for DeepSeek V2 (MLA + MoE) model.

    DeepSeek V2 使用：
      - MLA 注意力：q_a_proj / kv_a_proj_with_mqa 为第一阶段投影；
      - MoE MLP：每个 expert 为 DeepseekV2MLP(gate_proj, up_proj, down_proj)，
        以及可选 shared_experts，结构相同。
    """
    layers, _ = get_module_by_name_prefix(model, layers_node)

    for layer in layers:
        mlp = layer.mlp
        # Fuse post_attention_layernorm into MLP inputs
        if hasattr(mlp, "experts"):
            # MoE layers: experts is a ModuleList of DeepseekV2MLP
            for expert in mlp.experts:
                if expert is None:
                    continue
                expert_linears = []
                if hasattr(expert, "gate_proj"):
                    expert_linears.append(expert.gate_proj)
                if hasattr(expert, "up_proj"):
                    expert_linears.append(expert.up_proj)
                if expert_linears:
                    fuse_ln_linear(layer.post_attention_layernorm, expert_linears)

            if hasattr(mlp, "shared_experts"):
                shared_linears = []
                if hasattr(mlp.shared_experts, "gate_proj"):
                    shared_linears.append(mlp.shared_experts.gate_proj)
                if hasattr(mlp.shared_experts, "up_proj"):
                    shared_linears.append(mlp.shared_experts.up_proj)
                if shared_linears:
                    fuse_ln_linear(layer.post_attention_layernorm, shared_linears)
        else:
            # Dense MLP layers
            fuse_ln_linear(
                layer.post_attention_layernorm,
                [mlp.up_proj, mlp.gate_proj],
            )

        # Fuse input_layernorm into first-stage MLA attention projections
        attn = layer.self_attn
        attn_inputs = []
        if hasattr(attn, "q_a_proj"):
            attn_inputs.append(attn.q_a_proj)
        if hasattr(attn, "kv_a_proj_with_mqa"):
            attn_inputs.append(attn.kv_a_proj_with_mqa)

        if attn_inputs:
            fuse_ln_linear(layer.input_layernorm, attn_inputs)

        # Fuse q_a_layernorm into q_b_proj
        if hasattr(attn, "q_a_layernorm") and hasattr(attn, "q_b_proj"):
            fuse_ln_linear(attn.q_a_layernorm, [attn.q_b_proj])

        # Fuse kv_a_layernorm into kv_b_proj
        if hasattr(attn, "kv_a_layernorm") and hasattr(attn, "kv_b_proj"):
            fuse_ln_linear(attn.kv_a_layernorm, [attn.kv_b_proj])

        reset_ln(layer.post_attention_layernorm)
        reset_ln(layer.input_layernorm)
        if hasattr(attn, "q_a_layernorm"):
            reset_ln(attn.q_a_layernorm)
        if hasattr(attn, "kv_a_layernorm"):
            reset_ln(attn.kv_a_layernorm)

    # Fuse pre-head layernorm into lm_head
    pre_head_layernorm, _ = get_module_by_name_prefix(model, pre_lm_head_norm_module_name)
    lm_head, _ = get_module_by_name_prefix(model, lm_head_name)

    fuse_ln_linear(pre_head_layernorm, [lm_head])
    reset_ln(pre_head_layernorm)
    return model

def rotate_attention_inputs_mla(layer, Q, device) -> None:
    """
    Rotate MLA attention input projections.
    
    After MLA fusion, the attention has:
    - q_a_proj: first stage Q projection (optional, if q_lora_rank is set)
    - q_rope_proj: Q projection for RoPE part
    - q_absorbed_proj: Q projection with absorbed K (for content attention)
    - kv_a_proj_latent: KV first stage projection for latent
    - kv_a_proj_rope: KV first stage projection for RoPE
    """
    attn = layer.self_attn
    
    # Collect all attention input projections that exist
    projections = []
    if hasattr(attn, 'q_a_proj'):
        projections.append(attn.q_a_proj)
    # if hasattr(attn, 'kv_a_proj_latent'):
    #     projections.append(attn.kv_a_proj_latent)
    # if hasattr(attn, 'kv_a_proj_rope'):
    #     projections.append(attn.kv_a_proj_rope)
    # Fallback for before fusion
    if hasattr(attn, 'kv_a_proj_with_mqa'):
        projections.append(attn.kv_a_proj_with_mqa)
        
    for W in projections:
        dtype = W.weight.dtype
        W_ = W.weight.to(device=device, dtype=torch.float64)
        W.weight.data = torch.matmul(W_, Q).to(device="cpu", dtype=dtype)


def rotate_attention_output_mla(layer, Q, device) -> None:
    """
    Rotate MLA attention output projection (o_proj).
    """
    W = layer.self_attn.o_proj
    dtype = W.weight.data.dtype
    W_ = W.weight.data.to(device=device, dtype=torch.float64)
    W.weight.data = torch.matmul(Q.T, W_).to(device="cpu", dtype=dtype)
    if W.bias is not None:
        b = W.bias.data.to(device=device, dtype=torch.float64)
        W.bias.data = torch.matmul(Q.T, b).to(device="cpu", dtype=dtype)

def rotate_q_low_rank_mla(layer, Q, device) -> None:
    """
    Rotate MLA low rank projection (q_a_proj, q_b_proj).
    """
    q_a_proj = layer.self_attn.q_a_proj
    q_b_proj = layer.self_attn.q_b_proj
    if Q is None:
        apply_exact_had_to_linear(q_a_proj, had_dim=-1, output=True)
        apply_exact_had_to_linear(q_b_proj, had_dim=-1, output=False)
        return

    dtype = q_a_proj.weight.dtype
    w_q_a = q_a_proj.weight.data.to(device=device, dtype=torch.float64)
    q_a_proj.weight.data = torch.matmul(Q.T, w_q_a).to(device="cpu", dtype=dtype)
    if q_a_proj.bias is not None:
        b = q_a_proj.bias.data.to(device=device, dtype=torch.float64)
        q_a_proj.bias.data = torch.matmul(Q.T, b).to(device="cpu", dtype=dtype)

    dtype = q_b_proj.weight.dtype
    w_q_b = q_b_proj.weight.data.to(device=device, dtype=torch.float64)
    q_b_proj.weight.data = torch.matmul(w_q_b, Q).to(device="cpu", dtype=dtype)

def rotate_kv_low_rank_mla(layer, Q, device) -> None:
    # kv 部分的lowrank的旋转，需要将kv_a_proj_with_mqa的权重拆分成latent和rope两个部分，然后分别进行旋转
    kv_a_proj_with_mqa = layer.self_attn.kv_a_proj_with_mqa
    kv_b_proj = layer.self_attn.kv_b_proj
    W_kv_a = kv_a_proj_with_mqa.weight.data
    W_kv_b = kv_b_proj.weight.data
    
    # latent部分是前kv_lora_rank行，不是后kv_lora_rank行
    # kv_a_proj_with_mqa输出: [kv_lora_rank + qk_rope_head_dim, hidden_size]
    # split后: k_pass (latent) = [:kv_lora_rank], k_rot (rope) = [kv_lora_rank:]
    kv_lora_rank = layer.self_attn.kv_lora_rank
    W_kv_a_latent = W_kv_a[:kv_lora_rank, :]  # 修复索引：前kv_lora_rank行
    dtype = W_kv_a_latent.dtype

    # 对于输出维度的旋转，权重变换为 Q.T @ W (在输出维度上左乘)
    # 但这里W是[out, in]，所以变换后 W_new = Q.T @ W
    W_kv_a_latent = W_kv_a_latent.to(device=device, dtype=torch.float64)
    W_kv_a_latent = torch.matmul(Q.T, W_kv_a_latent).to(device="cpu", dtype=dtype)
    
    # bias也需要在前kv_lora_rank维度上旋转（如果存在）
    if kv_a_proj_with_mqa.bias is not None:
        b = kv_a_proj_with_mqa.bias.data[:kv_lora_rank].to(device=device, dtype=torch.float64)
        kv_a_proj_with_mqa.bias.data[:kv_lora_rank] = torch.matmul(Q.T, b).to(device="cpu", dtype=dtype)

    # kv_b_proj输入维度旋转: W @ Q
    dtype = W_kv_b.dtype
    W_kv_b = W_kv_b.to(device=device, dtype=torch.float64)
    W_kv_b = torch.matmul(W_kv_b, Q).to(device="cpu", dtype=dtype)

    kv_a_proj_with_mqa.weight.data[:kv_lora_rank, :] = W_kv_a_latent  # 修复索引
    kv_b_proj.weight.data = W_kv_b

def rotate_glm4_moe_lite_mlp_input_moe(layer, Q, device):
    # Rotate the MLP input weights.
    mlp_inputs = [layer.mlp.gate]
    for i in range(layer.mlp.n_routed_experts):
        mlp_inputs.append(layer.mlp.experts.gate_proj[i])
        mlp_inputs.append(layer.mlp.experts.up_proj[i])
    if hasattr(layer.mlp, 'shared_experts'):
        mlp_inputs.append(layer.mlp.shared_experts.gate_proj)
        mlp_inputs.append(layer.mlp.shared_experts.up_proj)
    for W in mlp_inputs:
        dtype = W.weight.dtype
        W_ = W.weight.data.to(device=device, dtype=torch.float64)
        W.weight.data = torch.matmul(W_, Q).to(device="cpu", dtype=dtype)

def rotate_glm4_moe_lite_mlp_output_moe(layer, Q, device):
    # Rotate the MLP output weights and bias.
    for i in range(layer.mlp.n_routed_experts):
        W = layer.mlp.experts.down_proj[i]
        dtype = W.weight.data.dtype
        W_ = W.weight.data.to(device=device, dtype=torch.float64)
        W.weight.data = torch.matmul(Q.T, W_).to(device="cpu", dtype=dtype)
        # apply_exact_had_to_linear(W, had_dim=-1, output=False) #apply exact (inverse) hadamard on the weights of mlp output
        if W.bias is not None:
            b = W.bias.data.to(device=device, dtype=torch.float64)
            W.bias.data = torch.matmul(Q.T, b).to(device="cpu", dtype=dtype)
    if hasattr(layer.mlp, 'shared_experts'):
        W = layer.mlp.shared_experts.down_proj
        dtype = W.weight.data.dtype
        W_ = W.weight.data.to(device=device, dtype=torch.float64)
        W.weight.data = torch.matmul(Q.T, W_).to(device="cpu", dtype=dtype)
        # apply_exact_had_to_linear(W, had_dim=-1, output=False) #apply exact (inverse) hadamard on the weights of mlp output
        if W.bias is not None:
            b = W.bias.data.to(device=device, dtype=torch.float64)
            W.bias.data = torch.matmul(Q.T, b).to(device="cpu", dtype=dtype)


def rotate_deepseekv2_mlp_input_moe(layer, Q, device):
    # Rotate the MoE MLP input weights for DeepSeek V2.
    mlp = layer.mlp
    mlp_inputs = []
    if hasattr(mlp, "experts"):
        for expert in mlp.experts:
            if expert is None:
                continue
            if hasattr(expert, "gate_proj"):
                mlp_inputs.append(expert.gate_proj)
            if hasattr(expert, "up_proj"):
                mlp_inputs.append(expert.up_proj)
    if hasattr(mlp, "shared_experts"):
        shared = mlp.shared_experts
        if hasattr(shared, "gate_proj"):
            mlp_inputs.append(shared.gate_proj)
        if hasattr(shared, "up_proj"):
            mlp_inputs.append(shared.up_proj)
    for W in mlp_inputs:
        dtype = W.weight.dtype
        W_ = W.weight.data.to(device=device, dtype=torch.float64)
        W.weight.data = torch.matmul(W_, Q).to(device="cpu", dtype=dtype)


def rotate_deepseekv2_mlp_output_moe(layer, Q, device):
    # Rotate the MoE MLP output weights and bias for DeepSeek V2.
    mlp = layer.mlp
    if hasattr(mlp, "experts"):
        for expert in mlp.experts:
            if expert is None:
                continue
            if hasattr(expert, "down_proj"):
                W = expert.down_proj
                dtype = W.weight.data.dtype
                W_ = W.weight.data.to(device=device, dtype=torch.float64)
                W.weight.data = torch.matmul(Q.T, W_).to(device="cpu", dtype=dtype)
                if W.bias is not None:
                    b = W.bias.data.to(device=device, dtype=torch.float64)
                    W.bias.data = torch.matmul(Q.T, b).to(device="cpu", dtype=dtype)
    if hasattr(mlp, "shared_experts") and hasattr(mlp.shared_experts, "down_proj"):
        W = mlp.shared_experts.down_proj
        dtype = W.weight.data.dtype
        W_ = W.weight.data.to(device=device, dtype=torch.float64)
        W.weight.data = torch.matmul(Q.T, W_).to(device="cpu", dtype=dtype)
        if W.bias is not None:
            b = W.bias.data.to(device=device, dtype=torch.float64)
            W.bias.data = torch.matmul(Q.T, b).to(device="cpu", dtype=dtype)


def _resolve_rotation_matrix(spec, layer_idx: int, target_name: str):
    if spec is None:
        return None
    if torch.is_tensor(spec):
        return spec
    if isinstance(spec, list):
        if layer_idx < len(spec) and torch.is_tensor(spec[layer_idx]):
            return spec[layer_idx]
        return None
    if isinstance(spec, dict):
        direct_key = f"{target_name}_layer_{layer_idx}"
        if direct_key in spec and torch.is_tensor(spec[direct_key]):
            return spec[direct_key]
        if target_name in spec and torch.is_tensor(spec[target_name]):
            return spec[target_name]
        candidate_keys = [k for k in spec if k.startswith(f"{target_name}_layer_")]
        if candidate_keys:
            for key in sorted(candidate_keys):
                if torch.is_tensor(spec[key]):
                    return spec[key]
        for value in spec.values():
            if torch.is_tensor(value):
                return value
    return None


@torch.inference_mode()
def rotate_model_glm4_moe_lite(model: PreTrainedModel, rotate_mode: str, device: torch.device, lm_head_name: str,
                               layers_node: str, Q=None):
    """
    Rotate GLM4 MoE Lite model weights for improved quantization.
    
    This function handles the MLA attention structure and MoE architecture.
    It calls the model's internal `_setup` method to perform fusion and rotation.
    """
    q_block_spec = Q
    q_q_lowrank_spec = None
    q_kv_lowrank_spec = None
    if isinstance(Q, dict):
        q_block_spec = Q.get("Q_block", Q.get("Q"))
        q_q_lowrank_spec = Q.get("Q_q_lowrank")
        q_kv_lowrank_spec = Q.get("Q_kv_lowrank")

    Q_block = _resolve_rotation_matrix(q_block_spec, 0, "Q_block")
    if Q_block is None:
        Q_block = get_orthogonal_matrix(model.config.hidden_size, rotate_mode, device)

    Q_q_lowrank_default = get_orthogonal_matrix(model.config.q_lora_rank, rotate_mode, device)
    Q_kv_lowrank_default = get_orthogonal_matrix(model.config.kv_lora_rank, rotate_mode, device)
    
    rotate_embeddings(model, Q_block.to(device=device, dtype=torch.float64), device)
    rotate_head(model, Q_block.to(device=device, dtype=torch.float64), device, lm_head_name)
    torch_empty_cache()
    
    layers, _ = get_module_by_name_prefix(model, layers_node)
    for idx, layer in enumerate(log.pb(layers).title("Rotating GLM4 MoE Lite")):
        q_block_layer = _resolve_rotation_matrix(q_block_spec, idx, "Q_block")
        if q_block_layer is None:
            q_block_layer = Q_block
        q_q_lowrank_layer = _resolve_rotation_matrix(q_q_lowrank_spec, idx, "Q_q_lowrank")
        if q_q_lowrank_layer is None:
            q_q_lowrank_layer = Q_q_lowrank_default
        q_kv_lowrank_layer = _resolve_rotation_matrix(q_kv_lowrank_spec, idx, "Q_kv_lowrank")
        if q_kv_lowrank_layer is None:
            q_kv_lowrank_layer = Q_kv_lowrank_default

        rotate_attention_inputs_mla(layer, q_block_layer.to(device=device, dtype=torch.float64), device)
        rotate_attention_output_mla(layer, q_block_layer.to(device=device, dtype=torch.float64), device)
        rotate_q_low_rank_mla(layer, q_q_lowrank_layer.to(device=device, dtype=torch.float64), device)
        rotate_kv_low_rank_mla(layer, q_kv_lowrank_layer.to(device=device, dtype=torch.float64), device)
        if hasattr(layer.mlp, 'experts'):
            rotate_glm4_moe_lite_mlp_input_moe(layer, q_block_layer.to(device=device, dtype=torch.float64), device)
            rotate_glm4_moe_lite_mlp_output_moe(layer, q_block_layer.to(device=device, dtype=torch.float64), device)
        else:
            rotate_mlp_input(layer, q_block_layer.to(device=device, dtype=torch.float64), device)
            rotate_mlp_output(layer, q_block_layer.to(device=device, dtype=torch.float64), device)
    
    return model, Q_block  # 修复：返回Q_block而不是Q


@torch.inference_mode()
def rotate_model_deepseekv3_mla_moe(
    model: PreTrainedModel,
    rotate_mode: str,
    device: torch.device,
    lm_head_name: str,
    layers_node: str,
    Q=None,
):
    """
    Alias for DeepSeek V3 MLA+MoE rotation.

    DeepSeek V3 的 MLA+MoE 结构与 GLM4 MoE Lite 一致，因此这里直接复用
    `rotate_model_glm4_moe_lite` 的实现，仅在命名上区分，便于在注册表中对
    DeepSeekV3QModel 和 Glm4MoeLiteQModel 做不同映射。
    """
    return rotate_model_glm4_moe_lite(
        model=model,
        rotate_mode=rotate_mode,
        device=device,
        lm_head_name=lm_head_name,
        layers_node=layers_node,
        Q=Q,
    )



@torch.inference_mode()
def rotate_model_deepseekv2_ocr(
    model: PreTrainedModel,
    rotate_mode: str,
    device: torch.device,
    lm_head_name: str,
    layers_node: str,
    Q=None,
):
    """
    Rotate DeepSeek V2 OCR (MHA + MoE) model weights for improved quantization.

    注意：
    - 这里的 DeepSeekV2 指的是当前工程中接入的 DeepSeek-OCR 文本模型实现，
      其注意力为标准 MHA（LlamaAttention），没有 MLA 的 q_a_proj/kv_a_proj_with_mqa 等结构。
    - MoE 结构与 DeepSeekV2 MLA 版本兼容：experts 为 DeepseekV2MLP(gate_proj, up_proj, down_proj)，
      可选 shared_experts 结构相同。

    本函数仅对：
      - embedding 与 lm_head 做块级旋转（Q_block）
      - MHA 的 q_proj/k_proj/v_proj 以及 o_proj 做块级旋转
      - MoE / dense MLP 的 up_proj/gate_proj/down_proj 做块级旋转

    DiceQ 可能会提供包含 Q_q_lowrank / Q_kv_lowrank 的 Q 字典，这里会安全忽略低秩部分，仅使用 Q_block。
    """
    q_block_spec = Q
    if isinstance(Q, dict):
        # 仅关心块级旋转矩阵；低秩部分在 OCR MHA 结构中不存在
        q_block_spec = Q.get("Q_block", Q.get("Q"))

    Q_block = _resolve_rotation_matrix(q_block_spec, 0, "Q_block")
    if Q_block is None:
        Q_block = get_orthogonal_matrix(model.config.hidden_size, rotate_mode, device)

    # 旋转 embedding 与 lm_head
    rotate_embeddings(model, Q_block.to(device=device, dtype=torch.float64), device)
    rotate_head(model, Q_block.to(device=device, dtype=torch.float64), device, lm_head_name)
    torch_empty_cache()

    layers, _ = get_module_by_name_prefix(model, layers_node)
    for idx, layer in enumerate(log.pb(layers).title("Rotating DeepSeek V2 MLA-MoE")):
        q_block_layer = _resolve_rotation_matrix(q_block_spec, idx, "Q_block")
        if q_block_layer is None:
            q_block_layer = Q_block
        q_block_layer = q_block_layer.to(device=device, dtype=torch.float64)

        # MHA attention: 使用通用的 q_proj/k_proj/v_proj / o_proj 旋转逻辑
        rotate_attention_inputs(layer, q_block_layer, device)
        rotate_attention_output(layer, q_block_layer, device)

        # MLP：根据是否为 MoE 选择 DeepSeekV2 专用或通用旋转逻辑
        if hasattr(layer.mlp, "experts"):
            rotate_deepseekv2_mlp_input_moe(layer, q_block_layer, device)
            rotate_deepseekv2_mlp_output_moe(layer, q_block_layer, device)
        else:
            rotate_mlp_input(layer, q_block_layer, device)
            rotate_mlp_output(layer, q_block_layer, device)

    return model, Q_block


@torch.inference_mode()
def rotate_model_deepseekv2_mla_moe(
    model: PreTrainedModel,
    rotate_mode: str,
    device: torch.device,
    lm_head_name: str,
    layers_node: str,
    Q=None,
):
    """
    Rotate DeepSeek V2 (MLA + MoE) model weights for improved quantization.
    """
    q_block_spec = Q
    q_q_lowrank_spec = None
    q_kv_lowrank_spec = None
    if isinstance(Q, dict):
        q_block_spec = Q.get("Q_block", Q.get("Q"))
        q_q_lowrank_spec = Q.get("Q_q_lowrank")
        q_kv_lowrank_spec = Q.get("Q_kv_lowrank")

    Q_block = _resolve_rotation_matrix(q_block_spec, 0, "Q_block")
    if Q_block is None:
        Q_block = get_orthogonal_matrix(model.config.hidden_size, rotate_mode, device)

    Q_q_lowrank_default = get_orthogonal_matrix(model.config.q_lora_rank, rotate_mode, device)
    Q_kv_lowrank_default = get_orthogonal_matrix(model.config.kv_lora_rank, rotate_mode, device)

    rotate_embeddings(model, Q_block.to(device=device, dtype=torch.float64), device)
    rotate_head(model, Q_block.to(device=device, dtype=torch.float64), device, lm_head_name)
    torch_empty_cache()

    layers, _ = get_module_by_name_prefix(model, layers_node)
    for idx, layer in enumerate(log.pb(layers).title("Rotating DeepSeek V2 MLA-MoE")):
        q_block_layer = _resolve_rotation_matrix(q_block_spec, idx, "Q_block")
        if q_block_layer is None:
            q_block_layer = Q_block
        q_q_lowrank_layer = _resolve_rotation_matrix(q_q_lowrank_spec, idx, "Q_q_lowrank")
        if q_q_lowrank_layer is None:
            q_q_lowrank_layer = Q_q_lowrank_default
        q_kv_lowrank_layer = _resolve_rotation_matrix(q_kv_lowrank_spec, idx, "Q_kv_lowrank")
        if q_kv_lowrank_layer is None:
            q_kv_lowrank_layer = Q_kv_lowrank_default

        rotate_attention_inputs_mla(layer, q_block_layer.to(device=device, dtype=torch.float64), device)
        rotate_attention_output_mla(layer, q_block_layer.to(device=device, dtype=torch.float64), device)
        rotate_q_low_rank_mla(layer, q_q_lowrank_layer.to(device=device, dtype=torch.float64), device)
        rotate_kv_low_rank_mla(layer, q_kv_lowrank_layer.to(device=device, dtype=torch.float64), device)
        if hasattr(layer.mlp, "experts"):
            rotate_deepseekv2_mlp_input_moe(layer, q_block_layer.to(device=device, dtype=torch.float64), device)
            rotate_deepseekv2_mlp_output_moe(layer, q_block_layer.to(device=device, dtype=torch.float64), device)
        else:
            rotate_mlp_input(layer, q_block_layer.to(device=device, dtype=torch.float64), device)
            rotate_mlp_output(layer, q_block_layer.to(device=device, dtype=torch.float64), device)

    return model, Q_block
