"""XH2 Ling MLA weight layout, shared by GPTQ and AutoRound.

This is a floating-point reference for the *weight* boundaries used on XH2,
not an emulation of SEFP activations. K and V have independent cache payloads.
The checkpoint-side hook installs the same structure before packed loading.
"""

from __future__ import annotations

import ast
import functools
import json
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

XH2_MLA_LAYOUT = "xh2_absorbed_v1"
HF_MLA_LAYOUT = "hf_unabsorbed"
_CHECKPOINT_HOOK = "# GPTQModel Ling XH2 MLA layout hook v1"


def validate_mla_layout(layout: str) -> str:
    if layout not in (HF_MLA_LAYOUT, XH2_MLA_LAYOUT):
        raise ValueError(f"Unsupported Ling MLA weight layout: {layout!r}")
    return layout


def _validate_attention(module):
    if module.gated_attention_proj_granularity_type not in (None, "head_wise"):
        raise ValueError(
            "xh2_absorbed_v1 supports no gate or head_wise gate only; "
            "element_wise requires value-up before the gate and cannot absorb dense"
        )
    if not module.config.rope_interleave:
        raise ValueError("xh2_absorbed_v1 requires Ling's interleaved RoPE weights")
    h, c, n, r, v = (
        module.num_heads,
        module.kv_lora_rank,
        module.qk_nope_head_dim,
        module.qk_rope_head_dim,
        module.v_head_dim,
    )
    if min(h, c, n, r, v) <= 0 or r % 2:
        raise ValueError("Invalid Ling MLA head/latent/RoPE dimensions")
    return h, c, n, r, v


def _build_absorbed(module, *, from_weights: bool):
    h, c, n, r, v = _validate_attention(module)
    q_name = "q_proj" if module.q_lora_rank is None else "q_b_proj"
    query, kv, output = getattr(module, q_name), module.kv_b_proj, module.dense
    for projection in (query, kv, output):
        if not isinstance(projection, nn.Linear) or any(
            hasattr(projection, attr) for attr in ("qweight", "quant_weight")
        ):
            raise ValueError(
                "MLA absorption requires original floating weights, not quantized projections"
            )
    if query.bias is not None or kv.bias is not None:
        raise ValueError(
            "Ling MLA Q-up and KV-up biases are not supported by this layout"
        )
    if query.out_features != h * (n + r) or (kv.out_features, kv.in_features) != (
        h * (n + v),
        c,
    ):
        raise ValueError("Ling MLA projection shapes disagree with config")
    if (output.in_features, output.out_features) != (h * v, module.config.hidden_size):
        raise ValueError("Ling MLA output projection shape disagrees with config")
    options = {"device": query.weight.device, "dtype": query.weight.dtype}
    q_abs = nn.Linear(query.in_features, h * c, bias=False, **options)
    q_rope = nn.Linear(query.in_features, h * r, bias=False, **options)
    o_abs = nn.Linear(
        h * c, output.out_features, bias=output.bias is not None, **options
    )
    if from_weights:
        if any(layer.weight.is_meta for layer in (query, kv, output)):
            raise ValueError(
                "Materialize original Ling weights before MLA absorption; disk offload is unsupported"
            )
        with torch.no_grad():
            w_kv = kv.weight.float().reshape(h, n + v, c)
            w_q = query.weight.float().reshape(h, n + r, query.in_features)
            # [out, in] convention: W_UK.T @ W_Q_nope, W_O_head @ W_UV.
            q_abs.weight.copy_(
                torch.einsum("hnc,hni->hci", w_kv[:, :n], w_q[:, :n]).reshape_as(
                    q_abs.weight
                )
            )
            q_rope.weight.copy_(w_q[:, n:].reshape_as(q_rope.weight))
            w_o = output.weight.float().reshape(output.out_features, h, v)
            o_abs.weight.copy_(
                torch.einsum("xhv,hvc->xhc", w_o, w_kv[:, n:]).reshape_as(o_abs.weight)
            )
            if output.bias is not None:
                o_abs.bias.copy_(output.bias)
            if not all(
                torch.isfinite(layer.weight).all() for layer in (q_abs, q_rope, o_abs)
            ):
                raise FloatingPointError(
                    "Non-finite absorbed Ling MLA weights; refusing quantization"
                )
    module.q_absorbed_proj, module.q_rope_proj, module.absorbed_dense = (
        q_abs,
        q_rope,
        o_abs,
    )
    delattr(module, q_name)
    del module.kv_b_proj, module.dense
    module._ling_mla_layout = XH2_MLA_LAYOUT


def _rope_interleaved(states, cos, sin):
    shape = states.shape
    states = (
        states.reshape(*shape[:-1], shape[-1] // 2, 2).transpose(-1, -2).reshape(shape)
    )
    first, second = states.chunk(2, dim=-1)
    return states * cos.unsqueeze(1) + torch.cat(
        (-second, first), dim=-1
    ) * sin.unsqueeze(1)


def absorbed_mla_forward(
    self,
    hidden_states,
    position_embeddings,
    attention_mask,
    past_key_values=None,
    cache_position=None,
    **kwargs,
):
    """Eager reference; cache dimensions are K=[B,1,T,C+R], V=[B,1,T,C]."""
    if "past_key_value" in kwargs:
        if past_key_values is not None:
            raise ValueError("Specify only one of past_key_value / past_key_values")
        past_key_values = kwargs.pop("past_key_value")
    batch, length, _ = hidden_states.shape
    h, c, r = self.num_heads, self.kv_lora_rank, self.qk_rope_head_dim
    q_input = hidden_states
    if self.q_lora_rank is not None:
        q_input = self.q_a_layernorm(self.q_a_proj(q_input))
    q_content = self.q_absorbed_proj(q_input).view(batch, length, h, c).transpose(1, 2)
    q_rope = self.q_rope_proj(q_input).view(batch, length, h, r).transpose(1, 2)
    compressed, k_rope = self.kv_a_proj_with_mqa(hidden_states).split((c, r), dim=-1)
    latent = self.kv_a_layernorm(compressed).unsqueeze(1)
    cos, sin = position_embeddings
    q_rope = _rope_interleaved(q_rope, cos, sin)
    k_rope = _rope_interleaved(k_rope.unsqueeze(1), cos, sin)
    query = torch.cat((q_content, q_rope), dim=-1)
    key = torch.cat((latent, k_rope), dim=-1)
    # Do not alias the two caches, even though both contain the same FP latent.
    # XH2 quantizes K along head_dim, and V along the token axis independently.
    value = latent.contiguous()
    if past_key_values is not None:
        from transformers.cache_utils import DynamicCache

        if not isinstance(past_key_values, DynamicCache):
            raise ValueError(
                "xh2_absorbed_v1 reference supports DynamicCache, not raw-layout static caches"
            )
        if self.layer_idx < len(past_key_values.layers):
            cached = past_key_values.layers[self.layer_idx]
            if getattr(cached, "is_initialized", False) and (
                cached.keys.shape[1] != 1
                or cached.keys.shape[-1] != c + r
                or cached.values.shape[1] != 1
                or cached.values.shape[-1] != c
            ):
                raise ValueError(
                    "Cannot reuse an unabsorbed/foreign cache with xh2_absorbed_v1"
                )
        key, value = past_key_values.update(
            key,
            value,
            self.layer_idx,
            {"sin": sin, "cos": cos, "cache_position": cache_position},
        )
    scores = torch.matmul(query, key.transpose(-1, -2)) * self.scaling
    # Use the original attention scale, NOT (C + R)**-0.5.
    key_length = key.shape[-2]
    positions = cache_position
    if positions is None:
        positions = torch.arange(key_length - length, key_length, device=query.device)
    causal = torch.arange(key_length, device=query.device) <= positions.to(
        query.device
    ).reshape(-1, 1)
    scores = scores.masked_fill(~causal, torch.finfo(scores.dtype).min)
    if attention_mask is not None:
        mask = attention_mask[..., :key_length]
        if mask.ndim == 2:
            scores = scores.masked_fill(
                ~mask[:, None, None, :].bool(), torch.finfo(scores.dtype).min
            )
        elif mask.ndim == 4:
            if mask.dtype == torch.bool:
                scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
            else:
                scores = scores + mask
        else:
            raise ValueError(
                "Ling MLA attention_mask must be 2D padding or 4D attention mask"
            )
    probabilities = F.softmax(scores, dim=-1, dtype=torch.float32).to(query.dtype)
    probabilities = F.dropout(
        probabilities, self.attention_dropout, training=self.training
    )
    context = torch.matmul(probabilities, value).transpose(1, 2)
    if self.g_proj is not None:
        gate = self.g_proj(hidden_states).float().sigmoid().to(hidden_states.dtype)
        context = context * gate.unsqueeze(-1)
    output = self.absorbed_dense(context.reshape(batch, length, h * c))
    return output, probabilities, past_key_values


def install_ling_mla_class(attention_cls):
    """Install a config-gated constructor/forward on a checkpoint's own class."""
    if attention_cls.__dict__.get("_gptqmodel_xh2_mla_installed", False):
        return
    original_init, original_forward = attention_cls.__init__, attention_cls.forward

    @functools.wraps(original_init)
    def init(self, config, layer_idx, *args, **kwargs):
        layout = validate_mla_layout(getattr(config, "ling_mla_layout", HF_MLA_LAYOUT))
        original_init(self, config, layer_idx, *args, **kwargs)
        # MTP is kept in its original floating layout and excluded from tuning.
        if layout == XH2_MLA_LAYOUT and layer_idx < config.num_hidden_layers:
            _build_absorbed(self, from_weights=False)

    @functools.wraps(original_forward)
    def forward(self, *args, **kwargs):
        if getattr(self, "_ling_mla_layout", None) == XH2_MLA_LAYOUT:
            return absorbed_mla_forward(self, *args, **kwargs)
        return original_forward(self, *args, **kwargs)

    attention_cls.__init__, attention_cls.forward = init, forward
    attention_cls._gptqmodel_xh2_mla_installed = True


def absorb_ling_mla(model):
    """Absorb original FP decoder weights once, before any quantization hooks."""
    from .ling_3_flash import patch_ling_3_flash_remote_code

    patch_ling_3_flash_remote_code()
    layout = validate_mla_layout(
        getattr(model.config, "ling_mla_layout", HF_MLA_LAYOUT)
    )
    if layout != HF_MLA_LAYOUT or getattr(model.config, "quantization_config", None):
        raise ValueError(
            "Start Ling MLA quantization from the original unquantized HF checkpoint"
        )
    modules = [
        layer.attention
        for layer in model.model.layers[: model.config.num_hidden_layers]
        if type(layer.attention).__name__ == "BailingMoeV3MultiLatentAttention"
    ]
    if not modules:
        raise ValueError("No Ling MLA decoder attention found")
    for module in modules:
        _validate_attention(module)
    for module in modules:
        install_ling_mla_class(type(module))
        _build_absorbed(module, from_weights=True)
    model.config.ling_mla_layout = XH2_MLA_LAYOUT
    model.config.ling_mla_transform = {
        "version": 1,
        "matmul_dtype": "float32",
        "weight_dtype": str(modules[0].q_absorbed_proj.weight.dtype),
        "decoder_only": True,
        "cache_layout": "separate_k_latent_rope_v_latent",
    }
    return model


def validate_ling_mla_model(model):
    """Catch missing checkpoint hooks instead of accepting random raw projections."""
    layout = validate_mla_layout(
        getattr(model.config, "ling_mla_layout", HF_MLA_LAYOUT)
    )
    if layout == HF_MLA_LAYOUT:
        return
    found = 0
    for layer in model.model.layers[: model.config.num_hidden_layers]:
        module = layer.attention
        if type(module).__name__ != "BailingMoeV3MultiLatentAttention":
            continue
        found += 1
        _validate_attention(module)
        if (
            getattr(module, "_ling_mla_layout", None) != XH2_MLA_LAYOUT
            or any(
                not hasattr(module, name)
                for name in ("q_absorbed_proj", "q_rope_proj", "absorbed_dense")
            )
            or any(
                hasattr(module, name)
                for name in ("q_proj", "q_b_proj", "kv_b_proj", "dense")
            )
        ):
            raise ValueError(
                "Ling xh2_absorbed_v1 checkpoint is missing its model-construction hook or has mixed layouts"
            )
    if not found:
        raise ValueError("Ling xh2_absorbed_v1 checkpoint has no MLA decoder")


def install_ling_mla_model_class(model_cls):
    """Also repair derived RoPE state for standalone HF checkpoint loading."""
    if model_cls.__dict__.get("_gptqmodel_xh2_load_installed", False):
        return
    original = model_cls.from_pretrained.__func__

    @classmethod
    @functools.wraps(original)
    def from_pretrained(cls, *args, **kwargs):
        result = original(cls, *args, **kwargs)
        model = result[0] if isinstance(result, tuple) else result
        if getattr(model.config, "ling_mla_layout", None) == XH2_MLA_LAYOUT:
            from .ling_3_flash import refresh_ling_rotary_buffers

            refresh_ling_rotary_buffers(model)
            validate_ling_mla_model(model)
        return result

    model_cls.from_pretrained = from_pretrained
    model_cls._gptqmodel_xh2_load_installed = True


def finalize_ling_mla_checkpoint(output_dir):
    """Make HF/GPTQModel reconstruction self-describing, including later resaves.

    Only the newly saved checkpoint's remote code is annotated; source model
    files are never edited. HF and GPTQModel already preserve these .py files
    on resave. An old backend fails to import the hook instead of silently
    loading fused tensors into an unabsorbed model.
    """
    root = Path(output_dir).resolve()
    config = json.loads((root / "config.json").read_text())
    if config.get("ling_mla_layout") != XH2_MLA_LAYOUT:
        raise ValueError(
            "Cannot finalize a checkpoint without xh2_absorbed_v1 metadata"
        )
    reference = config.get("auto_map", {}).get("AutoModelForCausalLM", "")
    if reference != "modeling_bailing_moe_v3.BailingMoeV3ForCausalLM":
        raise ValueError(
            f"Unexpected Ling checkpoint model implementation: {reference!r}"
        )
    source_path = root / "modeling_bailing_moe_v3.py"
    source = source_path.read_text(encoding="utf-8")
    if _CHECKPOINT_HOOK in source:
        return
    tree = ast.parse(source)
    insert_after = 0
    for node in tree.body:
        if (
            isinstance(node, ast.Expr)
            and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, str)
        ):
            insert_after = node.end_lineno
        elif isinstance(node, ast.ImportFrom) and node.module == "__future__":
            insert_after = node.end_lineno
        else:
            break
    lines = source.splitlines(keepends=True)
    header = (
        f"\n{_CHECKPOINT_HOOK}\n"
        "from gptqmodel.models.definitions.ling_3_flash import patch_ling_3_flash_remote_code as _ling_compat\n"
        "_ling_compat()\n\n"
    )
    footer = (
        "\nfrom gptqmodel.models.definitions.ling3_flash_mla import install_ling_mla_class as _install_ling_mla\n"
        "_install_ling_mla(BailingMoeV3MultiLatentAttention)\n"
        "from gptqmodel.models.definitions.ling3_flash_mla import install_ling_mla_model_class as _install_ling_model\n"
        "_install_ling_model(BailingMoeV3ForCausalLM)\n"
        "_ling_compat()\n"
    )
    source_path.write_text(
        "".join(lines[:insert_after]) + header + "".join(lines[insert_after:]) + footer,
        encoding="utf-8",
    )
