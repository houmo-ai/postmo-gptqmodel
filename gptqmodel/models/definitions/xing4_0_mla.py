"""Xing MLA producer layout and checkpoint reconstruction contract.

Only original floating weights may be absorbed. Packed loading constructs the
same Linear boundaries without repeating the matrix products. This module has
no dependency on another model family's implementation or metadata.
"""

from __future__ import annotations

import functools
import json
import sys
from itertools import chain
from pathlib import Path

import torch
from torch import nn

HF_MLA_LAYOUT = "hf_unabsorbed"
XH2_MLA_LAYOUT = "xh2_absorbed_v1"
_CHECKPOINT_HOOK = "# GPTQModel Xing MLA layout hook v1"


def validate_mla_config(config):
    layout = getattr(config, "mla_weight_layout", HF_MLA_LAYOUT)
    transform = getattr(config, "mla_transform", None)
    if layout not in (HF_MLA_LAYOUT, XH2_MLA_LAYOUT):
        raise ValueError(f"Unsupported Xing MLA weight layout: {layout!r}")
    if layout == XH2_MLA_LAYOUT:
        if not isinstance(transform, dict) or any(
            transform.get(key) != value
            for key, value in {
                "version": 1,
                "matmul_dtype": "float32",
                "cache_layout": "separate_k_latent_rope_v_latent",
            }.items()
        ):
            raise ValueError("Invalid Xing MLA transform version or cache layout")
    elif transform is not None:
        raise ValueError("Unabsorbed Xing checkpoint cannot declare an MLA transform")
    return layout


def _source_projections(module, *, from_weights):
    h, c, n, r, v = (
        module.num_heads,
        module.kv_lora_rank,
        module.qk_nope_head_dim,
        module.qk_rope_head_dim,
        module.v_head_dim,
    )
    if min(h, c, n, r, v) <= 0 or r % 2 or module.qk_head_dim != n + r:
        raise ValueError("Invalid Xing MLA dimensions")
    q_name = "q_proj" if module.q_lora_rank is None else "q_b_proj"
    query, kv, output = getattr(module, q_name), module.kv_b_proj, module.o_proj
    for layer in (query, kv, output, module.kv_a_proj_with_mqa):
        if not isinstance(layer, nn.Linear) or any(
            hasattr(layer, attr) for attr in ("qweight", "quant_weight")
        ):
            raise ValueError("Xing MLA absorption requires original floating weights")
        if from_weights and (
            layer.weight.is_meta or not layer.weight.is_floating_point()
        ):
            raise ValueError(
                "Materialize original floating Xing weights before absorption"
            )
    if query.bias is not None or kv.bias is not None:
        raise ValueError("Xing MLA Q-up and KV-up biases cannot be absorbed")
    if (
        (query.out_features, query.in_features)
        != (
            h * (n + r),
            (
                module.config.hidden_size
                if module.q_lora_rank is None
                else module.q_lora_rank
            ),
        )
        or (kv.out_features, kv.in_features) != (h * (n + v), c)
        or (output.out_features, output.in_features)
        != (module.config.hidden_size, h * v)
        or (
            module.kv_a_proj_with_mqa.out_features,
            module.kv_a_proj_with_mqa.in_features,
        )
        != (c + r, module.config.hidden_size)
    ):
        raise ValueError("Xing MLA projection shapes disagree with config")
    return q_name, query, kv, output


def _build_absorbed(module, *, from_weights):
    q_name, query, kv, output = _source_projections(module, from_weights=from_weights)
    h, c, n, r, v = (
        module.num_heads,
        module.kv_lora_rank,
        module.qk_nope_head_dim,
        module.qk_rope_head_dim,
        module.v_head_dim,
    )
    options = {"device": query.weight.device, "dtype": query.weight.dtype}
    q_abs = nn.Linear(query.in_features, h * c, bias=False, **options)
    q_rope = nn.Linear(query.in_features, h * r, bias=False, **options)
    o_abs = nn.Linear(
        h * c, output.out_features, bias=output.bias is not None, **options
    )
    if from_weights:
        with torch.no_grad(), torch.autocast(query.weight.device.type, enabled=False):
            # [out, in]: W_UK.T @ W_Q_nope and W_O_head @ W_UV.
            w_q = query.weight.float().reshape(h, n + r, query.in_features)
            w_kv = kv.weight.to(query.weight.device, torch.float32).reshape(h, n + v, c)
            q_abs.weight.copy_(
                torch.einsum("hnc,hni->hci", w_kv[:, :n], w_q[:, :n]).reshape_as(
                    q_abs.weight
                )
            )
            q_rope.weight.copy_(w_q[:, n:].reshape_as(q_rope.weight))
            w_o = output.weight.to(query.weight.device, torch.float32).reshape(
                output.out_features, h, v
            )
            o_abs.weight.copy_(
                torch.einsum("xhv,hvc->xhc", w_o, w_kv[:, n:]).reshape_as(o_abs.weight)
            )
            if output.bias is not None:
                o_abs.bias.copy_(output.bias)
            if not all(
                torch.isfinite(p).all()
                for layer in (q_abs, q_rope, o_abs)
                for p in layer.parameters()
            ):
                raise FloatingPointError("Non-finite absorbed Xing MLA weights")
    module.q_absorbed_proj, module.q_rope_proj, module.o_proj = q_abs, q_rope, o_abs
    delattr(module, q_name)
    del module.kv_b_proj
    module._mla_weight_layout = XH2_MLA_LAYOUT


def absorbed_mla_forward(
    self,
    hidden_states,
    position_embeddings,
    attention_mask,
    past_key_values=None,
    **kwargs,
):
    """Native eager attention with absorbed boundaries and independent K/V."""
    from transformers.cache_utils import DynamicCache

    native = sys.modules[type(self).__module__]
    b, s, _ = hidden_states.shape
    h, c, r = self.num_heads, self.kv_lora_rank, self.qk_rope_head_dim
    q_input = hidden_states
    if self.q_lora_rank is not None:
        q_input = self.q_a_layernorm(self.q_a_proj(q_input))
    q_content = self.q_absorbed_proj(q_input).view(b, s, h, c).transpose(1, 2)
    q_rope = self.q_rope_proj(q_input).view(b, s, h, r).transpose(1, 2)
    latent, k_rope = self.kv_a_proj_with_mqa(hidden_states).split((c, r), dim=-1)
    latent = self.kv_a_layernorm(latent).unsqueeze(1)
    rope = (
        native.apply_rotary_pos_emb_interleave
        if self.config.rope_interleave
        else native.apply_rotary_pos_emb
    )
    q_rope, k_rope = rope(q_rope, k_rope.unsqueeze(1), *position_embeddings)
    query = torch.cat((q_content, q_rope), dim=-1)
    key, value = torch.cat((latent, k_rope), dim=-1), latent.clone()
    if past_key_values is not None:
        if not isinstance(past_key_values, DynamicCache):
            raise ValueError("Absorbed Xing MLA requires DynamicCache")
        if self.layer_idx < len(past_key_values.layers):
            cached = past_key_values.layers[self.layer_idx]
            if getattr(cached, "is_initialized", False) and (
                cached.keys.shape[1] != 1
                or cached.keys.shape[-1] != c + r
                or cached.values.shape[1] != 1
                or cached.values.shape[-1] != c
            ):
                raise ValueError("Cannot reuse an unabsorbed/foreign Xing MLA cache")
        key, value = past_key_values.update(key, value, self.layer_idx)
    if attention_mask is None or attention_mask.ndim == 2:
        # HF's SDPA mask builder may elide the mask and rely on is_causal.
        # This reference executes eager attention at the absorbed boundary.
        key_length = key.shape[-2]
        future = torch.arange(key_length, device=query.device)[None, :] > (
            torch.arange(s, device=query.device)[:, None] + key_length - s
        )
        causal_mask = query.new_zeros(s, key_length).masked_fill(future, float("-inf"))
        if attention_mask is None:
            attention_mask = causal_mask
        else:
            attention_mask = (
                causal_mask[None, None]
                .expand(b, 1, s, key_length)
                .masked_fill(
                    ~attention_mask[:, None, None, :key_length].bool(), float("-inf")
                )
            )
    elif attention_mask.ndim != 4:
        raise ValueError(
            "Xing MLA attention_mask must be 2D padding or 4D attention mask"
        )
    elif attention_mask.dtype == torch.bool:
        attention_mask = query.new_zeros(attention_mask.shape).masked_fill(
            ~attention_mask, float("-inf")
        )
    output, weights = native.eager_attention_forward(
        self,
        query,
        key,
        value,
        attention_mask,
        scaling=self.scaling,
        dropout=self.attention_dropout if self.training else 0.0,
    )
    return self.o_proj(output.reshape(b, s, h * c)), weights


def install_xing_mla_class(attention_cls):
    if attention_cls.__dict__.get("_gptqmodel_mla_installed", False):
        return
    original_init, original_forward = attention_cls.__init__, attention_cls.forward

    @functools.wraps(original_init)
    def init(self, config, layer_idx, *args, **kwargs):
        layout = validate_mla_config(config)
        original_init(self, config, layer_idx, *args, **kwargs)
        if layout == XH2_MLA_LAYOUT:
            _build_absorbed(self, from_weights=False)

    @functools.wraps(original_forward)
    def forward(self, *args, **kwargs):
        if getattr(self, "_mla_weight_layout", None) == XH2_MLA_LAYOUT:
            return absorbed_mla_forward(self, *args, **kwargs)
        return original_forward(self, *args, **kwargs)

    attention_cls.__init__, attention_cls.forward = init, forward
    attention_cls._gptqmodel_mla_installed = True


def absorb_xing_mla(model):
    if validate_mla_config(model.config) != HF_MLA_LAYOUT or getattr(
        model.config, "quantization_config", None
    ):
        raise ValueError(
            "Start Xing MLA quantization from an original unquantized checkpoint"
        )
    modules = [layer.self_attn for layer in model.model.layers]
    if not modules:
        raise ValueError("No Xing MLA decoder found")
    # Validate the entire decoder before changing any module.
    for module in modules:
        _source_projections(module, from_weights=True)
    for module in modules:
        install_xing_mla_class(type(module))
        _build_absorbed(module, from_weights=True)
    model.config.mla_weight_layout = XH2_MLA_LAYOUT
    model.config.mla_transform = {
        "version": 1,
        "matmul_dtype": "float32",
        "weight_dtype": str(modules[0].q_absorbed_proj.weight.dtype),
        "cache_layout": "separate_k_latent_rope_v_latent",
    }
    return model


def validate_xing_mla_model(model):
    layout = validate_mla_config(model.config)
    for layer in model.model.layers:
        attention = layer.self_attn
        absorbed = all(
            hasattr(attention, name) for name in ("q_absorbed_proj", "q_rope_proj")
        )
        raw = hasattr(attention, "kv_b_proj")
        if (layout == XH2_MLA_LAYOUT) != absorbed or (layout == HF_MLA_LAYOUT) != raw:
            raise ValueError(
                "Xing MLA checkpoint has mixed layouts or a missing construction hook"
            )
        if absorbed and any(
            hasattr(attention, name) for name in ("q_proj", "q_b_proj")
        ):
            raise ValueError("Xing MLA checkpoint has mixed query layouts")


def refresh_xing_rotary_buffers(model):
    for module in model.modules():
        if type(module).__name__ != "Xing4_0RotaryEmbedding" or module.inv_freq.is_meta:
            continue
        # Reconstruct only derived, non-persistent buffers from the native config.
        fresh = type(module)(module.config, device=module.inv_freq.device)
        module.register_buffer("inv_freq", fresh.inv_freq, persistent=False)
        module.register_buffer(
            "original_inv_freq", fresh.original_inv_freq, persistent=False
        )
        module.attention_scaling = fresh.attention_scaling


def prepare_xing_quantization_dtype(model, dtype):
    """Cast calibration weights while retaining HF's strict FP32 tensors."""
    strict = set(getattr(model, "_keep_in_fp32_modules_strict", ()))
    retained = {
        name: tensor.data
        for name, tensor in chain(model.named_parameters(), model.named_buffers())
        if name in strict or strict.intersection(name.split("."))
    }
    model.to(dtype=dtype)
    for name, data in retained.items():
        parent, _, leaf = name.rpartition(".")
        getattr(model.get_submodule(parent), leaf).data = data
    refresh_xing_rotary_buffers(model)


def install_xing_model_class(model_cls):
    if model_cls.__dict__.get("_gptqmodel_mla_load_installed", False):
        return
    original = model_cls.from_pretrained.__func__
    original_init = model_cls.__init__

    @functools.wraps(original_init)
    def init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        # Accelerate matches full tensor keys or module prefixes. Expand the
        # native leaf-name declaration so packed reload also preserves buffers.
        strict = set(self._keep_in_fp32_modules_strict or ())
        self._keep_in_fp32_modules_strict = {
            name
            for name, _ in chain(self.named_parameters(), self.named_buffers())
            if name in strict or strict.intersection(name.split("."))
        }

    @classmethod
    @functools.wraps(original)
    def from_pretrained(cls, *args, **kwargs):
        requested_info = kwargs.pop("output_loading_info", False)
        model, info = original(cls, *args, output_loading_info=True, **kwargs)
        if validate_mla_config(model.config) == XH2_MLA_LAYOUT and any(
            info.get(key)
            for key in (
                "missing_keys",
                "unexpected_keys",
                "mismatched_keys",
                "error_msgs",
            )
        ):
            raise ValueError(f"Invalid absorbed Xing checkpoint tensors: {info}")
        refresh_xing_rotary_buffers(model)
        validate_xing_mla_model(model)
        return (model, info) if requested_info else model

    model_cls.__init__ = init
    model_cls.from_pretrained = from_pretrained
    model_cls._gptqmodel_mla_load_installed = True


def patch_loaded_xing_classes():
    # HF may reuse an already registered class when two checkpoint configs
    # contain identical remote code. Keep both old and new classes config-gated.
    for module_name, module in tuple(sys.modules.items()):
        if module is None or "modeling_xing4_0" not in module_name:
            continue
        cls = getattr(module, "Xing4_0Attention", None)
        if isinstance(cls, type):
            install_xing_mla_class(cls)
        cls = getattr(module, "Xing4_0ForCausalLM", None)
        if isinstance(cls, type):
            install_xing_model_class(cls)
        cls = getattr(module, "Xing4_0PreTrainedModel", None)
        if isinstance(cls, type) and not cls.__dict__.get(
            "_gptqmodel_hc_init_installed", False
        ):
            original_init = cls._init_weights

            def init_weights(self, child, _original=original_init):
                # The source initializer still uses pre-rename fn/base/scale.
                if type(child).__name__ == "Xing4_0HyperConnection":
                    nn.init.normal_(
                        child.hc_fn, mean=0.0, std=self.config.initializer_range
                    )
                    nn.init.zeros_(child.hc_base)
                    nn.init.ones_(child.hc_scale)
                else:
                    _original(self, child)

            cls._init_weights = init_weights
            cls._gptqmodel_hc_init_installed = True


def finalize_xing_mla_checkpoint(output_dir):
    root = Path(output_dir)
    config = json.loads((root / "config.json").read_text())
    from types import SimpleNamespace

    if validate_mla_config(SimpleNamespace(**config)) != XH2_MLA_LAYOUT:
        raise ValueError("Cannot finalize an unabsorbed Xing checkpoint")
    reference = config.get("auto_map", {}).get("AutoModelForCausalLM")
    if reference != "modeling_xing4_0.Xing4_0ForCausalLM":
        raise ValueError(
            f"Unexpected Xing checkpoint model implementation: {reference!r}"
        )
    path = root / "modeling_xing4_0.py"
    source = path.read_text(encoding="utf-8")
    if _CHECKPOINT_HOOK not in source:
        path.write_text(
            source + f"\n{_CHECKPOINT_HOOK}\n"
            "from gptqmodel.models.definitions.xing4_0_mla import patch_loaded_xing_classes\n"
            "patch_loaded_xing_classes()\n",
            encoding="utf-8",
        )
