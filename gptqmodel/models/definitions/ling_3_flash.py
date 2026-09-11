# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

"""GPTQModel definition for Ling-3-Flash (``bailing_hybrid``)."""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM

from ..base import BaseQModel
from ..moe_lifecycle import GateUpDownMoELifecycleHooks


def _shield_short_convolutions_from_gptq_hooks(model: torch.nn.Module) -> int:
    """Keep Ling's FLA convolution contract without changing GPTQ core.

    FLA ``ShortConvolution`` inherits ``nn.Conv1d`` but accepts and returns
    recurrent cache state. GPTQ's generic input-capture pass otherwise sees a
    Conv1d and replaces it with a tensor-only wrapper. Give each loaded Ling
    instance a model-local class view that keeps the exact implementation and
    state-dict layout while no longer advertising itself as ``nn.Conv1d``.
    """

    shielded = 0
    for module in model.modules():
        module_cls = type(module)
        if module_cls.__name__ != "ShortConvolution" or not isinstance(
            module, torch.nn.Conv1d
        ):
            continue

        safe_cls = getattr(module_cls, "_gptqmodel_ling_safe_class", None)
        if safe_cls is None:
            attrs = {
                name: value
                for name, value in module_cls.__dict__.items()
                if name not in {"__dict__", "__weakref__", "__classcell__"}
            }
            attrs["__module__"] = module_cls.__module__
            attrs["_gptqmodel_ling_original_class"] = module_cls
            safe_cls = type(
                f"GPTQModel{module_cls.__name__}",
                (torch.nn.Module,),
                attrs,
            )
            module_cls._gptqmodel_ling_safe_class = safe_cls

        module.__class__ = safe_cls
        shielded += 1
    return shielded


def _patch_config_class(config_cls: type) -> None:
    # The checkpoint's remote class declares no model_type even though its
    # config.json correctly stores bailing_hybrid.  Transformers serializes
    # the class value and would otherwise create an unloadable quantized copy.
    config_cls.model_type = "bailing_hybrid"

    if getattr(config_cls, "_gptqmodel_ling_rope_compat", False):
        return

    def _get_rope_scaling(instance):
        value = instance.__dict__.get("rope_scaling", None)
        if (
            isinstance(value, dict)
            and value.get("rope_type") == "default"
            and "factor" not in value
        ):
            return None
        return value

    def _set_rope_scaling(instance, value):
        instance.__dict__["rope_scaling"] = value

    config_cls.rope_scaling = property(_get_rope_scaling, _set_rope_scaling)
    config_cls._gptqmodel_ling_rope_compat = True


def patch_ling_3_flash_remote_code() -> None:
    """Bridge Ling's checkpoint code to current Transformers releases."""

    import transformers.modeling_rope_utils as rope_utils
    import transformers.utils.import_utils as import_utils

    if not hasattr(import_utils, "is_torch_fx_available"):
        import_utils.is_torch_fx_available = lambda: True

    if "default" not in rope_utils.ROPE_INIT_FUNCTIONS:

        def _default_rope(config=None, device=None, seq_len=None, layer_type=None):
            del seq_len, layer_type
            head_dim = getattr(config, "head_dim", None) or (
                config.hidden_size // config.num_attention_heads
            )
            dim = int(head_dim * float(getattr(config, "partial_rotary_factor", 1.0)))
            theta = float(getattr(config, "rope_theta", 10000.0))
            positions = torch.arange(0, dim, 2, dtype=torch.int64, device=device).float()
            inv_freq = 1.0 / (theta ** (positions / dim))
            return inv_freq, 1.0

        rope_utils.ROPE_INIT_FUNCTIONS["default"] = _default_rope

    # AutoConfig imports trust_remote_code modules before before_model_load is
    # called.  Patch the dynamically named module that is already resident.
    for module_name, module in tuple(sys.modules.items()):
        if module is None:
            continue
        if "configuration_bailing_moe_v3" in module_name:
            config_cls = getattr(module, "BailingMoeV3Config", None)
            if isinstance(config_cls, type):
                _patch_config_class(config_cls)
        elif "modeling_bailing_moe_v3" in module_name:
            # HF caches config classes by source content. Two checkpoints
            # named "out" can therefore share a config class even when one
            # carries newly absorbed model code. AutoModel may reuse the
            # earlier registered model class instead of importing that code.
            # Keep every loaded Ling class aware of the config-gated ABI;
            # ordinary/MTP layers retain their original construction.
            from .ling3_flash_mla import (
                install_ling_mla_class,
                install_ling_mla_model_class,
            )

            attention_cls = getattr(module, "BailingMoeV3MultiLatentAttention", None)
            if isinstance(attention_cls, type):
                install_ling_mla_class(attention_cls)
            causal_cls = getattr(module, "BailingMoeV3ForCausalLM", None)
            if isinstance(causal_cls, type):
                install_ling_mla_model_class(causal_cls)
            for model_cls in vars(module).values():
                if (
                    isinstance(model_cls, type)
                    and model_cls.__module__ == module_name
                    and isinstance(getattr(model_cls, "_tied_weights_keys", None), list)
                ):
                    # Transformers 5.x expects this metadata to be a mapping.
                    # Ling does not tie the input and output embeddings.
                    model_cls._tied_weights_keys = {}


def load_ling_3_flash_config(model_dir: str):
    """Load the checkpoint config after normalizing its remote class metadata."""
    from transformers.dynamic_module_utils import get_class_from_dynamic_module

    patch_ling_3_flash_remote_code()
    config_cls = get_class_from_dynamic_module(
        "configuration_bailing_moe_v3.BailingMoeV3Config", str(model_dir),
    )
    _patch_config_class(config_cls)
    config = config_cls.from_pretrained(str(model_dir))
    config._name_or_path = str(Path(model_dir).expanduser().resolve())
    return config


def refresh_ling_rotary_buffers(model):
    """Rebuild derived RoPE buffers after HF meta loading.

    Ling's remote _init_weights only initializes Linear/Embedding. HF 5.x can
    materialize its non-persistent inv_freq from meta as uninitialized storage;
    it is absent from the checkpoint, so a clean missing-keys report is not
    sufficient. Recompute from the rotary module's own (RoPE-dimension) config.
    """
    for module in model.modules():
        if type(module).__name__ != "BailingMoeV3RotaryEmbedding":
            continue
        device = module.inv_freq.device
        if device.type == "meta":
            continue  # Empty export skeleton: materialize at its normal boundary.
        inv_freq, scaling = module.rope_init_fn(module.config, device)
        module.register_buffer("inv_freq", inv_freq.float(), persistent=False)
        module.original_inv_freq = module.inv_freq
        module.attention_scaling = scaling


class Ling3FlashQModel(BaseQModel):
    """Quantization layout for Ling-3-Flash's hybrid KDA/MLA decoder."""

    loader = AutoModelForCausalLM
    require_trust_remote_code = True
    dynamic_expert_index = "num_experts"
    layer_modules_strict = False
    pre_lm_head_norm_module = "model.norm"
    moe_lifecycle_hooks = GateUpDownMoELifecycleHooks()

    # KDA q/k/v and output-gate paths, all MLA boundaries, dense MLPs and the
    # shared expert use the base quantization policy (W8 in the Ling recipe).
    # The decay/beta f*/b paths stay in floating point, matching the Qwen3.5
    # linear-attention policy. Routed experts are changed to W4 through the
    # caller's QuantizeConfig.dynamic rule.
    module_tree = [
        "model",
        "layers",
        "#",
        {
            "input_layernorm": ("input_layernorm:!",),
            "attention": (
                "q_proj:0",
                "k_proj:0",
                "v_proj:0",
                "q_a_proj:0",
                "kv_a_proj_with_mqa:0",
                "g_proj:0",
                "g_a_proj:0",
                "f_proj:!",
                "f_a_proj:!",
                "b_proj:!",
                "q_b_proj:1",
                "kv_b_proj:1",
                "q_absorbed_proj:1",
                "q_rope_proj:1",
                "g_b_proj:1",
                "f_b_proj:!",
                "o_proj:2",
                "dense:2",
                "absorbed_dense:2",
            ),
            "post_attention_layernorm": ("post_attention_layernorm:!",),
            "mlp:0": ("gate_proj:0", "up_proj:0", "down_proj:1"),
            "mlp:moe:?": {
                "gate": ("gate:!",),
                "shared_experts": ("gate_proj:0", "up_proj:0", "down_proj:1"),
                "experts": {
                    "#": ("gate_proj:0", "up_proj:0", "down_proj:1"),
                },
            },
        },
    ]

    def before_model_load(self, load_quantized_model=False):
        del self, load_quantized_model
        patch_ling_3_flash_remote_code()

    def after_model_load(self, model, load_quantized_model=False):
        del load_quantized_model
        patch_ling_3_flash_remote_code()
        # Also normalize classes reached through the concrete model.  This
        # covers a fresh dynamic-module import that occurred during loading.
        for submodule in model.modules():
            model_cls = type(submodule)
            if isinstance(getattr(model_cls, "_tied_weights_keys", None), list):
                model_cls._tied_weights_keys = {}
        _shield_short_convolutions_from_gptq_hooks(model)
        refresh_ling_rotary_buffers(model)
        from .ling3_flash_mla import validate_ling_mla_model

        validate_ling_mla_model(model)
        return model

    def after_model_save(self, save_dir):
        from .ling3_flash_mla import XH2_MLA_LAYOUT, finalize_ling_mla_checkpoint

        if getattr(self.model.config, "ling_mla_layout", None) == XH2_MLA_LAYOUT:
            finalize_ling_mla_checkpoint(save_dir)


__all__ = [
    "Ling3FlashQModel",
    "load_ling_3_flash_config",
    "_shield_short_convolutions_from_gptq_hooks",
    "patch_ling_3_flash_remote_code",
]
