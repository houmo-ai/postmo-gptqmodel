# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

import json
from collections.abc import Mapping
from pathlib import Path

from ..base import BaseQModel
from ..moe_lifecycle import GateUpDownMoELifecycleHooks


def _config_aliases(config):
    head_dim = config.get("qk_rope_head_dim")
    if "head_dim" in config and (head_dim is None or config["head_dim"] != head_dim):
        raise ValueError("Xing head_dim must equal the explicit qk_rope_head_dim")
    theta = config.get("rope_theta")

    def rope_fields(value):
        value = dict(value)
        if "type" in value:
            if "rope_type" in value and value["type"] != value["rope_type"]:
                raise ValueError("Xing RoPE type aliases disagree")
            value["rope_type"] = value.pop("type")
        if theta is not None:
            if "rope_theta" in value and value["rope_theta"] != theta:
                raise ValueError("Xing RoPE theta aliases disagree")
            value["rope_theta"] = theta
        return value

    rope, legacy = config.get("rope_parameters"), config.get("rope_scaling")
    rope = rope_fields(rope) if isinstance(rope, Mapping) else None
    if isinstance(legacy, Mapping):
        legacy = rope_fields(legacy)
        if rope is not None and rope != legacy:
            raise ValueError("Xing legacy and current RoPE declarations disagree")
        rope = legacy
    elif rope is None and "rope_scaling" in config and legacy is None and theta is not None:
        rope = {"rope_type": "default", "rope_theta": theta}
    return {"qk_rope_head_dim": head_dim, "rope_parameters": rope}


def _preserve_source_config(source_dir, save_dir):
    """Keep equivalent head/RoPE fields in the source checkpoint's representation."""
    source = json.loads((Path(source_dir) / "config.json").read_text(encoding="utf-8"))
    path = Path(save_dir) / "config.json"
    saved = json.loads(path.read_text(encoding="utf-8"))
    if _config_aliases(source) != _config_aliases(saved):
        raise ValueError("Saved Xing configuration changed the source attention geometry or RoPE")
    for field in ("head_dim", "rope_parameters", "rope_scaling", "rope_theta"):
        if field in source:
            saved[field] = source[field]
        else:
            saved.pop(field, None)
    path.write_text(json.dumps(saved, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


class Xing4_0QModel(BaseQModel):
    """GPTQModel definition for remote-code Xing4.0 MLA+MoE checkpoints.

    Xing4.0 uses DeepSeek-style MLA attention, ModuleList routed experts, shared
    experts, and HyperConnection residual mixing. HyperConnection parameters and
    MLA/RMSNorm modules stay outside the quantized module tree.
    """

    require_trust_remote_code = True
    require_fast_init = False

    # Early layers are dense MLP; later layers use routed experts.
    layer_modules_strict = False

    # Expand the expert placeholder from config.n_routed_experts.
    dynamic_expert_index = "n_routed_experts"
    pre_lm_head_norm_module = "model.norm"

    # Xing4.0 experts use the gate_proj/up_proj/down_proj convention.
    moe_lifecycle_hooks = GateUpDownMoELifecycleHooks()

    module_tree = [
        "model",
        "layers",
        "#",
        {
            "input_layernorm": ("input_layernorm:!",),
            # MLA projections; q_a_layernorm / kv_a_layernorm are omitted on purpose.
            "self_attn": (
                "q_a_proj:0",
                "q_proj:1",
                "q_b_proj:1",
                "q_absorbed_proj:1",
                "q_rope_proj:1",
                "kv_a_proj_with_mqa:0",
                "kv_b_proj:1",
                "o_proj:2",
            ),
            "post_attention_layernorm": ("post_attention_layernorm:!",),
            # attn_hc / ffn_hc HyperConnection params are intentionally omitted.
            "mlp:moe": {
                # Keep the FP32 router outside weight quantization.
                "gate": ("gate:!",),
                # Dense MLP path used by the first_k_dense_replace layers.
                "": ("gate_proj:0", "up_proj:0", "down_proj:1"),
                "experts": {
                    "#": ("gate_proj:0", "up_proj:0", "down_proj:1"),
                },
                "shared_experts": ("gate_proj:0", "up_proj:0", "down_proj:1"),
            },
        },
    ]

    def before_model_load(self, load_quantized_model=False):
        from .xing4_0_mla import patch_loaded_xing_classes

        patch_loaded_xing_classes()

    def after_model_load(self, model, load_quantized_model=False):
        from .xing4_0_mla import (
            patch_loaded_xing_classes,
            refresh_xing_rotary_buffers,
            validate_xing_mla_model,
        )

        patch_loaded_xing_classes()
        refresh_xing_rotary_buffers(model)
        validate_xing_mla_model(model)
        return model

    def after_model_save(self, save_dir):
        from .xing4_0_mla import XH2_MLA_LAYOUT, finalize_xing_mla_checkpoint

        _preserve_source_config(self.model_local_path, save_dir)
        if getattr(self.model.config, "mla_weight_layout", None) == XH2_MLA_LAYOUT:
            finalize_xing_mla_checkpoint(save_dir)
