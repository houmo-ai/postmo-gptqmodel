# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from ..base import BaseQModel
from ..moe_lifecycle import GateUpDownMoELifecycleHooks


class MiniCPM5MoEQModel(BaseQModel):
    """GPTQModel definition for remote-code MiniCPM5-MoE checkpoints."""

    require_trust_remote_code = True
    require_fast_init = False

    # Layer 0 is dense while the remaining layers use routed experts.
    layer_modules_strict = False

    # Expand the expert placeholder from config.n_routed_experts.
    dynamic_expert_index = "n_routed_experts"
    pre_lm_head_norm_module = "model.norm"

    # MiniCPM5 experts use the gate_proj/up_proj/down_proj convention.
    moe_lifecycle_hooks = GateUpDownMoELifecycleHooks()

    module_tree = [
        "model",
        "layers",
        "#",
        {
            "input_layernorm": ("input_layernorm:!",),
            "self_attn": ("q_proj:0", "k_proj:0", "v_proj:0", "o_proj:1"),
            "post_attention_layernorm": ("post_attention_layernorm:!",),
            "mlp:moe": {
                # Keep the FP32 router outside weight quantization.
                "gate": ("gate:!",),
                # Dense MLP path used by the first decoder layer.
                "": ("gate_proj:0", "up_proj:0", "down_proj:1"),
                "experts": {
                    "#": ("gate_proj:0", "up_proj:0", "down_proj:1"),
                },
                "shared_experts": ("gate_proj:0", "up_proj:0", "down_proj:1"),
            },
        },
    ]
