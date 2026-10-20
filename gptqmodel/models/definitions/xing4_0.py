# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from ..base import BaseQModel
from ..moe_lifecycle import GateUpDownMoELifecycleHooks


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
                "q_b_proj:0",
                "kv_a_proj_with_mqa:0",
                "kv_b_proj:0",
                "o_proj:1",
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
