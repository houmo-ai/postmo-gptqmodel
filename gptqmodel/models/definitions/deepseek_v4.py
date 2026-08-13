# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

import torch

from .deepseek_v3 import DeepSeekV3QModel


class DeepSeekV4QModel(DeepSeekV3QModel):
    """GPTQModel definition for the Transformers DeepSeek-V4 implementation."""

    dynamic_expert_index = "n_routed_experts"
    rotary_embedding = "model.rotary_emb"
    layer_modules_strict = False
    # Transformers keeps mHC and RMSNorm parameters in fp32. Their outputs feed
    # bf16 projections, so direct decoder-layer replay must retain inference's
    # autocast behavior instead of coercing the numerically sensitive modules.
    replay_autocast_dtype = torch.bfloat16
    module_tree = [
        "model",
        "layers",
        "#",
        {
            "input_layernorm": ("input_layernorm:!",),
            "self_attn": (
                "q_a_norm:!",
                "q_a_proj:0",
                "q_b_norm:!",
                "q_b_proj:0",
                "o_a_proj:!",
                "o_b_proj:1",
                "kv_norm:!",
                "kv_proj:2",
            ),
            "post_attention_layernorm": ("post_attention_layernorm:!",),
            "mlp:moe": {
                "gate": ("gate:!",),
                "experts": {
                    "#": ("gate_proj:0", "up_proj:0", "down_proj:1"),
                },
                "shared_experts": ("gate_proj:0", "up_proj:0", "down_proj:1"),
            },
        },
    ]

    def _resolve_turtle_reload_threshold(self) -> int:
        """Keep one defused source model instead of reloading a huge checkpoint.

        Materialization copies tensors into the shell and does not consume the
        source tensor. Reloading after the generic 512 MiB threshold would read
        the complete 43-layer BF16 staging checkpoint once per decoder layer and
        can transiently hold two copies in host memory.
        """

        return 0


__all__ = ["DeepSeekV4QModel"]
