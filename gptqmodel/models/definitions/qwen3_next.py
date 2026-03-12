# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from transformers.models.qwen3_next.configuration_qwen3_next import Qwen3NextConfig

from ...quantization import METHOD
from ..base import BaseQModel
from ..moe_lifecycle import GateUpDownMoELifecycleHooks


def _patch_defuser_qwen3_next_registry() -> None:
    import defuser.model_registry as defuser_model_registry

    if getattr(defuser_model_registry, "_gptqmodel_qwen3_next_registry_patch", False):
        return

    registry_entry = defuser_model_registry.MODEL_CONFIG.setdefault("qwen3_next", {})
    registry_entry.setdefault("min_transformers_version", "5.2.0")
    defuser_model_registry._gptqmodel_qwen3_next_registry_patch = True


class Qwen3NextGPTQ(BaseQModel):
    """
    GPTQ config for Qwen3-Next (HF: Qwen3Next*).

    Qwen3-Next mixes:
      - full-attention layers (`self_attn`)
      - linear-attention layers (`linear_attn`)
      - dense MLP blocks
      - sparse MoE MLP blocks

    The module tree keeps these execution paths separate so AWQ/GPTQ can skip the
    inactive branch on each decoder layer instead of dropping the whole subset.
    """

    config_class = Qwen3NextConfig

    layer_modules_strict = False

    support_offload_to_disk = False

    pre_lm_head_norm_module = "model.norm"

    dynamic_expert_index = "num_experts"

    awq_scale_optimize_shape_dependent_modules = ["self_attn.o_proj", "linear_attn.out_proj"]

    moe_lifecycle_hooks = GateUpDownMoELifecycleHooks()

    module_tree = [
        "model",
        "layers",
        "#",
        {
            "input_layernorm": ("input_layernorm:!",),
            "self_attn": (
                "q_norm:!",
                "k_norm:!",
                "q_proj:0",
                "k_proj:0",
                "v_proj:0",
                "o_proj:1",
            ),
            "linear_attn": (
                "norm:!",
                "in_proj_qkvz:0",
                "in_proj_ba:!",
                "out_proj:1",
            ),
            "post_attention_layernorm": ("post_attention_layernorm:!",),
            "mlp": ("gate_proj:0", "up_proj:0", "down_proj:1"),
            "mlp:moe:?": {
                "gate": ("gate:!",),
                "shared_expert_gate": ("shared_expert_gate:!",),
                "shared_expert": ("gate_proj:0", "up_proj:0", "down_proj:1"),
                "experts": {
                    "#": ("gate_proj:0", "up_proj:0", "down_proj:1"),
                },
            },
        },
    ]

    module_tree_overrides = {
        METHOD.AWQ: [
            {
                "mlp:moe:?": {
                    "gate": ("gate",),
                }
            }
        ]
    }

    def before_model_load(self, load_quantized_model=False):
        # defuser already knows how to unfuse Qwen3-Next experts once the model
        # type is registered. Patch the in-process registry instead of modifying
        # the third-party package on disk.
        _patch_defuser_qwen3_next_registry()


Qwen3NextQModel = Qwen3NextGPTQ
