# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from .base_qwen3_vl import BaseQwen3VLGPTQ


class BaseQwen3_5VLGPTQ(BaseQwen3VLGPTQ):
    """
    Qwen3.5 VLM (ForConditionalGeneration) variant.

    Inherits vision/processor handling from Qwen3-VL and extends the
    module_tree with Qwen3.5-specific linear-attention modules.
    """

    # Qwen3.5 alternates linear-attention and full-attention by layer;
    # not every layer has every sub-module.
    layer_modules_strict = False

    ATTENTION_MASKS_REQUIRED_FOR_INPUT = True

    module_tree = [
        "model",
        "language_model",
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
                "in_proj_qkv:0",
                "in_proj_z:1",
                "in_proj_b:!",
                "in_proj_a:!",
                "out_proj:2",
            ),
            "post_attention_layernorm": ("post_attention_layernorm:!",),
            "mlp": ("gate_proj:0", "up_proj:0", "down_proj:1"),
        },
    ]
