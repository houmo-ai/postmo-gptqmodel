# Copyright 2026 HOUMO AI
# SPDX-License-Identifier: Apache-2.0
"""GLM-5.3-Flash: native multimodal KDA / DSA / mHC with clamped experts."""

import torch
from transformers import AutoModelForImageTextToText

from ..base import BaseQModel
from ..moe_lifecycle import GateUpDownMoELifecycleHooks


def register_glm53_defuser():
    """Enable Defuser's existing, exact ``_apply_gate`` expert implementation.

    Keep this registration model-local. The native HF class and checkpoint
    conversions remain unchanged, including vision and fused convolution.
    """
    from defuser.model_registry import MODEL_CONFIG

    for model_type in ("glm5_next", "glm5_next_text"):
        MODEL_CONFIG.setdefault(model_type, {"min_transformers_version": "5.16.0"})


class Glm53MoELifecycleHooks(GateUpDownMoELifecycleHooks):
    def expert_intermediate(self, moe_block, expert, gate_out, up_out):
        # The fused HF owner defines the exact gate/up clipping, which Defuser
        # also calls during ordinary routed inference. Bypass must match it.
        return moe_block.experts._apply_gate(torch.cat((gate_out, up_out), dim=-1))


class Glm5NextQModel(BaseQModel):
    loader = AutoModelForImageTextToText
    dynamic_expert_index = "n_routed_experts"
    layer_modules_strict = False
    support_offload_to_disk = False
    ATTENTION_MASKS_REQUIRED_FOR_INPUT = True
    replay_autocast_dtype = torch.bfloat16
    pre_lm_head_norm_module = "model.language_model.norm"
    moe_lifecycle_hooks = Glm53MoELifecycleHooks()

    module_tree = [
        "model",
        "language_model",
        "layers",
        "#",
        {
            "self_attn": (
                "q_proj:0",
                "k_proj:0",
                "v_proj:0",
                "g_a_proj:0",
                "q_a_proj:0",
                "kv_a_proj_with_mqa:0",
                "q_b_proj:1",
                "g_b_proj:1",
                "o_proj:2",
                # K/V factors are split/transposed by static DSA. Preserve the
                # floating factors here; the deployment quantizer owns W8.
                "kv_b_proj:!",
                "b_proj:!",
                "forget_gate:!",
                "conv1d:!",
            ),
            "mlp:0": ("gate_proj:0", "up_proj:0", "down_proj:1"),
            "mlp:moe:?": {
                "gate": ("gate:!",),
                "shared_experts": ("gate_proj:0", "up_proj:0", "down_proj:1"),
                "experts": {"#": ("gate_proj:0", "up_proj:0", "down_proj:1")},
            },
        },
    ]

    def before_model_load(self, load_quantized_model=False):
        register_glm53_defuser()

    def pre_quantize_generate_hook_start(self):
        # Required-mask calibration batches are placed on the quantization
        # device by StageInputsCapture, including text-only VLM batches.
        self._glm53_capture_devices = []
        for module in (self.model.get_input_embeddings(), self.model.model.visual):
            self._glm53_capture_devices.append(
                (module, next(module.parameters()).device)
            )
            module.to(self.quantize_config.device)

    def pre_quantize_generate_hook_end(self):
        for module, device in getattr(self, "_glm53_capture_devices", ()):
            module.to(device)
        self._glm53_capture_devices = []

    def after_model_load(self, model, load_quantized_model):
        model = super().after_model_load(model, load_quantized_model)
        text = model.config.get_text_config()
        if any(kind != "full" for kind in text.indexer_types):
            raise ValueError(
                "GLM-5.3 quantization requires independent full DSA indexers"
            )
        if self.quantize_config and self.quantize_config.lm_head:
            raise ValueError(
                "GLM-5.3 lm_head is quantized to W8 by the deployment exporter"
            )
        return model


__all__ = ["Glm5NextQModel", "Glm53MoELifecycleHooks", "register_glm53_defuser"]
