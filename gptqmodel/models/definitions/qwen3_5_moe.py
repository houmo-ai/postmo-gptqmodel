# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium
from transformers.models.auto import AutoModelForCausalLM, AutoModelForImageTextToText

from gptqmodel.models.moe_lifecycle import GateUpDownMoELifecycleHooks

from ..base import BaseQModel
from .base_qwen3_vl import BaseQwen3VLGPTQ


class _BaseQwen3_5MoeQModel(BaseQModel):
    layer_modules_strict = False

    require_monkeypatch = False

    dynamic_expert_index = "num_experts"

    support_offload_to_disk = False

    awq_scale_optimize_shape_dependent_modules = ["self_attn.o_proj", "linear_attn.out_proj"]
    # awq scaling optimizations requires some modules within same subset to strictly match the shape of previous module
    # the o_proj must match v_proj or else scaling optimizations are skipped (GQA vs MHA)

    # MoE lifecycle hooks for gate_proj/up_proj/down_proj pattern
    moe_lifecycle_hooks = GateUpDownMoELifecycleHooks()

    _BRIDGE_ATTRS = ("bos_token_id", "eos_token_id", "pad_token_id", "vocab_size")

    def after_model_load(self, model, load_quantized_model):
        from .qwen3_5_kernels import guard_qwen35_cpu_kernels

        return guard_qwen35_cpu_kernels(super().after_model_load(model, load_quantized_model))

    def before_model_load(self, load_quantized_model=False):
        import transformers.models.qwen3_5_moe.configuration_qwen3_5_moe as qwen3_5_moe_config

        # Tokenizer loads config directly from path and expects text token ids at root level.
        if not getattr(qwen3_5_moe_config, "_gptqmodel_qwen35_cfg_bridge", False):
            cfg_cls = qwen3_5_moe_config.Qwen3_5MoeConfig

            def _bridge_property(name):
                def _get(instance):
                    value = instance.__dict__.get(name, None)
                    if value is not None:
                        return value
                    text_cfg = getattr(instance, "text_config", None)
                    return getattr(text_cfg, name, None) if text_cfg is not None else None

                def _set(instance, value):
                    instance.__dict__[name] = value
                    text_cfg = getattr(instance, "text_config", None)
                    if text_cfg is not None and hasattr(text_cfg, name):
                        setattr(text_cfg, name, value)

                return property(_get, _set)

            for attr in self._BRIDGE_ATTRS:
                setattr(cfg_cls, attr, _bridge_property(attr))

            qwen3_5_moe_config._gptqmodel_qwen35_cfg_bridge = True


class Qwen3_5_MoeQModel(_BaseQwen3_5MoeQModel):
    """Full Qwen3.5-MoE VLM (`model_type=qwen3_5_moe`), including vision."""

    loader = AutoModelForImageTextToText

    require_load_processor = True

    ATTENTION_MASKS_REQUIRED_FOR_INPUT = True

    pre_lm_head_norm_module = "model.language_model.norm"

    module_tree = [
        "model",
        "language_model",
        "layers",
        "#",
        {
            "input_layernorm": ("input_layernorm:!",),
            "self_attn": ("norm:!", "q_proj:0", "k_proj:0", "v_proj:0", "o_proj:1"),
            "linear_attn": (
                "norm:!",
                "in_proj_qkv:0",
                "in_proj_z:1",
                "out_proj:2",
            ),
            "post_attention_layernorm": ("post_attention_layernorm:!",),
            "mlp:moe:?": {
                "gate": ("gate:!",),
                "shared_expert_gate": ("shared_expert_gate:!",),
                "shared_expert:0": ("gate_proj:0", "up_proj:0", "down_proj:1"),
                "experts": {
                    "#": ("gate_proj:0", "up_proj:0", "down_proj:1"),
                },
            },
        },
    ]

    _language_model = BaseQwen3VLGPTQ._language_model
    _visual_model = BaseQwen3VLGPTQ._visual_model
    pre_quantize_generate_hook_start = BaseQwen3VLGPTQ.pre_quantize_generate_hook_start
    pre_quantize_generate_hook_end = BaseQwen3VLGPTQ.pre_quantize_generate_hook_end
    process_vision_info = staticmethod(BaseQwen3VLGPTQ.process_vision_info)
    preprocess_dataset = BaseQwen3VLGPTQ.preprocess_dataset
    load_processor = BaseQwen3VLGPTQ.load_processor
    prepare_dataset = BaseQwen3VLGPTQ.prepare_dataset


class Qwen3_5_MoeTextQModel(_BaseQwen3_5MoeQModel):
    """Text-only Qwen3.5-MoE (`model_type=qwen3_5_moe_text`), no vision tower."""

    try:
        from transformers.models.qwen3_5_moe import Qwen3_5MoeTextConfig
    except ImportError:
        Qwen3_5MoeTextConfig = None

    loader = AutoModelForCausalLM

    config_class = Qwen3_5MoeTextConfig

    pre_lm_head_norm_module = "model.norm"

    module_tree = [
        "model",
        "layers",
        "#",
        {
            "input_layernorm": ("input_layernorm:!",),
            "self_attn": ("norm:!", "q_proj:0", "k_proj:0", "v_proj:0", "o_proj:1"),
            "linear_attn": (
                "norm:!",
                "in_proj_qkv:0",
                "in_proj_z:1",
                "out_proj:2",
            ),
            "post_attention_layernorm": ("post_attention_layernorm:!",),
            "mlp:moe:?": {
                "gate": ("gate:!",),
                "shared_expert_gate": ("shared_expert_gate:!",),
                "shared_expert:0": ("gate_proj:0", "up_proj:0", "down_proj:1"),
                "experts": {
                    "#": ("gate_proj:0", "up_proj:0", "down_proj:1"),
                },
            },
        },
    ]

    #     if load_quantized_model:
    #         import transformers.models.qwen3_5_moe.modeling_qwen3_5_moe as qwen3_5_moe_modeling

    #         from ...nn_modules.converter import Qwen3_5MoeExpertsNew

    #         qwen3_5_moe_modeling.Qwen3_5MoeExperts = Qwen3_5MoeExpertsNew

    #         if not getattr(qwen3_5_moe_modeling, "_gptqmodel_qwen35_skip_expert_init", False):
    #             original_init_weights = qwen3_5_moe_modeling.Qwen3_5MoePreTrainedModel._init_weights

    #             def _patched_init_weights(model_self, module):
    #                 if isinstance(module, Qwen3_5MoeExpertsNew):
    #                     return
    #                 return original_init_weights(model_self, module)

    #             qwen3_5_moe_modeling.Qwen3_5MoePreTrainedModel._init_weights = _patched_init_weights
    #             qwen3_5_moe_modeling._gptqmodel_qwen35_skip_expert_init = True


Qwen3_5MoeQModel = Qwen3_5_MoeQModel
Qwen3_5MoeTextQModel = Qwen3_5_MoeTextQModel
