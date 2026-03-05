# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from ..base import BaseQModel


class Qwen3_5MoeQModel(BaseQModel):
    # Qwen3.5-MoE alternates linear attention and full attention by layer.
    layer_modules_strict = False

    # Expand expert module templates using config.text_config.num_experts.
    dynamic_expert_index = "num_experts"

    # AutoModelForCausalLM.from_config does not accept the multimodal wrapper config
    # in shell-mode loading; use direct from_pretrained path for this model.
    support_offload_to_disk = False

    pre_lm_head_norm_module = "model.norm"

    # AWQ scaling requires output projections to align with preceding grouped modules.
    awq_scale_optimize_shape_dependent_modules = ["self_attn.o_proj", "linear_attn.out_proj"]

    module_tree = [
        "model",
        "layers",
        "#",
        {
            "input_layernorm": ("input_layernorm:!",),
            "self_attn": ("q_proj:0", "k_proj:0", "v_proj:0", "o_proj:1"),
            "linear_attn": ("in_proj_qkv:0", "in_proj_z:0", "in_proj_b:!", "in_proj_a:!", "out_proj:1"),
            "post_attention_layernorm": ("post_attention_layernorm:!",),
            "mlp": {
                "gate": ("gate:!",),
                "shared_expert_gate": ("shared_expert_gate:!",),
                "shared_expert": ("gate_proj:0", "up_proj:0", "down_proj:1"),
                "experts": (
                    "gate_proj.{expert_index}:0",
                    "up_proj.{expert_index}:0",
                    "down_proj.{expert_index}:1",
                ),
            },
        },
    ]

    def __init__(self, *args, **kwargs):
        model = kwargs.get("model")
        if model is None and args:
            model = args[0]
        self._sync_text_config_attrs(model)
        super().__init__(*args, **kwargs)

    @staticmethod
    def _sync_text_config_attrs(model):
        if model is None:
            return
        cfg = getattr(model, "config", None)
        text_cfg = getattr(cfg, "text_config", None) if cfg is not None else None
        if cfg is None or text_cfg is None:
            return

        for key in ("bos_token_id", "eos_token_id", "pad_token_id", "vocab_size"):
            if hasattr(text_cfg, key):
                value = getattr(text_cfg, key)
                if value is not None:
                    setattr(cfg, key, value)

    def before_model_load(self, load_quantized_model=False):
        import transformers.models.qwen3_5_moe.configuration_qwen3_5_moe as qwen3_5_moe_config

        # Tokenicer loads config directly from path and expects text-token ids at root level.
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

            for attr in ("bos_token_id", "eos_token_id", "pad_token_id", "vocab_size"):
                setattr(cfg_cls, attr, _bridge_property(attr))

            qwen3_5_moe_config._gptqmodel_qwen35_cfg_bridge = True

        # from_quantized() builds model from config and must instantiate the converted expert layout.
        if load_quantized_model:
            import transformers.models.qwen3_5_moe.modeling_qwen3_5_moe as qwen3_5_moe_modeling

            from ...nn_modules.converter import Qwen3_5MoeExpertsNew

            qwen3_5_moe_modeling.Qwen3_5MoeExperts = Qwen3_5MoeExpertsNew

            # In quantized loading path we construct from config and then load packed weights;
            # expert tensors do not need random initialization.
            if not getattr(qwen3_5_moe_modeling, "_gptqmodel_qwen35_skip_expert_init", False):
                original_init_weights = qwen3_5_moe_modeling.Qwen3_5MoePreTrainedModel._init_weights

                def _patched_init_weights(model_self, module):
                    if isinstance(module, Qwen3_5MoeExpertsNew):
                        return
                    return original_init_weights(model_self, module)

                qwen3_5_moe_modeling.Qwen3_5MoePreTrainedModel._init_weights = _patched_init_weights
                qwen3_5_moe_modeling._gptqmodel_qwen35_skip_expert_init = True
