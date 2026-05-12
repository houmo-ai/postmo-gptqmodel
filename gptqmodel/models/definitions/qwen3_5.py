# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium



from . import LlamaQModel

try:
    from transformers.models.qwen3_5 import Qwen3_5TextConfig
except ImportError:
    Qwen3_5TextConfig = None


class Qwen3_5QModel(LlamaQModel):
    """
    Qwen3.5 inherits the Llama-style layout but introduces linear-attention
    layers and Q/K norm helper modules that should be tracked but not quantized.
    """
    
    config_class = Qwen3_5TextConfig

    # Qwen3.5 alternates linear-attention and full-attention by layer.
    layer_modules_strict = False

    # Shell-mode loading constructs from config; Qwen3.5 wrapper configs are
    # multimodal and not suitable for this path in text-only quantization.
    support_offload_to_disk = False

    pre_lm_head_norm_module = "model.norm"

    # AWQ scaling requires output projections to align with previous grouped modules.
    awq_scale_optimize_shape_dependent_modules = ["self_attn.o_proj", "linear_attn.out_proj"]

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

    _BRIDGE_ATTRS = (
        "bos_token_id",
        "eos_token_id",
        "pad_token_id",
        "vocab_size",
        "hidden_size",
        "intermediate_size",
        "num_hidden_layers",
        "num_attention_heads",
        "num_key_value_heads",
        "hidden_act",
        "max_position_embeddings",
        "initializer_range",
        "rms_norm_eps",
        "use_cache",
        "tie_word_embeddings",
        "rope_parameters",
        "attention_bias",
        "attention_dropout",
        "head_dim",
        "linear_conv_kernel_dim",
        "linear_key_head_dim",
        "linear_value_head_dim",
        "linear_num_key_heads",
        "linear_num_value_heads",
        "layer_types",
    )

    def __init__(self, *args, **kwargs):
        model = kwargs.get("model")
        if model is None and args:
            model = args[0]
        self._sync_text_config_attrs(model)
        super().__init__(*args, **kwargs)

    @classmethod
    def _sync_text_config_attrs(cls, model):
        if model is None:
            return
        cfg = getattr(model, "config", None)
        text_cfg = getattr(cfg, "text_config", None) if cfg is not None else None
        if cfg is None or text_cfg is None:
            return

        for key in cls._BRIDGE_ATTRS:
            if hasattr(text_cfg, key):
                value = getattr(text_cfg, key)
                if value is not None:
                    setattr(cfg, key, value)

    def before_model_load(self, load_quantized_model=False):
        try:
            import transformers.models.qwen3_5.configuration_qwen3_5 as qwen3_5_config
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "Qwen3.5 requires a transformers version with qwen3_5 support."
            ) from exc

        # Tokenizer and some loading paths resolve from root config fields; bridge
        # missing text attributes from text_config.
        if not getattr(qwen3_5_config, "_gptqmodel_qwen35_cfg_bridge", False):
            cfg_cls = qwen3_5_config.Qwen3_5Config

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

            qwen3_5_config._gptqmodel_qwen35_cfg_bridge = True
