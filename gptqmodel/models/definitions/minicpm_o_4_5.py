# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from __future__ import annotations

from typing import Any

import torch
from transformers import AutoConfig, AutoModel, Qwen3Config, Qwen3ForCausalLM

from .qwen3 import Qwen3QModel

_TTS_SAMPLING_DEFAULTS = {
    "top_p": 0.85,
    "top_k": 25,
    "repetition_penalty": 1.05,
    "temperature": 0.8,
}

_LLM_DTYPE = torch.bfloat16


def _ensure_tts_sampling_config(config: Any) -> None:
    """Fill official MiniCPM TTS defaults required by its remote constructor."""
    tts_config = getattr(config, "tts_config", None)
    if tts_config is None:
        return
    for name, value in _TTS_SAMPLING_DEFAULTS.items():
        if not hasattr(tts_config, name):
            setattr(tts_config, name, value)


def _qwen3_config_from_minicpm(config: Any) -> Qwen3Config:
    """Rebuild a pure Qwen3 config from MiniCPM-o-4.5's LLM config fields."""

    return Qwen3Config(
        vocab_size=config.vocab_size,
        hidden_size=config.hidden_size,
        intermediate_size=config.intermediate_size,
        num_hidden_layers=config.num_hidden_layers,
        num_attention_heads=config.num_attention_heads,
        num_key_value_heads=config.num_key_value_heads,
        head_dim=getattr(config, "head_dim", None),
        hidden_act=config.hidden_act,
        max_position_embeddings=config.max_position_embeddings,
        initializer_range=getattr(config, "initializer_range", 0.02),
        rms_norm_eps=config.rms_norm_eps,
        rope_theta=getattr(config, "rope_theta", 10000.0),
        rope_scaling=getattr(config, "rope_scaling", None),
        attention_bias=bool(getattr(config, "attention_bias", False)),
        attention_dropout=float(getattr(config, "attention_dropout", 0.0)),
        use_sliding_window=bool(
            getattr(
                config,
                "use_sliding_window",
                getattr(config, "sliding_window", None) is not None,
            )
        ),
        sliding_window=getattr(config, "sliding_window", None),
        max_window_layers=int(getattr(config, "max_window_layers", 28)),
        layer_types=getattr(config, "layer_types", None),
        tie_word_embeddings=bool(getattr(config, "tie_word_embeddings", False)),
        use_cache=bool(getattr(config, "use_cache", True)),
    )


class _MiniCPMO45Loader:
    """Load a MiniCPM-o-4.5 checkpoint and return its nested Qwen3 LLM.

    MiniCPM-o-4.5 wraps an unmodified ``Qwen3ForCausalLM`` as ``self.llm``.
    GPTQModel only needs that submodule, so this loader extracts it and rebinds
    its config to a pure ``Qwen3Config``.  The saved GPTQ artifact therefore
    remains a standalone Qwen3 checkpoint, matching the existing exporter.
    """

    @classmethod
    def _extract_llm(cls, host: Any) -> Qwen3ForCausalLM:
        llm = getattr(host, "llm", None)
        if llm is None:
            raise ValueError("MiniCPM-o-4.5 checkpoint does not expose a `host.llm` submodule")
        if not isinstance(llm, Qwen3ForCausalLM):
            raise TypeError(
                "MiniCPM-o-4.5 `host.llm` must be a Qwen3ForCausalLM instance, "
                f"got {type(llm).__name__}"
            )
        llm.config = _qwen3_config_from_minicpm(host.config)
        return llm

    @classmethod
    def _filter_load_kwargs(cls, kwargs: dict[str, Any]) -> dict[str, Any]:
        allowed = {
            "attn_implementation",
            "device_map",
            "low_cpu_mem_usage",
            "torch_dtype",
            "_fast_init",
        }
        return {key: value for key, value in kwargs.items() if key in allowed}

    @classmethod
    def from_pretrained(
        cls,
        model_id_or_path: str,
        config: Any = None,
        **kwargs: Any,
    ) -> Qwen3ForCausalLM:
        trust_remote_code = bool(kwargs.get("trust_remote_code", True))
        # During offload-to-disk turtle reloads, GPTQModel passes the extracted
        # Qwen3 LLM config back to this loader, but the checkpoint path still
        # points to the MiniCPM host. Reload the host config in that case so
        # weights are mapped through ``host.llm`` instead of trying to load a
        # standalone Qwen3 checkpoint from MiniCPM shards.
        if config is None or isinstance(config, Qwen3Config):
            config = AutoConfig.from_pretrained(
                model_id_or_path, trust_remote_code=trust_remote_code
            )
        _ensure_tts_sampling_config(config)
        load_kwargs = cls._filter_load_kwargs(kwargs)
        # MiniCPM-o-4.5 stores its nested Qwen3 backbone in bfloat16. Let
        # GPTQModel's dtype selection flow through instead of round-tripping
        # through float16; Qwen3 keeps non-quantized buffers such as rotary
        # inv_freq in their native float32 dtype.
        load_kwargs.setdefault("torch_dtype", _LLM_DTYPE)
        host = AutoModel.from_pretrained(
            model_id_or_path,
            config=config,
            trust_remote_code=trust_remote_code,
            **load_kwargs,
        )
        return cls._extract_llm(host)

    @classmethod
    def from_config(cls, config: Any, **kwargs: Any) -> Qwen3ForCausalLM:
        trust_remote_code = bool(kwargs.get("trust_remote_code", True))
        _ensure_tts_sampling_config(config)
        load_kwargs = cls._filter_load_kwargs(kwargs)
        load_kwargs.setdefault("torch_dtype", _LLM_DTYPE)
        host = AutoModel.from_config(
            config,
            trust_remote_code=trust_remote_code,
            **load_kwargs,
        )
        return cls._extract_llm(host)


class MiniCPMO45QModel(Qwen3QModel):
    """GPTQModel definition for MiniCPM-o-4.5's Qwen3 language backbone."""

    require_trust_remote_code = True
    require_dtype = torch.bfloat16
    require_fast_init = False
    loader = _MiniCPMO45Loader
