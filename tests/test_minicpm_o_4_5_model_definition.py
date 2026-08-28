from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from transformers import AutoConfig, AutoModel, Qwen3Config, Qwen3ForCausalLM

from gptqmodel.models import auto
from gptqmodel.models.definitions.minicpm_o_4_5 import (
    MiniCPMO45QModel,
    _MiniCPMO45Loader,
    _qwen3_config_from_minicpm,
)
from gptqmodel.models.definitions.qwen3 import Qwen3QModel


def test_minicpmo45_model_type_selects_definition(monkeypatch):
    fake_config = SimpleNamespace(model_type="minicpmo")
    monkeypatch.setattr(
        auto.AutoConfig,
        "from_pretrained",
        lambda *args, **kwargs: fake_config,
    )

    assert auto.check_and_get_model_definition("/tmp/minicpmo45") is MiniCPMO45QModel
    assert "minicpmo" in auto.SUPPORTED_MODELS


def test_minicpmo45_definition_reuses_qwen3_module_tree():
    assert issubclass(MiniCPMO45QModel, Qwen3QModel)
    assert MiniCPMO45QModel.module_tree == Qwen3QModel.module_tree
    assert MiniCPMO45QModel.pre_lm_head_norm_module == "model.norm"
    assert MiniCPMO45QModel.require_trust_remote_code is True
    assert MiniCPMO45QModel.require_dtype is torch.bfloat16
    assert MiniCPMO45QModel.require_fast_init is False


def test_qwen3_config_conversion_preserves_llm_shape_fields():
    cfg = SimpleNamespace(
        vocab_size=151936,
        hidden_size=4096,
        intermediate_size=12288,
        num_hidden_layers=36,
        num_attention_heads=32,
        num_key_value_heads=8,
        head_dim=128,
        hidden_act="silu",
        max_position_embeddings=32768,
        initializer_range=0.02,
        rms_norm_eps=1e-06,
        rope_theta=1000000.0,
        rope_scaling=None,
        attention_bias=False,
        attention_dropout=0.0,
        use_sliding_window=True,
        sliding_window=512,
        max_window_layers=28,
        layer_types=None,
        tie_word_embeddings=False,
        use_cache=True,
    )

    qwen_cfg = _qwen3_config_from_minicpm(cfg)

    assert isinstance(qwen_cfg, Qwen3Config)
    assert qwen_cfg.model_type == "qwen3"
    assert qwen_cfg.vocab_size == 151936
    assert qwen_cfg.hidden_size == 4096
    assert qwen_cfg.num_hidden_layers == 36
    assert qwen_cfg.sliding_window == 512


def test_loader_extracts_qwen3_llm_from_minicpmo():
    llm = _make_qwen3_llm()
    llm.config = None
    host = SimpleNamespace(llm=llm, config=_minimal_minicpmo_config())

    extracted = _MiniCPMO45Loader._extract_llm(host)

    assert extracted is llm
    assert isinstance(extracted.config, Qwen3Config)
    assert extracted.config.model_type == "qwen3"


def test_loader_rejects_missing_llm():
    with pytest.raises(ValueError, match="does not expose a `host.llm`"):
        _MiniCPMO45Loader._extract_llm(
            SimpleNamespace(config=_minimal_minicpmo_config())
        )


def test_loader_rejects_non_qwen3_llm():
    with pytest.raises(TypeError, match="must be a Qwen3ForCausalLM instance"):
        _MiniCPMO45Loader._extract_llm(
            SimpleNamespace(llm=torch.nn.Module(), config=_minimal_minicpmo_config())
        )



def test_loader_from_pretrained_reloads_host_config_for_qwen3_config(monkeypatch):
    extracted_llm = _make_qwen3_llm()
    extracted_llm.config = None
    host = SimpleNamespace(llm=extracted_llm, config=_minimal_minicpmo_config())
    captured: dict[str, object] = {}

    def fake_auto_config_from_pretrained(path, **kwargs):
        captured["config_path"] = path
        captured["config_kwargs"] = kwargs
        return _minimal_minicpmo_config()

    def fake_model_from_pretrained(path, config, **kwargs):
        captured["model_path"] = path
        captured["model_config"] = config
        captured["model_kwargs"] = kwargs
        return host

    monkeypatch.setattr(
        AutoConfig,
        "from_pretrained",
        staticmethod(fake_auto_config_from_pretrained),
    )
    monkeypatch.setattr(
        AutoModel,
        "from_pretrained",
        staticmethod(fake_model_from_pretrained),
    )

    result = _MiniCPMO45Loader.from_pretrained(
        "/models/minicpm-o-4_5",
        config=Qwen3Config(vocab_size=256, hidden_size=64),
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )

    assert result is extracted_llm
    assert isinstance(result.config, Qwen3Config)
    assert captured["config_path"] == "/models/minicpm-o-4_5"
    assert captured["model_path"] == "/models/minicpm-o-4_5"
    assert captured["model_kwargs"]["torch_dtype"] == torch.bfloat16


def test_loader_from_config_extracts_qwen3_llm(monkeypatch):
    extracted_llm = _make_qwen3_llm()
    extracted_llm.config = None
    host = SimpleNamespace(llm=extracted_llm, config=_minimal_minicpmo_config())
    captured: dict[str, object] = {}

    def fake_from_config(config, **kwargs):
        captured["config"] = config
        captured["kwargs"] = kwargs
        return host

    monkeypatch.setattr(AutoModel, "from_config", staticmethod(fake_from_config))

    result = _MiniCPMO45Loader.from_config(
        _minimal_minicpmo_config(),
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )

    assert result is extracted_llm
    assert isinstance(result.config, Qwen3Config)
    assert result.config.model_type == "qwen3"
    assert captured["kwargs"]["torch_dtype"] == torch.bfloat16


def _minimal_minicpmo_config():
    cfg = SimpleNamespace(
        vocab_size=256,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        head_dim=16,
        hidden_act="silu",
        max_position_embeddings=128,
        initializer_range=0.02,
        rms_norm_eps=1e-05,
        rope_theta=10000.0,
        rope_scaling=None,
        attention_bias=False,
        attention_dropout=0.0,
        use_sliding_window=False,
        sliding_window=None,
        max_window_layers=2,
        layer_types=None,
        tie_word_embeddings=False,
        use_cache=True,
    )
    cfg.tts_config = SimpleNamespace()
    return cfg


def _make_qwen3_llm() -> Qwen3ForCausalLM:
    config = Qwen3Config(
        vocab_size=256,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=4,
        head_dim=16,
    )
    return Qwen3ForCausalLM(config)
