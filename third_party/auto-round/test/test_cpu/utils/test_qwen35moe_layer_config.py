import json
import sys
from types import SimpleNamespace

import torch
import torch.nn as nn
from safetensors import safe_open
from safetensors.torch import save_file

import scripts_qwen35moe.quantize_moe as quantize_moe
from scripts_qwen35moe.quantize_moe import build_qwen35moe_layer_config, fix_checkpoint_after_save


class _SharedExpert(nn.Module):
    def __init__(self, hidden_size=8, intermediate_size=4):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)


class _Expert(nn.Module):
    def __init__(self, hidden_size=8, intermediate_size=4):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)


class _Layer(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = nn.Module()
        self.self_attn.q_proj = nn.Linear(8, 8, bias=False)
        self.self_attn.k_proj = nn.Linear(8, 8, bias=False)
        self.self_attn.v_proj = nn.Linear(8, 8, bias=False)
        self.self_attn.o_proj = nn.Linear(8, 8, bias=False)

        self.linear_attn = nn.Module()
        self.linear_attn.in_proj_qkv = nn.Linear(8, 8, bias=False)
        self.linear_attn.in_proj_z = nn.Linear(8, 8, bias=False)
        self.linear_attn.in_proj_b = nn.Linear(8, 8, bias=False)
        self.linear_attn.in_proj_a = nn.Linear(8, 8, bias=False)
        self.linear_attn.out_proj = nn.Linear(8, 8, bias=False)

        self.mlp = nn.Module()
        self.mlp.shared_expert = _SharedExpert()
        self.mlp.experts = nn.ModuleList([_Expert(), _Expert()])


class _PackedExperts(nn.Module):
    def __init__(self):
        super().__init__()
        self.act_fn = nn.SiLU()


class _PackedLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = nn.Module()
        self.self_attn.q_proj = nn.Linear(8, 8, bias=False)
        self.self_attn.k_proj = nn.Linear(8, 8, bias=False)
        self.self_attn.v_proj = nn.Linear(8, 8, bias=False)
        self.self_attn.o_proj = nn.Linear(8, 8, bias=False)

        self.linear_attn = nn.Module()
        self.linear_attn.in_proj_qkv = nn.Linear(8, 8, bias=False)
        self.linear_attn.in_proj_z = nn.Linear(8, 8, bias=False)
        self.linear_attn.in_proj_b = nn.Linear(8, 8, bias=False)
        self.linear_attn.in_proj_a = nn.Linear(8, 8, bias=False)
        self.linear_attn.out_proj = nn.Linear(8, 8, bias=False)

        self.mlp = nn.Module()
        self.mlp.shared_expert = _SharedExpert()
        self.mlp.experts = _PackedExperts()


class _LanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([_Layer()])


class _PackedLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([_PackedLayer()])


class _Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Module()
        self.model.language_model = _LanguageModel()


class _PackedModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Module()
        self.model.language_model = _PackedLanguageModel()


def _cfg(bits, group_size=64):
    return {"bits": bits, "group_size": group_size}


def test_returns_empty_when_no_llm_overrides_requested():
    model = _Model()

    layer_config = build_qwen35moe_layer_config(
        model,
        llm_group_size=64,
        attn_bits=None,
        shared_expert_bits=None,
        expert_bits=None,
        expert_up_gate_bits=None,
        expert_down_bits=None,
    )

    assert layer_config == {}


def test_attention_override_touches_only_attention_linears():
    model = _Model()

    layer_config = build_qwen35moe_layer_config(
        model,
        llm_group_size=64,
        attn_bits=5,
        shared_expert_bits=None,
        expert_bits=None,
        expert_up_gate_bits=None,
        expert_down_bits=None,
    )

    assert layer_config["model.language_model.layers.0.self_attn.q_proj"] == _cfg(5)
    assert layer_config["model.language_model.layers.0.self_attn.o_proj"] == _cfg(5)
    assert layer_config["model.language_model.layers.0.linear_attn.in_proj_qkv"] == _cfg(5)
    assert layer_config["model.language_model.layers.0.linear_attn.out_proj"] == _cfg(5)
    assert "model.language_model.layers.0.mlp.shared_expert.gate_proj" not in layer_config
    assert "model.language_model.layers.0.mlp.experts.0.down_proj" not in layer_config


def test_non_shared_expert_projection_overrides_take_precedence():
    model = _Model()

    layer_config = build_qwen35moe_layer_config(
        model,
        llm_group_size=128,
        attn_bits=5,
        shared_expert_bits=5,
        expert_bits=4,
        expert_up_gate_bits=4,
        expert_down_bits=5,
    )

    assert layer_config["model.language_model.layers.0.mlp.shared_expert.gate_proj"] == _cfg(5, 128)
    assert layer_config["model.language_model.layers.0.mlp.shared_expert.down_proj"] == _cfg(5, 128)
    assert layer_config["model.language_model.layers.0.mlp.experts.0.gate_proj"] == _cfg(4, 128)
    assert layer_config["model.language_model.layers.0.mlp.experts.0.up_proj"] == _cfg(4, 128)
    assert layer_config["model.language_model.layers.0.mlp.experts.0.down_proj"] == _cfg(5, 128)
    assert layer_config["model.language_model.layers.0.mlp.experts.1.down_proj"] == _cfg(5, 128)


def test_packed_experts_fall_back_to_regex_overrides():
    model = _PackedModel()

    layer_config = build_qwen35moe_layer_config(
        model,
        llm_group_size=128,
        attn_bits=None,
        shared_expert_bits=5,
        expert_bits=4,
        expert_up_gate_bits=4,
        expert_down_bits=5,
    )

    assert layer_config["model.language_model.layers.0.mlp.shared_expert.gate_proj"] == _cfg(5, 128)
    assert layer_config["model.language_model.layers.*.mlp.experts.*.(gate_proj|up_proj)"] == _cfg(4, 128)
    assert layer_config["model.language_model.layers.*.mlp.experts.*.down_proj"] == _cfg(5, 128)


def test_fix_checkpoint_after_save_normalizes_duplicate_language_model_prefixes(tmp_path):
    bad_key = "model.language_model.language_model.language_model.layers.0.linear_attn.out_proj.qweight"
    good_key = "model.language_model.layers.0.linear_attn.out_proj.qweight"
    shard_path = tmp_path / "model-00001-of-00001.safetensors"
    extra_path = tmp_path / "model_extra_tensors.safetensors"
    index_path = tmp_path / "model.safetensors.index.json"

    save_file({bad_key: torch.ones(1)}, str(shard_path))
    save_file({"model.language_model.embed_tokens.weight": torch.zeros(1)}, str(extra_path))
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump({"weight_map": {bad_key: shard_path.name}}, f)

    fix_checkpoint_after_save(str(tmp_path))

    with safe_open(str(shard_path), framework="pt", device="cpu") as tensors:
        keys = set(tensors.keys())
    assert good_key in keys
    assert bad_key not in keys

    with open(index_path, "r", encoding="utf-8") as f:
        weight_map = json.load(f)["weight_map"]
    assert weight_map[good_key] == shard_path.name
    assert bad_key not in weight_map


def test_main_runs_checkpoint_fix_for_llm_only_exports(monkeypatch, tmp_path):
    output_dir = tmp_path / "out"
    calls = []

    args = SimpleNamespace(
        model="dummy-model",
        output_dir=str(output_dir),
        mode="llm-only",
        device="cuda:0",
        llm_bits=4,
        llm_group_size=64,
        attn_bits=None,
        shared_expert_bits=None,
        expert_bits=None,
        expert_up_gate_bits=None,
        expert_down_bits=None,
        vision_bits=8,
        vision_group_size=64,
        sym=True,
        iters=0,
        nsamples=1,
        seqlen=16,
        batch_size=1,
        dataset="dummy",
        seed=42,
        format="auto_round:gptqmodel",
        save_rotated=None,
        deterministic=False,
        two_stage=False,
    )

    class _FakeAutoTokenizer:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            return cls()

    class _FakeAutoProcessor:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            return cls()

    class _FakeModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace()

    class _FakeAutoModelForImageTextToText:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            return _FakeModel()

    class _FakeAutoRound:
        def __init__(self, *args, **kwargs):
            pass

        def quantize(self):
            return None

        def save_quantized(self, *args, **kwargs):
            return None

    fake_transformers = SimpleNamespace(
        AutoTokenizer=_FakeAutoTokenizer,
        AutoProcessor=_FakeAutoProcessor,
        AutoModelForImageTextToText=_FakeAutoModelForImageTextToText,
        AutoModel=_FakeAutoModelForImageTextToText,
    )
    fake_auto_round = SimpleNamespace(AutoRound=_FakeAutoRound)

    monkeypatch.setattr(quantize_moe, "parse_args", lambda: args)
    monkeypatch.setattr(quantize_moe, "validate_export_requirements", lambda fmt: None)
    monkeypatch.setattr(quantize_moe, "build_qwen35moe_layer_config", lambda *args, **kwargs: {})
    monkeypatch.setattr(quantize_moe, "_print_llm_override_summary", lambda *args, **kwargs: None)
    monkeypatch.setattr(quantize_moe, "fix_checkpoint_after_save", lambda path: calls.append(path))
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    monkeypatch.setitem(sys.modules, "auto_round", fake_auto_round)

    quantize_moe.main()

    assert calls == [str(output_dir)]
