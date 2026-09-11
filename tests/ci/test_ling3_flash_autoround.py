from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from gptqmodel.recipes.ling3_flash_autoround import (
    _calibration_loader,
    _install_autoround_nonfinite_loss_guard,
    build_autoround_block_names,
    build_autoround_layer_config,
    canonicalize_autoround_gptqmodel_config,
)
from gptqmodel.recipes.ling3_flash import (
    autoround_calibration_source,
    quantize_ling3_flash,
)


@pytest.mark.parametrize("dtype", ["int_sym_xh", "int_sym_gptq", "int_sym"])
def test_public_autoround_recipe_defaults_and_overrides(tmp_path, dtype):
    result = quantize_ling3_flash(
        model_dir=str(tmp_path / "source"),
        output_dir=str(tmp_path / "out"),
        method="auto-round",
        data_type=dtype,
        dry_run=True,
    )
    assert result.method == "autoround"
    assert result.calibration["nsamples"] == 256
    assert result.provenance["iters"] == 400
    assert result.provenance["data_type"] == dtype
    assert result.provenance["attention_backend"] == "eager"
    assert result.provenance["mla_weight_layout"] == "hf_unabsorbed"
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize(
    "override",
    [
        {"max_quant_layers": 1},
        {"offload_to_disk": True},
        {"moe_routing_batch_size": 2},
        {"hessian_mse": False},
    ],
)
def test_autoround_rejects_unused_gptq_options(tmp_path, override):
    with pytest.raises(ValueError, match="GPTQ-only"):
        quantize_ling3_flash(
            model_dir="source",
            output_dir=str(tmp_path / "out"),
            method="autoround",
            dry_run=True,
            **override,
        )


def test_calibration_keeps_full_corpus_and_honors_text_key(tmp_path):
    path = tmp_path / "data.jsonl"
    rows = [{"content": f"example {i}", "ignored": "wrong"} for i in range(6)]
    path.write_text("\n".join(map(json.dumps, rows)))
    source = autoround_calibration_source(
        calibration_jsonl=path, calibration_text_key="content"
    )
    assert list(source["text"]) == [row["content"] for row in rows]
    assert (
        len(
            autoround_calibration_source(
                calibration_data=rows, calibration_text_key="content"
            )
        )
        == 6
    )
    assert autoround_calibration_source(calibration_jsonl=path) == str(path.resolve())


def test_autoround_honors_remote_code_opt_out(tmp_path):
    with pytest.raises(ValueError, match="requires trust_remote_code=True"):
        quantize_ling3_flash(
            model_dir=str(tmp_path / "source"),
            output_dir=str(tmp_path / "out"),
            method="autoround",
            trust_remote_code=False,
            dry_run=True,
        )


def test_calibration_loader_filters_before_selecting_and_uses_hf_shuffle():
    from datasets import Dataset
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import PreTrainedTokenizerFast

    raw = Tokenizer(WordLevel(vocab={str(i): i for i in range(32)}, unk_token="0"))
    raw.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=raw, unk_token="0", pad_token="0"
    )
    valid = [" ".join(str(i + j) for j in range(8)) for i in range(1, 9)]
    source = Dataset.from_dict({"text": ["1", "2 2 2 2 2 2 2 2", *valid]})
    expected = [row["text"] for row in source.shuffle(seed=42) if row["text"] in valid][
        :4
    ]
    loader = _calibration_loader(
        source, tokenizer, seqlen=8, nsamples=4, batch_size=2, seed=42
    )
    batches = list(loader)
    actual = torch.cat([batch["input_ids"] for batch in batches])
    expected_ids = tokenizer(expected, truncation=True, max_length=8)["input_ids"]
    assert torch.equal(actual, torch.tensor(expected_ids))
    assert [tuple(batch["input_ids"].shape) for batch in batches] == [(2, 8), (2, 8)]
    with pytest.raises(ValueError, match="full-length calibration rows"):
        _calibration_loader(
            source, tokenizer, seqlen=8, nsamples=9, batch_size=2, seed=42
        )


def _save_real_tiny_ling(path, mtp_layers, **overrides):
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import AutoModelForCausalLM, PreTrainedTokenizerFast
    from gptqmodel.models.definitions.ling_3_flash import (
        load_ling_3_flash_config,
        patch_ling_3_flash_remote_code,
    )

    from _ling_fixture import ling_code_dir

    remote = ling_code_dir()
    # Only the remote config/code is loaded, never the full checkpoint weights.
    config = load_ling_3_flash_config(str(remote))
    values = dict(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=32,
        layer_group_size=2,
        kv_lora_rank=64,
        q_lora_rank=None,
        qk_nope_head_dim=32,
        qk_rope_head_dim=16,
        qk_head_dim=48,
        v_head_dim=32,
        num_experts=2,
        num_experts_per_tok=1,
        n_group=1,
        topk_group=1,
        moe_intermediate_size=128,
        moe_shared_expert_intermediate_size=128,
        first_k_dense_replace=1,
        num_nextn_predict_layers=mtp_layers,
        max_position_embeddings=128,
        pad_token_id=0,
        eos_token_id=2,
        expert_swiglu_limit_list=[0] * 3,
        share_expert_swiglu_limit_list=[0] * 3,
        use_cache=False,
    )
    values.update(overrides)
    for name, value in values.items():
        setattr(config, name, value)
    torch.manual_seed(42)
    model = AutoModelForCausalLM.from_config(
        config, trust_remote_code=True, attn_implementation="eager"
    )
    patch_ling_3_flash_remote_code()
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if name.endswith("dt_bias"):
                parameter.zero_()
    model.to(torch.bfloat16).save_pretrained(path)
    tokenizer = Tokenizer(
        WordLevel(vocab={str(i): i for i in range(128)}, unk_token="3")
    )
    tokenizer.pre_tokenizer = Whitespace()
    PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        pad_token="0",
        bos_token="1",
        eos_token="2",
        unk_token="3",
    ).save_pretrained(path)


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="Real KDA uses CUDA")
@pytest.mark.parametrize("mtp_layers", [0, 1])
def test_real_ling_autoround_quantize_save_load(tmp_path, mtp_layers, monkeypatch):
    from gptqmodel import BACKEND, GPTQModel
    from safetensors.torch import load_file
    from transformers import modeling_utils

    for name in ("normal_", "uniform_", "kaiming_uniform_"):
        monkeypatch.setattr(torch.nn.init, name, getattr(torch.nn.init, name))
    monkeypatch.setattr(
        modeling_utils,
        "_init_weights",
        getattr(modeling_utils, "_init_weights", True),
        raising=False,
    )
    source, out = tmp_path / "source", tmp_path / "out"
    _save_real_tiny_ling(source, mtp_layers)
    calibration = tmp_path / "calibration.jsonl"
    calibration.write_text(
        "\n".join(
            json.dumps({"text": " ".join(str((i + j) % 120 + 4) for i in range(64))})
            for j in range(4)
        )
        + "\n"
    )
    # Keep source-model construction separate from the quantizer's global HF
    # and kernel state, as in a real CLI run against an existing checkpoint.
    script = (
        Path(__file__).resolve().parents[2]
        / "examples/quantization/examples/example_ling3_flash.py"
    )
    process = subprocess.run(
        [
            sys.executable,
            str(script),
            "--model",
            str(source),
            "--out",
            str(out),
            "--method",
            "autoround",
            "--calibration-jsonl",
            str(calibration),
            "--nsamples",
            "2",
            "--seqlen",
            "32",
            "--iters",
            "2",
            "--device-map",
            "0",
        ],
        capture_output=True,
        text=True,
    )
    assert process.returncode == 0, process.stdout + process.stderr
    provenance = json.loads(
        (out / "ling3_flash_quant_recipe_provenance.json").read_text()
    )
    assert provenance["provenance"]["data_type"] == "int_sym_xh"
    meta = json.loads((out / "config.json").read_text())["quantization_config"]
    assert meta["data_type"] == "int_sym_xh"
    assert "extra_config" not in meta
    model = GPTQModel.load(
        str(out),
        backend=BACKEND.TORCH,
        device="cpu",
        dtype=torch.float16,
        trust_remote_code=True,
    )
    quantized = {
        name: module
        for name, module in model.model.named_modules()
        if hasattr(module, "qweight")
    }
    assert quantized["model.layers.1.attention.q_proj"].bits == 8
    assert quantized["model.layers.1.attention.kv_b_proj"].bits == 8
    assert quantized["model.layers.1.attention.dense"].bits == 8
    assert quantized["model.layers.1.mlp.experts.0.gate_proj"].bits == 4
    assert "model.layers.0.attention.f_proj" not in quantized
    assert all(
        torch.isfinite(module.scales).all() and (module.scales >= 0).all()
        for module in quantized.values()
    )
    if mtp_layers:
        assert not any(name.startswith("model.layers.2.") for name in quantized)
        expected = load_file(source / "model.safetensors")
        actual = load_file(out / "model.safetensors")
        mtp_names = [name for name in expected if name.startswith("model.layers.2.")]
        assert mtp_names
        for name in mtp_names:
            assert torch.equal(actual[name], expected[name])
    resaved = tmp_path / "resaved"
    model.save(str(resaved))
    restored = GPTQModel.load(
        str(resaved),
        backend=BACKEND.TORCH,
        device="cpu",
        dtype=torch.float16,
        trust_remote_code=True,
    )
    restored_quantized = {
        name: module
        for name, module in restored.model.named_modules()
        if hasattr(module, "qweight")
    }
    assert restored_quantized.keys() == quantized.keys()
    assert (
        json.loads((resaved / "config.json").read_text())["quantization_config"][
            "data_type"
        ]
        == "int_sym_xh"
    )
    for name, module in quantized.items():
        for buffer in ("qweight", "qzeros", "scales", "g_idx"):
            assert torch.equal(
                getattr(module, buffer), getattr(restored_quantized[name], buffer)
            ), (name, buffer)


class _Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(64, 64, bias=False)
        self.f_proj = nn.Linear(64, 64, bias=False)
        self.b_proj = nn.Linear(64, 4, bias=False)


class _Expert(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_proj = nn.Linear(64, 64, bias=False)
        self.up_proj = nn.Linear(64, 64, bias=False)
        self.down_proj = nn.Linear(64, 64, bias=False)


class _MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.experts = nn.ModuleList([_Expert(), _Expert()])
        self.shared_experts = _Expert()


class _Layer(nn.Module):
    def __init__(self):
        super().__init__()
        self.attention = _Attention()
        self.mlp = _MLP()


class _TinyLing(nn.Module):
    def __init__(self, *, include_mtp: bool = False):
        super().__init__()
        self.config = SimpleNamespace(num_hidden_layers=1)
        self.model = nn.Module()
        layers = [_Layer()]
        if include_mtp:
            layers.append(_Layer())
        self.model.layers = nn.ModuleList(layers)


def _resolve_autoround_layer_config(layer_config, module_name):
    from auto_round.utils import to_standard_regex

    for pattern, config in layer_config.items():
        if re.search(to_standard_regex(pattern), module_name):
            return config
    return None


def test_autoround_profile_excludes_kda_and_only_routes_experts_to_w4():
    layer_config = build_autoround_layer_config(_TinyLing())

    assert len(layer_config) == 2
    assert _resolve_autoround_layer_config(
        layer_config, "model.layers.0.attention.f_proj"
    ) == {
        "bits": 16,
        "data_type": "fp",
    }
    assert _resolve_autoround_layer_config(
        layer_config, "model.layers.0.attention.b_proj"
    ) == {
        "bits": 16,
        "data_type": "fp",
    }
    assert _resolve_autoround_layer_config(
        layer_config, "model.layers.0.mlp.experts.0.gate_proj"
    ) == {
        "bits": 4,
        "group_size": 64,
    }
    assert (
        _resolve_autoround_layer_config(layer_config, "model.layers.0.attention.q_proj")
        is None
    )
    assert (
        _resolve_autoround_layer_config(
            layer_config, "model.layers.0.mlp.shared_experts.gate_proj"
        )
        is None
    )
    assert not any(key.startswith("model.layers.0.") for key in layer_config)


def test_autoround_excludes_mtp_from_blocks_and_keeps_its_linears_float():
    model = _TinyLing(include_mtp=True)

    assert build_autoround_block_names(model) == [["model.layers.0"]]
    layer_config = build_autoround_layer_config(model)
    mtp_linears = [
        name
        for name, module in model.named_modules()
        if name.startswith("model.layers.1.") and isinstance(module, nn.Linear)
    ]
    assert mtp_linears
    assert len(layer_config) == 3
    assert all(
        _resolve_autoround_layer_config(layer_config, name)
        == {"bits": 16, "data_type": "fp"}
        for name in mtp_linears
    )


def test_autoround_nonfinite_loss_guard_fails_before_export():
    class _AutoRound:
        @staticmethod
        def _get_loss(*args, **kwargs):
            return torch.tensor(float("nan"))

    autoround = _AutoRound()
    _install_autoround_nonfinite_loss_guard(autoround)

    with pytest.raises(FloatingPointError, match="non-finite tuning loss"):
        autoround._get_loss()


def _write_native_metadata(output_dir):
    from auto_round.utils import to_standard_regex

    output_dir.mkdir()
    config_path = output_dir / "config.json"
    layer_config = build_autoround_layer_config(_TinyLing(include_mtp=True))
    dynamic = {}
    regex_extra = {}
    for pattern, config in layer_config.items():
        standardized = to_standard_regex(pattern)
        regex_extra[standardized] = config
        if config["bits"] >= 16:
            dynamic[f"-:{standardized}"] = {}
        else:
            dynamic[f"+:{standardized}"] = {"bits": config["bits"]}
    expanded_quantization_config = {
        "bits": 8,
        "group_size": 64,
        "sym": True,
        "data_type": "int_sym_xh",
        "checkpoint_format": "gptq_v2",
        "quant_method": "gptq",
        "block_name_to_quantize": ["model.layers"],
        "dynamic": dynamic,
        "extra_config": {
            **regex_extra,
            "model.layers.0.mlp.experts.0.gate_proj": {"bits": 4},
            "model.layers.0.mlp.experts.0.up_proj": {"bits": 4},
            "model.layers.0.attention.f_proj": {
                "bits": 16,
                "data_type": "fp",
            },
            "model.layers.0.attention.b_proj": {
                "bits": 16,
                "data_type": "fp",
            },
            "model.layers.1.attention.q_proj": {
                "bits": 16,
                "data_type": "fp",
            },
            "model.layers.1.mlp.experts.0.gate_proj": {
                "bits": 16,
                "data_type": "fp",
            },
        },
    }
    config_path.write_text(
        json.dumps(
            {
                "num_hidden_layers": 1,
                "num_nextn_predict_layers": 1,
                "quantization_config": expanded_quantization_config,
            }
        ),
        encoding="utf-8",
    )
    standalone_path = output_dir / "quantization_config.json"
    standalone_path.write_text(
        json.dumps(expanded_quantization_config),
        encoding="utf-8",
    )
    return dynamic


def test_autoround_canonicalizes_native_regex_export(tmp_path):
    output_dir = tmp_path / "quantized"
    dynamic = _write_native_metadata(output_dir)
    config_path = output_dir / "config.json"
    standalone_path = output_dir / "quantization_config.json"

    canonicalize_autoround_gptqmodel_config(output_dir)

    quantization_config = json.loads(config_path.read_text(encoding="utf-8"))[
        "quantization_config"
    ]
    assert quantization_config["dynamic"] == dynamic
    assert "extra_config" not in quantization_config
    assert len(dynamic) == 3
    standalone_quantization_config = json.loads(
        standalone_path.read_text(encoding="utf-8")
    )
    assert standalone_quantization_config["dynamic"] == dynamic
    assert "extra_config" not in standalone_quantization_config
    before = config_path.read_bytes(), standalone_path.read_bytes()
    canonicalize_autoround_gptqmodel_config(output_dir)
    assert (config_path.read_bytes(), standalone_path.read_bytes()) == before


@pytest.mark.parametrize("invalid", ("unknown_rule", "missing_rule", "dtype_mismatch"))
def test_canonicalizer_does_not_rewrite_unverified_metadata(tmp_path, invalid):
    output_dir = tmp_path / "quantized"
    _write_native_metadata(output_dir)
    path = output_dir / "quantization_config.json"
    document = json.loads(path.read_text())
    document.pop("extra_config")
    if invalid == "unknown_rule":
        document["dynamic"]["+:.*unknown.*"] = {"bits": 4}
    elif invalid == "missing_rule":
        document["dynamic"].pop(next(iter(document["dynamic"])))
    else:
        document["data_type"] = "int_sym"
    path.write_text(json.dumps(document))
    config_path = output_dir / "config.json"
    before = config_path.read_bytes(), path.read_bytes()
    with pytest.raises(RuntimeError):
        canonicalize_autoround_gptqmodel_config(output_dir)
    assert (config_path.read_bytes(), path.read_bytes()) == before


@pytest.mark.parametrize("tensor_kind", ("unpackable", "packable", "packed", "missing"))
def test_canonicalizer_verifies_packing_exclusions_against_weights(
    tmp_path, tensor_kind
):
    from safetensors.torch import save_file

    output_dir = tmp_path / "quantized"
    _write_native_metadata(output_dir)
    name = "model.layers.0.attention.g_proj"
    pattern = f"-:^{re.escape(name)}$"
    for path in (output_dir / "config.json", output_dir / "quantization_config.json"):
        document = json.loads(path.read_text())
        quant = document.get("quantization_config", document)
        quant["dynamic"] = {pattern: {}, **quant["dynamic"]}
        quant["extra_config"][name] = {"bits": 16, "data_type": "fp"}
        path.write_text(json.dumps(document))
    tensors = {}
    if tensor_kind == "unpackable":
        tensors[f"{name}.weight"] = torch.zeros(2, 64, dtype=torch.bfloat16)
    elif tensor_kind == "packable":
        tensors[f"{name}.weight"] = torch.zeros(64, 64, dtype=torch.bfloat16)
    elif tensor_kind == "packed":
        tensors[f"{name}.qweight"] = torch.zeros(16, 64, dtype=torch.int32)
    if tensors:
        save_file(tensors, output_dir / "model.safetensors")
    if tensor_kind == "unpackable":
        canonicalize_autoround_gptqmodel_config(output_dir)
        quant = json.loads((output_dir / "config.json").read_text())[
            "quantization_config"
        ]
        assert pattern in quant["dynamic"]
    else:
        with pytest.raises(RuntimeError, match="packing exclusion|packed weights"):
            canonicalize_autoround_gptqmodel_config(output_dir)
