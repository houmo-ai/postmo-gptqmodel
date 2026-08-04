from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file

from gptqmodel.quantization.config import dynamic_get
from gptqmodel.models.definitions.qwen3_next import Qwen3NextGPTQ
from gptqmodel.utils.missing_tensors import (
    count_checkpoint_tensors_with_prefix,
    copy_missing_tensors_from_source,
    validate_tensor_prefix_parity,
)


def _load_example_module():
    path = Path(__file__).resolve().parents[1] / "examples/quantization/examples/example_qwen3_next.py"
    spec = importlib.util.spec_from_file_location("example_qwen3_next_under_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


example = _load_example_module()


def _args(**overrides):
    values = {
        "max_quant_layers": None,
        "self_attn_bits": 8,
        "linear_attn_bits": 8,
        "dense_mlp_bits": 4,
        "expert_bits": 4,
        "shared_expert_bits": 8,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_layer_limit_excludes_layers_four_and_above():
    skips = example.layer_limit_dynamic_skips(48, 4)

    assert len(skips) == 44
    assert r"-:.*\.layers\.4\..*" in skips
    assert r"-:.*\.layers\.47\..*" in skips
    assert r"-:.*\.layers\.3\..*" not in skips


@pytest.mark.parametrize("limit", [-1, 49])
def test_layer_limit_rejects_out_of_range_values(limit):
    with pytest.raises(ValueError, match="between 0 and 48"):
        example.layer_limit_dynamic_skips(48, limit)


def test_no_layer_limit_adds_no_exclusions():
    assert example.layer_limit_dynamic_skips(48, None) == {}


def test_exclusions_take_precedence_over_bit_overrides():
    dynamic = example.build_dynamic_quant_config(_args(max_quant_layers=4), 48)

    assert next(iter(dynamic)) == r"-:.*\.layers\.4\..*"
    assert dynamic_get(dynamic, "model.layers.4.self_attn.q_proj") is False
    assert dynamic_get(dynamic, "model.layers.3.self_attn.q_proj", "bits") == 8


def _write_source_and_quant_target(tmp_path: Path):
    source = tmp_path / "source"
    target = tmp_path / "target"
    source.mkdir()
    target.mkdir()
    save_file(
        {
            "model.layers.0.self_attn.q_proj.weight": torch.ones(2, 2, dtype=torch.bfloat16),
            "mtp.fc.weight": torch.full((2, 2), 2.0, dtype=torch.bfloat16),
            "mtp.layers.0.norm.weight": torch.full((2,), 3.0, dtype=torch.bfloat16),
            "auxiliary.unrelated.weight": torch.full((2,), 4.0, dtype=torch.bfloat16),
        },
        source / "model.safetensors",
        metadata={"format": "pt"},
    )
    target_shard = target / "model-00001-of-00001.safetensors"
    save_file(
        {
            "model.layers.0.self_attn.q_proj.qweight": torch.ones(1, 2, dtype=torch.int32),
        },
        target_shard,
        metadata={"format": "pt"},
    )
    (target / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {},
                "weight_map": {
                    "model.layers.0.self_attn.q_proj.qweight": target_shard.name,
                },
            }
        ),
        encoding="utf-8",
    )
    return source, target


def test_restore_copies_only_mtp_and_preserves_bf16(tmp_path: Path):
    source, target = _write_source_and_quant_target(tmp_path)

    copied = copy_missing_tensors_from_source(
        str(source), str(target), tensor_name_prefixes=("mtp.",)
    )

    assert copied == 2
    assert validate_tensor_prefix_parity(str(source), str(target), "mtp.") == 2
    with safe_open(
        target / "missing_tensors_passthrough.safetensors", framework="pt", device="cpu"
    ) as handle:
        assert set(handle.keys()) == {"mtp.fc.weight", "mtp.layers.0.norm.weight"}
        assert handle.get_tensor("mtp.fc.weight").dtype == torch.bfloat16


def test_target_only_checkpoint_skips_mtp_preservation(tmp_path: Path, capsys):
    source = tmp_path / "source"
    target = tmp_path / "target"
    source.mkdir()
    target.mkdir()
    save_file(
        {"model.layers.0.self_attn.q_proj.weight": torch.ones(2, 2)},
        source / "model.safetensors",
        metadata={"format": "pt"},
    )
    save_file(
        {"model.layers.0.self_attn.q_proj.qweight": torch.ones(1, 2, dtype=torch.int32)},
        target / "model.safetensors",
        metadata={"format": "pt"},
    )

    assert count_checkpoint_tensors_with_prefix(str(source), "mtp.") == 0
    assert example.preserve_mtp_tensors_if_present(str(source), str(target), rotation=False) == 0
    assert "keeping target-only output" in capsys.readouterr().out


def test_count_checkpoint_tensors_with_prefix_uses_index(tmp_path: Path):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "weight_map": {
                    "model.layers.0.weight": "model-00001.safetensors",
                    "mtp.fc.weight": "model-00002.safetensors",
                    "mtp.layers.0.norm.weight": "model-00002.safetensors",
                }
            }
        ),
        encoding="utf-8",
    )

    assert count_checkpoint_tensors_with_prefix(str(checkpoint), "mtp.") == 2


def test_mtp_parity_reports_missing_tensor(tmp_path: Path):
    source, target = _write_source_and_quant_target(tmp_path)

    with pytest.raises(ValueError, match="missing 2 tensors with prefix 'mtp.'"):
        validate_tensor_prefix_parity(str(source), str(target), "mtp.")


def test_precision_metrics_report_cosine_and_top1():
    metrics = example.compare_last_token_logits(
        torch.tensor([1.0, 3.0, -1.0]),
        torch.tensor([1.1, 2.9, -0.9]),
    )

    assert metrics["cosine_similarity"] > 0.99
    assert metrics["reference_top1"] == 1
    assert metrics["candidate_top1"] == 1
    assert metrics["top1_match"] is True
    assert metrics["max_abs_error"] == pytest.approx(0.1)


def test_precision_gate_rejects_bad_saved_logits():
    with pytest.raises(RuntimeError, match="saved-checkpoint precision gate"):
        example.enforce_precision_gate(
            {"cosine_similarity": 0.8, "top1_match": False},
            min_logit_cosine=0.99,
            require_top1_match=True,
        )


def test_precision_gate_allows_configured_top1_mismatch():
    example.enforce_precision_gate(
        {"cosine_similarity": 0.995, "top1_match": False},
        min_logit_cosine=0.99,
        require_top1_match=False,
    )


def test_auto_experts_implementation_uses_eager_on_a100(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _index: (8, 0))

    assert example.resolve_experts_implementation("auto") == "eager"


def test_auto_experts_implementation_keeps_grouped_mm_on_hopper(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _index: (9, 0))

    assert example.resolve_experts_implementation("auto") == "grouped_mm"


def test_qwen3_next_load_patch_keeps_checkpoint_experts_defused(monkeypatch):
    import defuser
    import defuser.model_registry as registry

    calls = []
    monkeypatch.setattr(defuser, "replace_fused_blocks", calls.append)

    Qwen3NextGPTQ.before_model_load(None, load_quantized_model=False)

    assert calls == ["qwen3_next"]
    assert registry.MODEL_CONFIG["qwen3_next"]["checkpoint_mapping"] == []
