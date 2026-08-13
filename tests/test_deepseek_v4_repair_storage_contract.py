from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "examples"
    / "quantization"
    / "examples"
    / "repair_deepseek_v4_packed_checkpoint.py"
)
SPEC = importlib.util.spec_from_file_location(
    "repair_deepseek_v4_packed_checkpoint_test",
    SCRIPT,
)
assert SPEC is not None and SPEC.loader is not None
REPAIR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(REPAIR)


def test_normalize_gptq_storage_contract_preserves_autoround_metadata(tmp_path):
    auto_round = {"auto_round_enabled": True, "iters": 200}
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "quantization_config": {
                    "quant_method": "auto_round",
                    "meta": {"auto_round": auto_round},
                }
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "quantize_config.json").write_text(
        json.dumps(
            {
                "quant_method": "auto_round",
                "meta": {"auto_round": auto_round},
            }
        ),
        encoding="utf-8",
    )

    assert REPAIR.normalize_gptq_storage_contract(tmp_path) is True

    model_config = json.loads((tmp_path / "config.json").read_text())
    quant_config = json.loads((tmp_path / "quantize_config.json").read_text())
    assert model_config["quantization_config"]["quant_method"] == "gptq"
    assert quant_config["quant_method"] == "gptq"
    assert model_config["quantization_config"]["meta"]["auto_round"] == auto_round
    assert quant_config["meta"]["auto_round"] == auto_round
    assert REPAIR.normalize_gptq_storage_contract(tmp_path) is False


def test_validate_completed_output_shards_requires_every_exact_shard(tmp_path):
    save_file({"first": torch.ones(1)}, tmp_path / "model-00001.safetensors")
    save_file({"wrong": torch.ones(1)}, tmp_path / "model-00002.safetensors")
    shard_keys = {
        "model-00001.safetensors": {"first"},
        "model-00002.safetensors": {"second"},
    }

    with pytest.raises(RuntimeError, match="model-00002.safetensors"):
        REPAIR.validate_completed_output_shards(tmp_path, shard_keys)

    save_file({"second": torch.ones(1)}, tmp_path / "model-00002.safetensors")
    REPAIR.validate_completed_output_shards(tmp_path, shard_keys)
