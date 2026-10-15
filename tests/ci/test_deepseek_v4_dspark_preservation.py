# Copyright 2026 HOUMO AI
# SPDX-License-Identifier: Apache-2.0
"""Lossless preservation of the draft omitted by target-only Transformers."""
import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file, save_file

from gptqmodel.utils.deepseek_v4_dspark import (
    inspect_deepseek_v4_dspark,
    prepare_deepseek_v4_dspark_bf16,
    preserve_deepseek_v4_dspark,
)


def _source(root: Path, *, mixed: bool = False) -> dict[str, torch.Tensor]:
    root.mkdir()
    config = {"model_type": "deepseek_v4", "hidden_size": 8, "vocab_size": 16,
              "num_hidden_layers": 3, "dspark_target_layer_ids": [0, 1, 2],
              "dspark_markov_rank": 4, "dspark_block_size": 5,
              "dspark_noise_token_id": 15, "expert_dtype": "fp4",
              "quantization_config": {"quant_method": "fp8", "fmt": "e4m3"}}
    (root / "config.json").write_text(json.dumps(config))
    torch.manual_seed(13)
    tensors = {f"mtp.{i}.attn.wkv.weight": torch.randn(4, 8, dtype=torch.bfloat16)
               for i in range(3)}
    tensors.update({"mtp.0.main_proj.weight": torch.randn(8, 24, dtype=torch.bfloat16),
                    "mtp.0.main_norm.weight": torch.ones(8),
                    "mtp.2.markov_head.markov_w1.weight": torch.randn(16, 4),
                    "mtp.2.markov_head.markov_w2.weight": torch.randn(16, 4),
                    "mtp.2.confidence_head.proj.weight": torch.randn(1, 12),
                    "mtp.1.ffn.experts.0.w1.weight": torch.arange(64, dtype=torch.int8).reshape(4, 16),
                    "mtp.1.ffn.experts.0.w1.scale": torch.ones(4, 1, dtype=torch.float8_e4m3fn)})
    values = dict(tensors)
    if mixed:
        values["head.weight"] = torch.randn(16, 8)
    save_file(values, root / "weights.safetensors")
    (root / "model.safetensors.index.json").write_text(json.dumps(
        {"weight_map": dict.fromkeys(values, "weights.safetensors")}))
    return tensors


@pytest.mark.parametrize("mixed", [False, True])
def test_preserve_native_dspark_and_round_trip_without_changing_target(tmp_path, mixed):
    source, target = tmp_path / "native", tmp_path / "packed"
    expected = _source(source, mixed=mixed)
    target.mkdir()
    body = {"model.layers.0.self_attn.q_a_proj.qweight": torch.arange(16, dtype=torch.int32)}
    save_file(body, target / "target.safetensors")
    before = (target / "target.safetensors").read_bytes()
    manifest = preserve_deepseek_v4_dspark(source, target)
    assert manifest["num_draft_layers"] == 3
    assert manifest["shared_lm_head"] == "lm_head"
    assert manifest["weight_format"] == "deepseek_v4_native_fp4_fp8"
    actual = load_file(target / "dspark" / "weights.safetensors")
    assert actual.keys() == expected.keys()
    for key, tensor in expected.items():
        assert actual[key].dtype == tensor.dtype
        assert torch.equal(actual[key].view(torch.uint8), tensor.view(torch.uint8)), key
    assert (target / "target.safetensors").read_bytes() == before
    assert preserve_deepseek_v4_dspark(source, target) == manifest
    # The saved component can be carried through subsequent model.save calls.
    copied = tmp_path / "resaved"
    assert preserve_deepseek_v4_dspark(target, copied) == manifest
    assert inspect_deepseek_v4_dspark(copied)["weight_map"] == dict.fromkeys(expected, "weights.safetensors")


def test_dspark_rejects_missing_layer_before_creating_output(tmp_path):
    source, target = tmp_path / "native", tmp_path / "packed"
    _source(source)
    path = source / "model.safetensors.index.json"
    index = json.loads(path.read_text())
    index["weight_map"].pop("mtp.1.attn.wkv.weight")
    path.write_text(json.dumps(index))
    with pytest.raises(ValueError, match="Incomplete DSpark weights"):
        preserve_deepseek_v4_dspark(source, target)
    assert not target.exists()


def test_dspark_refuses_same_config_with_different_weight_values(tmp_path):
    source, target = tmp_path / "native", tmp_path / "packed"
    values = _source(source)
    preserve_deepseek_v4_dspark(source, target)
    values["mtp.2.markov_head.markov_w1.weight"].add_(1)
    save_file(values, source / "weights.safetensors")
    with pytest.raises(FileExistsError, match="different DSpark"):
        preserve_deepseek_v4_dspark(source, target)


def test_target_only_checkpoint_does_not_claim_a_dspark_component(tmp_path):
    source, target = tmp_path / "native", tmp_path / "packed"
    _source(source)
    (source / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {}}))
    assert preserve_deepseek_v4_dspark(source, target) is None
    assert not target.exists()


def test_deepseek_model_save_hook_preserves_the_component(tmp_path):
    from gptqmodel.models.definitions.deepseek_v4 import DeepSeekV4QModel

    source, prepared, output = tmp_path / "native", tmp_path / "prepared", tmp_path / "packed"
    _source(source)
    preserve_deepseek_v4_dspark(source, prepared)
    owner = object.__new__(DeepSeekV4QModel)
    owner.model_local_path = str(prepared)
    owner.after_model_save(str(output))
    assert inspect_deepseek_v4_dspark(output)["num_draft_layers"] == 3


def test_dspark_preparation_separates_heads_and_native_expert_format(tmp_path):
    source, output = tmp_path / "native", tmp_path / "prepared"
    _source(source)
    result = prepare_deepseek_v4_dspark_bf16(source, output)
    index = json.loads((output / "model.safetensors.index.json").read_text())
    config = json.loads((output / "config.json").read_text())
    assert config["num_hidden_layers"] == 3 and config["num_hash_layers"] == 0
    assert config["layer_types"] == ["sliding_attention"] * 3
    assert "quantization_config" not in config
    assert result["shared_lm_head"] == "target"
    weights = load_file(output / index["weight_map"]["layers.1.ffn.experts.0.w1.weight"])
    assert weights["layers.1.ffn.experts.0.w1.weight"].dtype == torch.bfloat16
    assert weights["layers.1.ffn.experts.0.w1.weight"].shape == (4, 32)
    assert not any(key.endswith(".scale") for key in index["weight_map"])
    heads = load_file(output / result["heads"])
    assert "main_proj.weight" in heads and "markov_head.markov_w2.weight" in heads
    assert not any(key.startswith(("head.", "embed.")) for key in index["weight_map"])
