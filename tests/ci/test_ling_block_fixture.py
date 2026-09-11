"""Ling layer remapping must preserve both attention and MLP mechanisms."""
import importlib.util
import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file, save_file

spec = importlib.util.spec_from_file_location("prepare_ling_block", Path(__file__).with_name("prepare_ling_block.py"))
cutter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cutter)


def _config():
    return {"model_type": "bailing_hybrid", "num_hidden_layers": 12, "layer_group_size": 6,
            "first_k_dense_replace": 2, "num_nextn_predict_layers": 1, "hidden_size": 2560,
            "num_experts": 512, "num_experts_per_tok": 8, "n_group": 8, "topk_group": 4,
            "expert_swiglu_limit_list": list(range(12)), "share_expert_swiglu_limit_list": list(range(12))}


@pytest.mark.parametrize("mtp", [False, True])
def test_three_mechanisms_and_optional_original_mtp(mtp):
    original = _config()
    config, selected = cutter.layer_plan(original, mtp)
    assert selected == [0, 2, 5] + ([12] if mtp else [])
    assert config["num_hidden_layers"] == config["layer_group_size"] == 3
    assert config["first_k_dense_replace"] == 1
    assert config["num_nextn_predict_layers"] == int(mtp)
    assert config["expert_swiglu_limit_list"] == [0, 2, 5]
    for field in ("hidden_size", "num_experts", "num_experts_per_tok", "n_group", "topk_group"):
        assert config[field] == original[field]
    assert original["num_hidden_layers"] == 12


@pytest.mark.parametrize("field,value", [("quantization_config", {"bits": 4}), ("ling_mla_layout", "xh2_absorbed_v1"),
                                       ("layer_group_size", 1), ("first_k_dense_replace", 5)])
def test_incomplete_or_transformed_structure_is_rejected(field, value):
    config = _config()
    config[field] = value
    # first_k_dense_replace=5 still has KDA/MoE in the second group but MLA is
    # earlier, so preserving source order requires a different cutter.
    with pytest.raises(ValueError):
        cutter.layer_plan(config)


def test_weights_code_router_and_mtp_are_copied_without_changing_source(tmp_path):
    source, output = tmp_path / "source", tmp_path / "block"
    source.mkdir()
    (source / "config.json").write_text(json.dumps(_config()))
    for name in ("configuration_bailing_moe_v3.py", "modeling_bailing_moe_v3.py"):
        (source / name).write_text("# original remote code\n")
    weights = {f"model.layers.{i}.attention.q_proj.weight": torch.full((2, 2), float(i)) for i in range(13)}
    weights["model.layers.2.mlp.gate.expert_bias"] = torch.arange(512.)
    weights["model.word_embeddings.weight"] = torch.ones(3, 2)
    save_file(weights, str(source / "model.safetensors"))
    before = (source / "model.safetensors").read_bytes()
    report = cutter.prepare(source, output, keep_mtp=True)
    actual = load_file(str(output / "model.safetensors"))
    for new, old in enumerate([0, 2, 5, 12]):
        assert torch.equal(actual[f"model.layers.{new}.attention.q_proj.weight"], weights[f"model.layers.{old}.attention.q_proj.weight"])
    assert torch.equal(actual["model.layers.1.mlp.gate.expert_bias"], weights["model.layers.2.mlp.gate.expert_bias"])
    assert actual["model.layers.3.attention.q_proj.weight"][0, 0] == 12
    assert report["source_layers"] == [0, 2, 5, 12]
    assert report["source_code_sha256"]["modeling_bailing_moe_v3.py"] == cutter._sha256(output / "modeling_bailing_moe_v3.py")
    assert (source / "model.safetensors").read_bytes() == before
    with pytest.raises(ValueError, match="new output directory"):
        cutter.prepare(source, output)
