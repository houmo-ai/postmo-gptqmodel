"""Representative layer selection and real safetensors slicing contracts."""
import copy
import importlib.util
import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file, save_file

spec = importlib.util.spec_from_file_location("prepare_block", Path(__file__).with_name("prepare_block.py"))
cutter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cutter)


def _config(family="qwen3_5", shared=0):
    kind = "sliding_attention" if family == "gemma4" else "linear_attention"
    return {"model_type": family, "text_config": {
        "num_hidden_layers": 8, "layer_types": [kind, kind, kind, "full_attention"] * 2,
        "num_kv_shared_layers": shared, "hidden_size": 4,
        "hidden_size_per_layer_input": 2 if family == "gemma4" else 0,
        "num_experts": 256, "num_experts_per_tok": 8, "mtp_num_hidden_layers": 1,
    }}


@pytest.mark.parametrize("family", ["qwen3_5", "qwen3_5_moe", "gemma4"])
def test_minimum_keeps_attention_kinds_and_original_dimensions(family):
    original = _config(family)
    before = copy.deepcopy(original)
    config, indices = cutter.layer_plan(original)
    assert original == before
    assert indices == [0, 3]
    text = config["text_config"]
    assert text["num_hidden_layers"] == 2
    assert text["hidden_size"] == 4
    assert (text["num_experts"], text["num_experts_per_tok"]) == (256, 8)
    assert text["mtp_num_hidden_layers"] == 0


def test_gemma_keeps_actual_kv_donors_and_borrowers():
    config = _config("gemma4", shared=4)
    config["text_config"]["per_layer_config"] = [{"source": i} for i in range(8)]
    result, indices = cutter.layer_plan(config)
    assert indices == [2, 3, 4, 7]
    assert result["text_config"]["num_kv_shared_layers"] == 2
    assert result["text_config"]["per_layer_config"] == [{"source": i} for i in indices]


@pytest.mark.parametrize("invalid", ["quantized", "missing_kind", "missing_donor", "sparse_step"])
def test_reject_inputs_that_lose_required_structure(invalid):
    config = _config("gemma4", shared=4)
    if invalid == "quantized":
        config["quantization_config"] = {"bits": 4}
    elif invalid == "missing_kind":
        config["text_config"]["layer_types"] = ["sliding_attention"] * 8
    elif invalid == "missing_donor":
        config["text_config"]["layer_types"][3] = "sliding_attention"
    else:
        config["text_config"]["decoder_sparse_step"] = 2
    with pytest.raises(ValueError):
        cutter.layer_plan(config)


def _checkpoint(path):
    path.mkdir()
    config = _config("gemma4", shared=4)
    (path / "config.json").write_text(json.dumps(config))
    (path / "tokenizer_config.json").write_text("{}")
    prefix = "model.language_model."
    tensors = {f"{prefix}layers.{i}.self_attn.q_proj.weight": torch.full((4, 4), float(i)) for i in range(8)}
    tensors.update({f"{prefix}layers.{i}.self_attn.k_proj.weight": torch.full((4, 4), float(i)) for i in range(8)})
    tensors[f"{prefix}embed_tokens_per_layer.weight"] = torch.arange(48.).reshape(3, 16)
    tensors[f"{prefix}per_layer_model_projection.weight"] = torch.arange(64.).reshape(16, 4)
    tensors["model.visual.patch_embed.weight"] = torch.ones(2, 2)
    tensors["mtp.layers.0.weight"] = torch.ones(2, 2)
    save_file(tensors, str(path / "model.safetensors"))
    return tensors


def test_slicing_remaps_weights_and_packed_per_layer_inputs(tmp_path):
    source, output = tmp_path / "original", tmp_path / "block"
    original = _checkpoint(source)
    before = (source / "model.safetensors").read_bytes()
    report = cutter.prepare(source, output)
    weights = load_file(str(output / "model.safetensors"))
    selected = [2, 3, 4, 7]
    assert report["source_layers"] == selected
    for new, old in enumerate(selected):
        assert torch.equal(weights[f"model.language_model.layers.{new}.self_attn.q_proj.weight"],
                           original[f"model.language_model.layers.{old}.self_attn.q_proj.weight"])
        assert (f"model.language_model.layers.{new}.self_attn.k_proj.weight" in weights) == (old < 4)
    columns = [2 * i + offset for i in selected for offset in (0, 1)]
    for suffix, dimension in (("embed_tokens_per_layer.weight", 1), ("per_layer_model_projection.weight", 0)):
        key = f"model.language_model.{suffix}"
        assert torch.equal(weights[key], original[key].index_select(dimension, torch.tensor(columns)))
    assert torch.equal(weights["model.visual.patch_embed.weight"], original["model.visual.patch_embed.weight"])
    assert not any("mtp." in key for key in weights)
    assert (output / "tokenizer_config.json").exists()
    assert report["weights_sha256"] == cutter._sha256(output / "model.safetensors")
    assert (source / "model.safetensors").read_bytes() == before
    with pytest.raises(ValueError, match="new output directory"):
        cutter.prepare(source, output)


def test_missing_source_shard_cannot_produce_partial_success(tmp_path):
    source = tmp_path / "original"
    _checkpoint(source)
    (source / "model.safetensors.index.json").write_text(json.dumps({
        "weight_map": {"model.language_model.layers.0.mlp.weight": "missing.safetensors"}}))
    with pytest.raises(ValueError, match="incomplete source"):
        cutter.prepare(source, tmp_path / "block")
    assert not (tmp_path / "block").exists()


@pytest.mark.parametrize("invalid", [None, "erased_weight", "negative", "nan"])
def test_scale_validation_allows_only_unrepresentable_zero_groups(invalid):
    spec = importlib.util.spec_from_file_location("quant_guard", Path(__file__).with_name("test_quantization_guard.py"))
    guard = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(guard)
    weights = torch.tensor([[1e-37] * 64, [1.0] * 64])
    scales = torch.tensor([[0.0, 0.1]], dtype=torch.float16)
    if invalid == "erased_weight":
        weights[0] = 1.0
    elif invalid == "negative":
        scales[0, 1] = -0.1
    elif invalid == "nan":
        scales[0, 1] = float("nan")
    if invalid:
        with pytest.raises(AssertionError):
            guard._assert_scales(scales, weights, 64)
    else:
        guard._assert_scales(scales, weights, 64)
