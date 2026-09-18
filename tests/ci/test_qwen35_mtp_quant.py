"""Public MTP recipe on a real AutoRound backbone with its head already packed."""
import json
from pathlib import Path

import torch
import pytest

from _qwen35_mtp_fixture import save_mtp_source


@pytest.mark.parametrize("algorithm", ["autoround", "gptq"])
def test_complete_mtp_after_packed_backbone_and_head(tmp_path, algorithm):
    from gptqmodel import GPTQModel
    from gptqmodel.recipes.qwen35_autoround import quantize_qwen35_autoround
    from gptqmodel.recipes.qwen35_mtp import quantize_qwen35_mtp
    from gptqmodel.quantization.qwen35_mtp import CheckpointTensors, Qwen35MTP

    source = save_mtp_source(tmp_path / "source")
    data = tmp_path / "calibration.jsonl"
    data.write_text("\n".join(json.dumps({"text": " ".join(f"t{i}" for i in range(3, 100))})
                               for _ in range(3)) + "\n")
    common = dict(model_dir=str(source), output_dir=str(tmp_path / "main"),
        topology="moe", nsamples=2, seqlen=32, batch_size=1,
        head={"bits": 4, "iters": 2}, device="cuda:0")
    if algorithm == "autoround":
        result = quantize_qwen35_autoround(**common, dataset=str(data), iters=2,
            data_type="int_sym_xh", device_map="0")
    else:
        from gptqmodel.recipes.qwen35 import quantize_qwen35
        result = quantize_qwen35(**common, calibration_jsonl=str(data), device_map=None,
            check_quant_vision_demo=False, skip_inference=True)
        # Existing GPTQ exports may keep quantization metadata in a sidecar.
        root = Path(result.output_dir)
        raw_config = json.loads((root / "config.json").read_text())
        (root / "quantize_config.json").write_text(json.dumps(raw_config.pop("quantization_config")))
        (root / "config.json").write_text(json.dumps(raw_config))
    before = CheckpointTensors(result.output_dir)
    assert "lm_head.qweight" in before.mapping
    assert "mtp.layers.0.mlp.experts.gate_up_proj" in before.mapping
    output = quantize_qwen35_mtp(model_dir=result.output_dir,
        mtp={"iters": 3, "batch_tokens": 32, "validation_nsamples": 1},
        calibration_jsonl=str(data), nsamples=2, seqlen=32, device="cuda:0")
    after = CheckpointTensors(output.output_dir)
    assert output.output_dir == result.output_dir + "_mtp_w4"
    assert before.config["quantization_config"] == after.config["quantization_config"]
    source_config = json.loads((Path(result.output_dir) / "config.json").read_text())
    target_config = json.loads((Path(output.output_dir) / "config.json").read_text())
    target_config.pop("mtp_quantization")
    assert source_config == target_config
    for key in before.mapping:
        if not key.startswith("mtp.layers.0.mlp.experts."):
            assert torch.equal(before.get(key), after.get(key)), key
    assert after.config["mtp_quantization"]["algorithm"] == algorithm
    if algorithm == "autoround":
        assert all(m["loss"] <= m["initial_loss"] for m in output.provenance["experts"])
    assert output.provenance["validation"]["samples"] == 1
    assert output.provenance["validation"]["tokens"] == 31
    loaded = GPTQModel.load(output.output_dir, device="cuda:0", backend="torch", torch_dtype=torch.float16)
    mtp = Qwen35MTP.from_pretrained(output.output_dir).to("cuda:0")
    ids = torch.tensor([[3, 4, 5, 6]], device="cuda:0")
    with torch.inference_mode():
        hidden = loaded.model.model.language_model(input_ids=ids, use_cache=False).last_hidden_state
        embeds = loaded.model.get_input_embeddings()(ids[:, 1:])
        output_hidden = mtp(embeds.half(), hidden[:, :-1].half())
        logits = loaded.model.lm_head(output_hidden)
    assert logits.shape == (1, 3, 256) and torch.isfinite(logits).all()
    reused = quantize_qwen35_mtp(model_dir=output.output_dir, mtp={})
    assert reused.method == "skip"


@pytest.mark.parametrize("algorithm", ["autoround", "gptq"])
@pytest.mark.parametrize("full_recipe", [False, True])
def test_native_hf_and_full_recipe_mtp(full_recipe, algorithm, tmp_path):
    from gptqmodel.recipes.qwen35_autoround import quantize_qwen35_autoround
    from gptqmodel.recipes.qwen35_mtp import quantize_qwen35_mtp
    from gptqmodel.quantization.qwen35_mtp import CheckpointTensors, Qwen35MTP

    source = save_mtp_source(tmp_path / "source")
    data = tmp_path / "data.jsonl"
    data.write_text("\n".join(json.dumps({"text": " ".join(f"t{i}" for i in range(3, 100))})
                               for _ in range(3)) + "\n")
    common = dict(model_dir=str(source), output_dir=str(tmp_path / "out"), nsamples=2, seqlen=32,
                   device="cuda:0", mtp={"iters": 3, "batch_tokens": 32, "validation_nsamples": 1})
    if full_recipe and algorithm == "autoround":
        result = quantize_qwen35_autoround(**common, topology="moe", dataset=str(data),
            iters=2, batch_size=1, data_type="int_sym_xh", head={"bits": 4, "iters": 2}, device_map="0")
        assert Path(result.provenance["log_file"]).is_file()
    elif full_recipe:
        from gptqmodel.recipes.qwen35 import quantize_qwen35
        result = quantize_qwen35(**common, topology="moe", calibration_jsonl=str(data),
            batch_size=1, head={"bits": 4}, device_map=None,
            check_quant_vision_demo=False, skip_inference=True)
    else:
        common["mtp"]["algorithm"] = algorithm
        result = quantize_qwen35_mtp(**common, calibration_jsonl=str(data))
    after = CheckpointTensors(result.output_dir)
    assert after.config["mtp_quantization"]["expert_bits"] == 4
    assert after.config["mtp_quantization"]["algorithm"] == algorithm
    assert ("lm_head.qweight" in after.mapping) == full_recipe
    model = Qwen35MTP.from_pretrained(result.output_dir).cuda()
    with torch.inference_mode():
        x = torch.ones(1, 3, 128, dtype=torch.float16, device="cuda:0")
        assert torch.isfinite(model(x, x)).all()
