"""MTP joint AutoRound, source preservation, serialized validation and reload."""
import json

import pytest
import torch
from safetensors.torch import load_file, save_file

from _qwen35_mtp_fixture import save_mtp_source
from _qwen35_mixed_fixture import native_qwen35_cpu
from gptqmodel.quantization.qwen35_mtp import (
    CheckpointTensors, MTPQuantizationConfig, Qwen35MTP, calibrate_mtp_experts,
    inspect_mtp_checkpoint, save_mtp_checkpoint,
)


@pytest.fixture(scope="module", params=["autoround", "gptq"])
def calibrated(tmp_path_factory, request):
    torch.set_num_threads(2)
    root = tmp_path_factory.mktemp("mtp")
    source = save_mtp_source(root / "source")
    model = Qwen35MTP.from_pretrained(source)
    captured = []
    hook = model.layers[0].mlp.register_forward_pre_hook(lambda m, a: captured.append(a[0].detach().cpu()))
    generator = torch.Generator().manual_seed(5)
    inputs = [torch.randn(1, 32, 128, generator=generator).half() for _ in range(2)]
    with torch.no_grad():
        reference = model(*inputs)
    hook.remove()
    options = MTPQuantizationConfig(algorithm=request.param, iters=20, batch_tokens=32)
    payload, metrics = calibrate_mtp_experts(model, captured, options, device="cpu")
    target = root / "packed"
    save_mtp_checkpoint(source, target, payload, options, {"experts": metrics})
    return source, target, inputs, reference, metrics


def test_joint_autoround_reload_and_unchanged_main(calibrated):
    source, target, inputs, reference, metrics = calibrated
    before, after = CheckpointTensors(source), CheckpointTensors(target)
    for name in before.mapping:
        if not name.startswith("mtp.layers.0.mlp.experts."):
            assert torch.equal(before.get(name), after.get(name)), name
    assert before.config["text_config"] == after.config["text_config"]
    assert len([k for k in after.mapping if k.endswith(".qweight")]) == 4 * 3
    if after.config["mtp_quantization"]["algorithm"] == "autoround":
        assert all(m["loss"] <= m["initial_loss"] for m in metrics)
        assert any(m["best_step"] > 0 for m in metrics)
    else:
        assert all(m["true_sequential"] for m in metrics)
        assert all(p["hessian_tokens"] == m["tokens"] for m in metrics for p in m["projections"].values())
    reloaded = Qwen35MTP.from_pretrained(target)
    with torch.no_grad():
        output = reloaded(*inputs)
    assert torch.isfinite(output).all()
    assert (output - reference).float().square().mean() < 0.02
    assert inspect_mtp_checkpoint(target, MTPQuantizationConfig())["expert_bits"] == 4
    from gptqmodel.recipes.qwen35_mtp import quantize_qwen35_mtp
    reused = quantize_qwen35_mtp(model_dir=target, mtp={}, calibration_jsonl="/missing/data.jsonl")
    assert reused.method == "skip" and reused.output_dir == str(target)


def test_mtp_alignment_is_causal(calibrated):
    source, _, inputs, _, _ = calibrated
    model = Qwen35MTP.from_pretrained(source)
    changed = [x.clone() for x in inputs]
    changed[0][:, 16:] += 2
    changed[1][:, 16:] -= 2
    with torch.no_grad():
        before, after = model(*inputs), model(*changed)
    torch.testing.assert_close(before[:, :16], after[:, :16], rtol=0, atol=0)
    assert not torch.equal(before[:, 16:], after[:, 16:])


@pytest.mark.parametrize("value", [{"expert_bits": 5}, {"algorithm": "rtn"}, {"iters": 0},
                                    {"group_size": 128}, {"data_type": "int"}])
def test_reject_unsupported_options(value):
    with pytest.raises(ValueError):
        MTPQuantizationConfig(**value)


@pytest.mark.parametrize("suffix,diagnostic", [("g_idx", "sequential"), ("qzeros", "zero points")])
def test_incomplete_packed_payload_rejected(calibrated, tmp_path, suffix, diagnostic):
    import shutil
    _, target, _, _, _ = calibrated
    copied = tmp_path / "bad"
    shutil.copytree(target, copied)
    index = json.loads((copied / "model.safetensors.index.json").read_text())
    key = next(k for k in index["weight_map"] if k.endswith("." + suffix))
    filename = index["weight_map"][key]
    data = load_file(copied / filename)
    data[key][0] = 1
    save_file(data, copied / filename)
    with pytest.raises(ValueError, match=diagnostic):
        inspect_mtp_checkpoint(copied, MTPQuantizationConfig())


@pytest.mark.parametrize("algorithm", ["autoround", "gptq"])
def test_full_recipe_has_final_mtp_stage(algorithm, tmp_path):
    from gptqmodel.recipes.qwen35_autoround import quantize_qwen35_autoround
    from gptqmodel.recipes.qwen35 import quantize_qwen35
    source = save_mtp_source(tmp_path / "source")
    data = tmp_path / "data.jsonl"
    data.write_text('{"text": "t3 t4 t5"}\n')
    options = dict(model_dir=str(source), output_dir=str(tmp_path / "out"), topology="moe",
                   head={"bits": 4}, mtp={"iters": 3}, mtp_replay_device="cpu", dry_run=True)
    if algorithm == "autoround":
        result = quantize_qwen35_autoround(**options, dataset=str(data))
    else:
        result = quantize_qwen35(**options, calibration_jsonl=str(data))
    assert result.provenance["head_stage"]["head"]["bits"] == 4
    assert result.provenance["mtp_stage"]["mtp"]["iters"] == 3
    assert result.provenance["mtp_stage"]["mtp"]["algorithm"] == algorithm
    assert result.provenance["mtp_stage"]["replay_device"] == "cpu"
    assert result.output_dir == str(tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_mtp_input_replay_uses_explicit_device(tmp_path, native_qwen35_cpu, monkeypatch):
    from gptqmodel.recipes.qwen35_mtp import collect_mtp_inputs
    from _qwen35_mixed_fixture import tiny_model
    from transformers import AutoModelForImageTextToText

    source = save_mtp_source(tmp_path / "source")
    model = Qwen35MTP.from_pretrained(source)
    calibration = [{"input_ids": torch.tensor([[3, 4, 5, 6, 7, 8]])} for _ in range(2)]
    # Keep the tiny model's native Torch attention when the saved model is
    # loaded: FLA's optional Triton kernel cannot execute on a CPU tensor.
    monkeypatch.setattr(AutoModelForImageTextToText, "from_pretrained", lambda *a, **k: tiny_model().half())
    inputs, validation = collect_mtp_inputs(
        source, model, calibration, device="cpu", replay_device="cpu", validation_nsamples=1,
    )
    assert inputs and len(validation) == 1
    assert validation[0]["hidden"].shape == (1, 5, 128)
    assert validation[0]["embedding"].shape == (1, 5, 128)
    owner = AutoModelForImageTextToText.from_pretrained(source, dtype=torch.float16)
    with torch.inference_mode():
        expected = owner.model.language_model(input_ids=calibration[0]["input_ids"], use_cache=False).last_hidden_state
    torch.testing.assert_close(validation[0]["main_next_hidden"], expected[:, 1:], rtol=1e-3, atol=1e-3)


def test_mtp_inherits_actual_producer_and_rejects_conflict(tmp_path):
    from gptqmodel.quantization.qwen35_mtp import resolve_mtp_algorithm
    source = save_mtp_source(tmp_path / "source")
    with pytest.raises(ValueError, match="explicit algorithm"):
        resolve_mtp_algorithm(source)
    config = json.loads((source / "config.json").read_text())
    config["quantization_config"] = {"quant_method": "gptq", "provider": "auto-round"}
    (source / "config.json").write_text(json.dumps(config))
    assert resolve_mtp_algorithm(source) == "autoround"
    with pytest.raises(ValueError, match="conflicts"):
        resolve_mtp_algorithm(source, "gptq")
    del config["quantization_config"]["provider"]
    (source / "config.json").write_text(json.dumps(config))
    assert resolve_mtp_algorithm(source) == "gptq"
