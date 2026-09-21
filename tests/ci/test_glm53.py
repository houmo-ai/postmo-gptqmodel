"""GLM expert semantics, preparation and precision contracts on small tensors."""

import defuser
import pytest
import torch

from _glm53_fixture import write_tiny_glm53
from gptqmodel.models.definitions.glm5_next import (
    Glm53MoELifecycleHooks,
    register_glm53_defuser,
)
from gptqmodel.models.moe_lifecycle import GateUpDownMoELifecycleHooks
from gptqmodel.recipes.glm5_3_flash import (
    prepare_glm5_3_flash_bf16,
    quantize_glm5_3_flash,
)


def test_defused_experts_and_calibration_clamp(tmp_path):
    model = write_tiny_glm53(tmp_path / "source")
    moe = model.model.language_model.layers[1].mlp
    x = torch.randn(1, 12, 128) * 40
    with torch.no_grad():
        reference = moe(x)
        register_glm53_defuser()
        assert defuser.convert_model(model, cleanup_original=True)
        torch.testing.assert_close(moe(x), reference, rtol=1e-4, atol=1e-4)
    gate, up = torch.tensor([[-3.0, 7.0]]), torch.tensor([[-5.0, 9.0]])
    expert = next(child for name, child in moe.experts.named_children() if name == "0")
    actual = Glm53MoELifecycleHooks().expert_intermediate(moe, expert, gate, up)
    torch.testing.assert_close(
        actual, torch.nn.functional.silu(gate.clamp(max=1)) * up.clamp(-1, 1)
    )
    # The shared extension preserves the existing unbounded activation default.
    ordinary = GateUpDownMoELifecycleHooks().expert_intermediate(moe, expert, gate, up)
    torch.testing.assert_close(ordinary, torch.nn.functional.silu(gate) * up)
    assert not torch.equal(actual, ordinary)


def test_preparation_preserves_native_fp32_and_reuse_identity(tmp_path):
    from safetensors.torch import load_file

    source, output = tmp_path / "source", tmp_path / "prepared"
    write_tiny_glm53(source)
    prepare_glm5_3_flash_bf16(
        model_dir=str(source), output_dir=str(output), num_layers=1
    )
    weights = load_file(str(output / "model.safetensors"))
    assert weights["model.language_model.layers.0.hc_attn_fn"].dtype == torch.float32
    assert not any("layers.1." in name for name in weights)
    assert any("model.visual." in name for name in weights)
    assert prepare_glm5_3_flash_bf16(
        model_dir=str(source), output_dir=str(output), num_layers=1
    ) == str(output)
    with pytest.raises(FileExistsError):
        prepare_glm5_3_flash_bf16(
            model_dir=str(source), output_dir=str(output), num_layers=2
        )


def test_recipe_rejects_unsupported_precision(tmp_path):
    with pytest.raises(ValueError, match="W4 G64"):
        quantize_glm5_3_flash(
            model_dir=str(tmp_path / "source"),
            output_dir=str(tmp_path / "out"),
            bits=4,
            dry_run=True,
        )


def test_fp8_preparation_reads_cross_shard_scales_and_partial_blocks(tmp_path):
    import json
    from safetensors.torch import load_file, save_file

    source, output = tmp_path / "source", tmp_path / "prepared"
    write_tiny_glm53(source)
    weights = load_file(str(source / "model.safetensors"))
    scales, expected = {}, {}
    for projection, scale in (("q_proj", 1.25), ("g_a_proj", 0.5)):
        name = f"model.language_model.layers.0.self_attn.{projection}.weight"
        value = weights[name].to(torch.float8_e4m3fn)
        weights[name] = value
        scales[name + "_scale_inv"] = torch.tensor([[scale]], dtype=torch.float32)
        expected[name] = (value.float() * scale).bfloat16()
    save_file(weights, str(source / "weights.safetensors"))
    save_file(scales, str(source / "scales.safetensors"))
    (source / "model.safetensors").unlink()
    weight_map = dict.fromkeys(weights, "weights.safetensors")
    weight_map.update(dict.fromkeys(scales, "scales.safetensors"))
    (source / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}))
    config = json.loads((source / "config.json").read_text())
    config["quantization_config"] = {"quant_method": "fp8", "weight_block_size": [128, 128]}
    (source / "config.json").write_text(json.dumps(config))

    prepare_glm5_3_flash_bf16(model_dir=str(source), output_dir=str(output))
    actual = load_file(str(output / "weights.safetensors"))
    for name, value in expected.items():
        torch.testing.assert_close(actual[name], value, rtol=0, atol=0)
    index = json.loads((output / "model.safetensors.index.json").read_text())
    assert not any(name.endswith("_scale_inv") for name in index["weight_map"])
    assert not (output / "scales.safetensors").exists()
    assert "quantization_config" not in json.loads((output / "config.json").read_text())


def test_calibration_records_consumed_rows_and_tokens():
    from gptqmodel.recipes.glm5_3_flash import _select_calibration_rows

    rows = [
        {"input_ids": list(range(n)), "attention_mask": [1] * n}
        for n in (1, 10, 11, 32)
    ]
    selected, evidence = _select_calibration_rows(rows)
    assert selected == rows[2:]
    assert evidence["actual_nsamples"] == 2 and evidence["actual_tokens"] == 43
    assert evidence["dropped_short_samples"] == 2
    assert len(evidence["tokenized_sha256"]) == 64
    with pytest.raises(ValueError, match="no rows"):
        _select_calibration_rows(rows[:2])


def test_example_uses_public_recipe_contract(tmp_path, capsys):
    import importlib.util
    import json
    from pathlib import Path

    path = (
        Path(__file__).resolve().parents[2]
        / "examples/quantization/examples/example_glm5_3_flash.py"
    )
    spec = importlib.util.spec_from_file_location("glm53_recipe_example", path)
    example = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(example)
    example.main(
        [
            "--model",
            str(tmp_path / "source"),
            "--out",
            str(tmp_path / "out"),
            "--calibration-jsonl",
            str(tmp_path / "rows.jsonl"),
            "--nsamples",
            "16",
            "--dry-run",
        ]
    )
    result = json.loads(capsys.readouterr().out)
    assert (
        result["provenance"]["dry_run"] is True
        and result["calibration"]["nsamples"] == 16
    )
    assert result["precision"]["routed_non_shared_experts"] == "W4 G64"
