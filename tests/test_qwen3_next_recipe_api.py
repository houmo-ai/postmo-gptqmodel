from pathlib import Path

import pytest

from gptqmodel.recipes.qwen3_next import quantize_qwen3_next


def _flag_value(command: list[str], flag: str) -> str:
    assert flag in command
    return command[command.index(flag) + 1]


def test_first_four_layers_are_exposed_in_command(tmp_path: Path):
    result = quantize_qwen3_next(
        model_dir="weights/Qwen3-Next-80B-A3B-Instruct",
        output_dir=str(tmp_path / "out"),
        max_quant_layers=4,
        rotation=False,
        check_saved_precision=True,
        skip_ppl=True,
        dry_run=True,
    )

    command = result.provenance["command"]
    assert _flag_value(command, "--max-quant-layers") == "4"
    assert "--rotation" not in command
    assert "--check-saved-precision" in command
    assert "--skip-ppl" in command
    assert "--no-auto-forward-data-parallel" in command
    assert result.max_quant_layers == 4
    assert result.has_mtp is True


def test_default_recipe_does_not_limit_layers(tmp_path: Path):
    result = quantize_qwen3_next(
        model_dir="weights/Qwen3-Next-80B-A3B-Instruct",
        output_dir=str(tmp_path / "out"),
        dry_run=True,
    )

    assert "--max-quant-layers" not in result.provenance["command"]
    assert result.max_quant_layers is None


def test_negative_layer_limit_is_rejected(tmp_path: Path):
    with pytest.raises(ValueError, match="max_quant_layers must be >= 0"):
        quantize_qwen3_next(
            model_dir="weights/Qwen3-Next-80B-A3B-Instruct",
            output_dir=str(tmp_path / "out"),
            max_quant_layers=-1,
            dry_run=True,
        )


def test_non_gptq_method_is_rejected(tmp_path: Path):
    with pytest.raises(ValueError, match="method must be 'gptq'"):
        quantize_qwen3_next(
            model_dir="weights/Qwen3-Next-80B-A3B-Instruct",
            output_dir=str(tmp_path / "out"),
            method="autoround",
            dry_run=True,
        )


def test_quant_device_and_inference_device_map_are_independent(tmp_path: Path):
    result = quantize_qwen3_next(
        model_dir="weights/Qwen3-Next-80B-A3B-Instruct",
        output_dir=str(tmp_path / "out"),
        device="cuda:0",
        device_map=None,
        infer_device_map="auto",
        dry_run=True,
    )

    command = result.provenance["command"]
    assert _flag_value(command, "--device") == "cuda:0"
    assert _flag_value(command, "--infer-device-map") == "auto"
    assert "--device-map" not in command


def test_qwen3_next_recipe_can_opt_in_to_forward_data_parallel(tmp_path: Path):
    result = quantize_qwen3_next(
        model_dir="weights/Qwen3-Next-80B-A3B-Instruct",
        output_dir=str(tmp_path / "out"),
        auto_forward_data_parallel=True,
        dry_run=True,
    )

    command = result.provenance["command"]
    assert "--auto-forward-data-parallel" in command
    assert "--no-auto-forward-data-parallel" not in command
