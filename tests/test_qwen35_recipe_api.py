from __future__ import annotations

from pathlib import Path

import pytest

from gptqmodel.recipes.qwen35 import quantize_qwen35


def _cmd(result):
    return result.provenance["command"]


def _flag_value(command: list[str], flag: str) -> str:
    assert flag in command
    return command[command.index(flag) + 1]


def test_dense_gptq_defaults_match_readme_calibration(tmp_path: Path):
    result = quantize_qwen35(
        model_dir="weights/Qwen3.5-9B",
        output_dir=str(tmp_path / "dense-out"),
        method="gptq",
        topology="dense",
        rotation=False,
        dry_run=True,
    )

    command = _cmd(result)
    assert result.topology == "dense"
    assert result.method == "gptq"
    assert result.rotation is None
    assert "example_qwen35dense.py" in command[1]
    assert _flag_value(command, "--bits") == "4"
    assert _flag_value(command, "--group-size") == "64"
    assert _flag_value(command, "--batch-size") == "1"
    assert _flag_value(command, "--nsamples") == "256"
    assert _flag_value(command, "--seqlen") == "1024"
    assert _flag_value(command, "--calibration-jsonl").endswith(
        "gptqmodel/quantization/calibration/dense_ivsg/gen_data/Qwen3.5-27B.jsonl"
    )
    assert "--no-offload-to-disk" in command
    assert "--hessian-mse" in command
    assert "--wait-for-submodule-finalizers" in command
    assert "--rotation" not in command


def test_moe_gptq_defaults_match_readme_calibration(tmp_path: Path):
    result = quantize_qwen35(
        model_dir="weights/Qwen3.6-35B-A3B",
        output_dir=str(tmp_path / "moe-out"),
        method="gptq",
        topology="moe",
        rotation=False,
        dry_run=True,
    )

    command = _cmd(result)
    assert result.topology == "moe"
    assert "example_qwen35moe.py" in command[1]
    assert _flag_value(command, "--bits") == "4"
    assert _flag_value(command, "--self-attn-bits") == "8"
    assert _flag_value(command, "--shared-expert-bits") == "8"
    assert _flag_value(command, "--expert-bits") == "4"
    assert _flag_value(command, "--expert-down-bits") == "5"
    assert _flag_value(command, "--batch-size") == "1"
    assert _flag_value(command, "--nsamples") == "512"
    assert _flag_value(command, "--seqlen") == "1024"
    assert _flag_value(command, "--calibration-jsonl").endswith(
        "gptqmodel/quantization/calibration/moe_ebss/gen_data/Qwen3-Next-80B-A3B-Instruct.jsonl"
    )
    assert "--no-offload-to-disk" in command
    assert "--hessian-mse" in command
    assert "--wait-for-submodule-finalizers" in command
    assert "--rotation" not in command


def test_rotation_with_offload_is_rejected(tmp_path: Path):
    with pytest.raises(ValueError, match="rotated quantization requires offload_to_disk=False"):
        quantize_qwen35(
            model_dir="weights/Qwen3.5-9B",
            output_dir=str(tmp_path / "bad"),
            method="gptq",
            topology="dense",
            rotation="hadamard",
            offload_to_disk=True,
            dry_run=True,
        )


def test_gptq_recipe_rejects_autoround_method(tmp_path: Path):
    with pytest.raises(ValueError, match="method must be one of"):
        quantize_qwen35(
            model_dir="weights/Qwen3.5-9B",
            output_dir=str(tmp_path / "bad-method"),
            method="autoround",
            topology="dense",
            dry_run=True,
        )


def test_rotate_named_preset_requires_rotation(tmp_path: Path):
    with pytest.raises(ValueError, match="requires a non-null rotation"):
        quantize_qwen35(
            model_dir="weights/Qwen3.5-9B",
            output_dir=str(tmp_path / "bad-preset"),
            method="gptq",
            preset="full_vlm_rotate",
            topology="dense",
            rotation=False,
            dry_run=True,
        )


def test_dense_gptq_accepts_hessian_and_mse_flags(tmp_path: Path):
    result = quantize_qwen35(
        model_dir="weights/Qwen3.5-9B",
        output_dir=str(tmp_path / "dense-out"),
        method="gptq",
        topology="dense",
        hessian_mse=False,
        mse=1.5,
        dry_run=True,
    )

    command = _cmd(result)
    assert "--no-hessian-mse" in command
    assert "--wait-for-submodule-finalizers" in command
    assert _flag_value(command, "--mse") == "1.5"
