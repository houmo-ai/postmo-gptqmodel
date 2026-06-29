from __future__ import annotations

import ast
from pathlib import Path

import pytest

from gptqmodel.recipes.qwen35_autoround import quantize_qwen35_autoround


def _cmd(result):
    return result.provenance["command"]


def _flag_value(command: list[str], flag: str) -> str:
    assert flag in command
    return command[command.index(flag) + 1]


def _parser_flags(script_path: str) -> set[str]:
    tree = ast.parse(Path(script_path).read_text(encoding="utf-8"))
    flags: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not isinstance(node.func, ast.Attribute) or node.func.attr != "add_argument":
            continue
        for arg in node.args:
            if isinstance(arg, ast.Constant) and isinstance(arg.value, str) and arg.value.startswith("--"):
                flags.add(arg.value)
    return flags


def _emitted_flags(command: list[str]) -> set[str]:
    return {item for item in command[2:] if item.startswith("--")}


def _assert_backend_accepts_command(command: list[str]) -> None:
    accepted = _parser_flags(command[1])
    unknown = sorted(_emitted_flags(command) - accepted)
    assert unknown == []


def test_autoround_dense_mode1_matches_wrapper_defaults(tmp_path: Path):
    output_dir = tmp_path / "dense-ar"
    result = quantize_qwen35_autoround(
        model_dir="weights/Qwen3.5-9B",
        output_dir=str(output_dir),
        topology="dense",
        dry_run=True,
    )

    command = _cmd(result)
    assert result.method == "autoround"
    assert result.topology == "dense"
    assert "third_party/auto-round/scripts/quantize.py" in command[1]
    assert _flag_value(command, "--mode") == "llm-only"
    assert _flag_value(command, "--llm_bits") == "4"
    assert _flag_value(command, "--llm_group_size") == "64"
    assert _flag_value(command, "--iters") == "200"
    assert _flag_value(command, "--nsamples") == "128"
    assert _flag_value(command, "--seqlen") == "2048"
    assert _flag_value(command, "--batch_size") == "8"
    assert _flag_value(command, "--dataset") == "NeelNanda/pile-10k"
    assert _flag_value(command, "--format") == "auto_gptq"
    assert result.provenance["log_file"] == str(output_dir / "qwen35_autoround_quantize.log")
    assert "--sym" in command


def test_autoround_dense_memory_flags_are_backend_supported(tmp_path: Path):
    output_dir = tmp_path / "dense-ar-low-mem"
    result = quantize_qwen35_autoround(
        model_dir="weights/Qwen3.5-9B",
        output_dir=str(output_dir),
        topology="dense",
        low_gpu_mem_usage=True,
        gradient_accumulate_steps=2,
        dry_run=True,
    )

    command = _cmd(result)
    assert "third_party/auto-round/scripts/quantize.py" in command[1]
    assert "--low_gpu_mem_usage" in command
    assert _flag_value(command, "--gradient_accumulate_steps") == "2"
    _assert_backend_accepts_command(command)


def test_autoround_dense_rejects_moe_only_overrides(tmp_path: Path):
    with pytest.raises(ValueError, match="only supported for topology='moe'"):
        quantize_qwen35_autoround(
            model_dir="weights/Qwen3.5-9B",
            output_dir=str(tmp_path / "dense-ar-invalid"),
            topology="dense",
            attn_bits=8,
            dry_run=True,
        )


def test_autoround_moe_mode1_matches_wrapper_defaults(tmp_path: Path):
    output_dir = tmp_path / "moe-ar"
    result = quantize_qwen35_autoround(
        model_dir="weights/Qwen3.6-35B-A3B",
        output_dir=str(output_dir),
        topology="moe",
        dry_run=True,
    )

    command = _cmd(result)
    assert result.topology == "moe"
    assert "third_party/auto-round/scripts_qwen35moe/quantize_moe.py" in command[1]
    assert _flag_value(command, "--mode") == "llm-only"
    assert _flag_value(command, "--llm_bits") == "4"
    assert _flag_value(command, "--llm_group_size") == "64"
    assert _flag_value(command, "--device_map") == "0"
    assert _flag_value(command, "--batch_size") == "8"
    assert _flag_value(command, "--gradient_accumulate_steps") == "1"
    assert _flag_value(command, "--dataset") == "NeelNanda/pile-10k"
    assert _flag_value(command, "--format") == "auto_round:gptqmodel"
    assert _flag_value(command, "--attn_bits") == "8"
    assert _flag_value(command, "--shared_expert_bits") == "8"
    assert result.provenance["log_file"] == str(output_dir / "qwen35_autoround_quantize.log")
    assert "--low_gpu_mem_usage" in command
    assert "--sym" in command


def test_autoround_default_commands_are_backend_supported(tmp_path: Path):
    dense = quantize_qwen35_autoround(
        model_dir="weights/Qwen3.5-9B",
        output_dir=str(tmp_path / "dense-ar-supported"),
        topology="dense",
        dry_run=True,
    )
    moe = quantize_qwen35_autoround(
        model_dir="weights/Qwen3.6-35B-A3B",
        output_dir=str(tmp_path / "moe-ar-supported"),
        topology="moe",
        dry_run=True,
    )

    _assert_backend_accepts_command(_cmd(dense))
    _assert_backend_accepts_command(_cmd(moe))
