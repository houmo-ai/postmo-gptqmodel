from __future__ import annotations

from pathlib import Path

from gptqmodel.recipes.qwen35_autoround import quantize_qwen35_autoround


def _cmd(result):
    return result.provenance["command"]


def _flag_value(command: list[str], flag: str) -> str:
    assert flag in command
    return command[command.index(flag) + 1]


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
