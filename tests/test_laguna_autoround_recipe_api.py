from __future__ import annotations

import ast
import json
import os
from pathlib import Path

import pytest

from gptqmodel.recipes import laguna_autoround as recipe_module
from gptqmodel.recipes.laguna_autoround import quantize_laguna_autoround


def _command(result) -> list[str]:
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
        boolean_optional = any(
            keyword.arg == "action"
            and isinstance(keyword.value, ast.Attribute)
            and keyword.value.attr == "BooleanOptionalAction"
            for keyword in node.keywords
        )
        for arg in node.args:
            if (
                isinstance(arg, ast.Constant)
                and isinstance(arg.value, str)
                and arg.value.startswith("--")
            ):
                flags.add(arg.value)
                if boolean_optional:
                    flags.add(f"--no-{arg.value.removeprefix('--')}")
    return flags


def _emitted_flags(command: list[str]) -> set[str]:
    return {item for item in command[2:] if item.startswith("--")}


def test_laguna_autoround_defaults_match_validated_backend_profile(tmp_path: Path):
    output_dir = tmp_path / "laguna-ar"
    result = quantize_laguna_autoround(
        model_dir="weights/Laguna-S-2.1",
        output_dir=str(output_dir),
        dry_run=True,
    )

    command = _command(result)
    assert result.method == "autoround"
    assert result.artifact_format == "gptqmodel_hf"
    assert result.model_family == "laguna"
    assert result.rotation == "hadamard"
    assert result.precision == {
        "routed_experts": "W4",
        "other_quantized_linears": "W8",
        "group_size": 64,
        "symmetric": True,
    }
    assert "third_party/auto-round/scripts_laguna/quantize_laguna.py" in command[1]
    assert _flag_value(command, "--model") == "weights/Laguna-S-2.1"
    assert _flag_value(command, "--output-dir") == str(output_dir)
    assert _flag_value(command, "--expert-bits") == "4"
    assert _flag_value(command, "--rest-bits") == "8"
    assert _flag_value(command, "--group-size") == "64"
    assert _flag_value(command, "--rotation") == "hadamard"
    assert _flag_value(command, "--format") == "auto_round"
    assert "--sym" in command
    assert result.provenance["log_file"] == str(
        output_dir / "laguna_autoround_quantize.log"
    )
    assert sorted(_emitted_flags(command) - _parser_flags(command[1])) == []
    assert not output_dir.exists()


def test_laguna_autoround_forwards_model_specific_runtime_options(tmp_path: Path):
    result = quantize_laguna_autoround(
        model_dir="weights/Laguna-S-2.1",
        output_dir=str(tmp_path / "laguna-ar"),
        dataset="calibration.jsonl",
        python="/opt/gptqmodel/bin/python",
        device_map="0,1,2,3",
        rotation_device="cuda:4",
        rotation_headwise_v_o=False,
        low_gpu_mem_usage=False,
        low_cpu_mem_usage=False,
        quant_lm_head=True,
        save_rotated="work_dirs/laguna-rotated",
        extra_args=["--inspect-only"],
        dry_run=True,
    )

    command = _command(result)
    assert command[0] == "/opt/gptqmodel/bin/python"
    assert _flag_value(command, "--dataset") == "calibration.jsonl"
    assert _flag_value(command, "--device-map") == "0,1,2,3"
    assert _flag_value(command, "--rotation-device") == "cuda:4"
    assert "--no-rotation-headwise-v-o" in command
    assert "--no-low-gpu-mem-usage" in command
    assert "--no-low-cpu-mem-usage" in command
    assert "--quant-lm-head" in command
    assert _flag_value(command, "--save-rotated") == "work_dirs/laguna-rotated"
    assert "--inspect-only" in command
    assert sorted(_emitted_flags(command) - _parser_flags(command[1])) == []


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"rest_bits": 4}, "rest W8"),
        ({"expert_bits": 8}, "experts W4"),
        ({"group_size": 128}, "group_size=64"),
        ({"sym": False}, "symmetric"),
        ({"format": "auto_gptq"}, "format='auto_round'"),
        ({"rotation": "random"}, "none.*hadamard"),
    ],
)
def test_laguna_autoround_rejects_unvalidated_profiles(
    tmp_path: Path,
    kwargs: dict,
    message: str,
):
    with pytest.raises(ValueError, match=message):
        quantize_laguna_autoround(
            model_dir="weights/Laguna-S-2.1",
            output_dir=str(tmp_path / "laguna-ar"),
            dry_run=True,
            **kwargs,
        )


def test_laguna_autoround_recipe_owns_execution_and_provenance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    captured = {}

    def _run(command, *, cwd, env, log_file):
        captured.update(command=command, cwd=cwd, env=env, log_file=log_file)

    monkeypatch.setattr(recipe_module, "_run_command_with_log", _run)
    output_dir = tmp_path / "laguna-ar"
    result = quantize_laguna_autoround(
        model_dir="weights/Laguna-S-2.1",
        output_dir=str(output_dir),
        env={"LAGUNA_RECIPE_TEST": "1"},
    )

    assert captured["cwd"] == recipe_module._repo_root()
    assert captured["log_file"] == str(output_dir / "laguna_autoround_quantize.log")
    assert captured["env"]["LAGUNA_RECIPE_TEST"] == "1"
    python_paths = captured["env"]["PYTHONPATH"].split(os.pathsep)
    assert str(recipe_module._autoround_root()) in python_paths
    assert str(recipe_module._repo_root()) in python_paths
    provenance_path = output_dir / "laguna_autoround_recipe_provenance.json"
    assert json.loads(provenance_path.read_text(encoding="utf-8")) == result.provenance
