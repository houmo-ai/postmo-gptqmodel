import json
from pathlib import Path

import pytest

from gptqmodel.recipes.qwen3_next import quantize_qwen3_next


def _flag_value(command: list[str], flag: str) -> str:
    assert flag in command
    return command[command.index(flag) + 1]


def _write_checkpoint_index(model_dir: Path, tensor_names: list[str]) -> None:
    model_dir.mkdir()
    (model_dir / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {name: "model-00001.safetensors" for name in tensor_names}}),
        encoding="utf-8",
    )


def test_first_four_layers_are_exposed_in_command(tmp_path: Path):
    model_dir = tmp_path / "model"
    _write_checkpoint_index(model_dir, ["model.layers.0.weight", "mtp.layers.0.weight"])
    result = quantize_qwen3_next(
        model_dir=str(model_dir),
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


def test_target_only_checkpoint_is_reported_without_mtp(tmp_path: Path):
    model_dir = tmp_path / "model"
    _write_checkpoint_index(model_dir, ["model.layers.0.weight"])

    result = quantize_qwen3_next(
        model_dir=str(model_dir),
        output_dir=str(tmp_path / "out"),
        dry_run=True,
    )

    assert result.has_mtp is False


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


def test_unsupported_method_is_rejected(tmp_path: Path):
    with pytest.raises(ValueError, match="method must be 'gptq' or 'autoround'"):
        quantize_qwen3_next(
            model_dir="weights/Qwen3-Next-80B-A3B-Instruct",
            output_dir=str(tmp_path / "out"),
            method="awq",
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


@pytest.mark.parametrize("dtype", ["int_sym_xh", "int_sym", "int_sym_gptq"])
def test_autoround_recipe_builds_a_real_supported_command(tmp_path, dtype):
    from gptqmodel.recipes.qwen3_next_autoround import build_parser

    result = quantize_qwen3_next(
        model_dir=str(tmp_path / "model"), output_dir=str(tmp_path / "out"),
        method="autoround", data_type=dtype, device_map="0", dry_run=True,
    )
    command = result.provenance["command"]
    assert Path(command[1]).name == "example_qwen3_next_autoround.py"
    assert Path(command[1]).is_file()
    args = build_parser().parse_args(command[2:])
    assert args.data_type == dtype
    assert (args.nsamples, args.iters, args.batch_size, args.gradient_accumulate_steps, args.seqlen) == (256, 400, 8, 1, 2048)
    assert (args.bits, args.group_size, args.expert_bits, args.self_attn_bits, args.shared_expert_bits) == (4, 64, 4, 8, 8)
    assert args.sym is True
    assert args.deterministic is False
    assert args.dataset == "NeelNanda/pile-10k"
    assert args.format == "auto_round:gptqmodel"
    assert result.method == "autoround"
    assert not (tmp_path / "out").exists()


def test_gptq_defaults_are_unchanged(tmp_path):
    result = quantize_qwen3_next(model_dir="raw", output_dir=str(tmp_path / "out"), dry_run=True)
    command = result.provenance["command"]
    assert [_flag_value(command, flag) for flag in ("--batch-size", "--nsamples", "--seqlen")] == ["1", "256", "1024"]
    assert _flag_value(command, "--device-map") == "auto"
    assert "--data-type" not in command


def test_autoround_overrides_paths_and_determinism(tmp_path, monkeypatch):
    from gptqmodel.recipes.qwen3_next_autoround import build_parser

    monkeypatch.chdir(tmp_path)
    result = quantize_qwen3_next(
        model_dir="raw", output_dir="quant", method="autoround",
        calibration_jsonl="samples.jsonl", nsamples=256, iters=400,
        batch_size=1, gradient_accumulate_steps=8, seed=7, deterministic=True,
        low_gpu_mem_usage=False, trust_remote_code=False,
        self_attn_bits=16, expert_bits=8, device_map="cpu", dry_run=True,
    )
    args = build_parser().parse_args(result.provenance["command"][2:])
    assert args.model == str(tmp_path / "raw")
    assert args.out == str(tmp_path / "quant")
    assert args.calibration_jsonl == str(tmp_path / "samples.jsonl")
    assert (args.nsamples, args.iters, args.batch_size, args.gradient_accumulate_steps, args.seed) == (256, 400, 1, 8, 7)
    assert args.deterministic is True
    assert args.low_gpu_mem_usage is False
    assert args.trust_remote_code is False
    assert (args.self_attn_bits, args.expert_bits, args.device_map) == (16, 8, "cpu")


@pytest.mark.parametrize("kwargs,match", [
    ({"sym": False}, "requires sym=True"),
    ({"nsamples": 0}, "nsamples must be positive"),
    ({"iters": -1}, "iters must be >= 0"),
    ({"max_quant_layers": 1}, "GPTQ layer-limit/inference/validation"),
    ({"check_saved_precision": True}, "GPTQ layer-limit/inference/validation"),
    ({"extra_args": ["--iters", "1"]}, "extra_args are unsupported"),
])
def test_autoround_rejects_invalid_or_unused_options(tmp_path, kwargs, match):
    with pytest.raises(ValueError, match=match):
        quantize_qwen3_next(model_dir="raw", output_dir=str(tmp_path / "out"),
                           method="autoround", dry_run=True, **kwargs)


def test_autoround_refuses_to_overwrite_source_or_existing_artifact(tmp_path):
    with pytest.raises(ValueError, match="must differ"):
        quantize_qwen3_next(model_dir=str(tmp_path), output_dir=str(tmp_path), method="autoround", dry_run=True)
    out = tmp_path / "out"
    out.mkdir()
    checkpoint = out / "config.json"
    checkpoint.write_text("{}")
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        quantize_qwen3_next(model_dir="raw", output_dir=str(out), method="autoround")
    assert checkpoint.read_text() == "{}"
