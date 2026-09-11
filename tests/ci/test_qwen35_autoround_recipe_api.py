from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

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


def _parser_default(script_path: str, flag: str) -> str:
    tree = ast.parse(Path(script_path).read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not isinstance(node.func, ast.Attribute) or node.func.attr != "add_argument":
            continue
        if not any(isinstance(arg, ast.Constant) and arg.value == flag for arg in node.args):
            continue
        for keyword in node.keywords:
            if keyword.arg == "default" and isinstance(keyword.value, ast.Constant):
                return str(keyword.value.value)
    raise AssertionError(f"{flag} has no constant default in {script_path}")


def _parser_set_default(script_path: str, name: str) -> str:
    tree = ast.parse(Path(script_path).read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not isinstance(node.func, ast.Attribute) or node.func.attr != "set_defaults":
            continue
        for keyword in node.keywords:
            if keyword.arg == name and isinstance(keyword.value, ast.Constant):
                return str(keyword.value.value)
    raise AssertionError(f"{name} has no constant parser default in {script_path}")


def _autoround_data_type_values(script_path: str) -> list[str]:
    tree = ast.parse(Path(script_path).read_text(encoding="utf-8"))
    values: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name) or node.func.id != "AutoRound":
            continue
        for keyword in node.keywords:
            if keyword.arg == "data_type":
                values.append(ast.unparse(keyword.value))
                break
        else:
            raise AssertionError(f"AutoRound call at line {node.lineno} does not pass data_type")
    return values


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
    assert _flag_value(command, "--data_type") == "int_sym_gptq"
    assert _flag_value(command, "--iters") == "400"
    assert _flag_value(command, "--nsamples") == "256"
    assert _flag_value(command, "--seqlen") == "2048"
    assert _flag_value(command, "--batch_size") == "8"
    assert _flag_value(command, "--dataset") == "NeelNanda/pile-10k"
    assert _flag_value(command, "--format") == "auto_gptq"
    assert result.provenance["log_file"] == str(output_dir / "qwen35_autoround_quantize.log")
    assert result.provenance["data_type"] == "int_sym_gptq"
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
    assert _flag_value(command, "--data_type") == "int_sym_gptq"
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


def test_autoround_data_type_can_restore_signed_full_range(tmp_path: Path):
    result = quantize_qwen35_autoround(
        model_dir="weights/Qwen3.5-9B",
        output_dir=str(tmp_path / "dense-ar-signed"),
        topology="dense",
        data_type="int",
        dry_run=True,
    )

    assert _flag_value(_cmd(result), "--data_type") == "int"
    assert result.provenance["data_type"] == "int"


def test_autoround_int_sym_xh_is_forwarded_and_recorded(tmp_path: Path):
    result = quantize_qwen35_autoround(
        model_dir="weights/Qwen3.6-35B-A3B",
        output_dir=str(tmp_path / "moe-ar-int-sym-xh"),
        topology="moe",
        data_type="int_sym_xh",
        dry_run=True,
    )

    assert _flag_value(_cmd(result), "--data_type") == "int_sym_xh"
    assert result.provenance["data_type"] == "int_sym_xh"
    _assert_backend_accepts_command(_cmd(result))


def test_autoround_asymmetric_mode_requires_compatible_data_type(tmp_path: Path):
    with pytest.raises(ValueError, match="requires sym=True"):
        quantize_qwen35_autoround(
            model_dir="weights/Qwen3.5-9B",
            output_dir=str(tmp_path / "dense-ar-invalid-asym"),
            topology="dense",
            sym=False,
            dry_run=True,
        )

    result = quantize_qwen35_autoround(
        model_dir="weights/Qwen3.5-9B",
        output_dir=str(tmp_path / "dense-ar-asym"),
        topology="dense",
        sym=False,
        data_type="int",
        dry_run=True,
    )
    command = _cmd(result)
    assert "--asym" in command
    assert "--sym" not in command
    _assert_backend_accepts_command(command)


def test_autoround_int_sym_xh_rejects_asymmetric_mode(tmp_path: Path):
    with pytest.raises(ValueError, match="int_sym_xh.*requires sym=True"):
        quantize_qwen35_autoround(
            model_dir="weights/Qwen3.5-9B",
            output_dir=str(tmp_path / "dense-ar-xh-asym"),
            topology="dense",
            sym=False,
            data_type="int_sym_xh",
            dry_run=True,
        )


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


@pytest.mark.parametrize(
    "script_path",
    [
        "third_party/auto-round/scripts/quantize.py",
        "third_party/auto-round/scripts_qwen35moe/quantize_moe.py",
    ],
)
def test_autoround_backends_default_and_forward_positive_scale_data_type(script_path: str):
    assert _parser_default(script_path, "--data_type") == "int_sym_gptq"
    assert _parser_set_default(script_path, "sym") == "True"
    assert "--asym" in _parser_flags(script_path)
    assert _autoround_data_type_values(script_path) == ["args.data_type"] * 3


def test_int_sym_gptq_scale_is_always_positive():
    torch = pytest.importorskip("torch")
    from auto_round.data_type.int import quant_tensor_sym_gptq

    weight = torch.tensor(
        [
            [-1.0, 2.0, 0.5, -0.25],
            [-2.0, 1.0, -0.5, 0.25],
            [0.0, 0.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )
    _, scale, zero_point = quant_tensor_sym_gptq(
        weight,
        bits=4,
        group_size=4,
        v=torch.tensor(0.2),
        min_scale=torch.tensor(0.3),
        max_scale=torch.tensor(0.7),
        scale_dtype=torch.float32,
    )

    assert torch.isfinite(scale).all()
    assert (scale > 0).all()
    assert (zero_point == 8).all()


def test_int_sym_xh_uses_positive_hardware_grid_and_gptq_midpoint():
    torch = pytest.importorskip("torch")
    from auto_round.data_type import get_quant_func
    from auto_round.data_type.int import quant_tensor_sym_xh

    weight = torch.tensor(
        [
            [-1.0, 2.0, 0.5, -0.25],
            [-2.0, 1.0, -0.5, 0.25],
            [0.0, 0.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )
    qdq, scale, zero_point = quant_tensor_sym_xh(
        weight,
        bits=4,
        group_size=4,
        v=0,
        min_scale=torch.tensor(1.0),
        max_scale=torch.tensor(1.0),
        scale_dtype=torch.float32,
    )

    expected_scale = torch.tensor([[2.0 / 7.0], [2.0 / 8.0], [1.0e-5]])
    assert torch.allclose(scale, expected_scale)
    assert (scale > 0).all()
    assert (zero_point == 8).all()

    signed_codes = torch.round(qdq / scale)
    packed_codes = signed_codes + zero_point
    assert signed_codes.min() >= -8
    assert signed_codes.max() <= 7
    assert packed_codes.min() >= 0
    assert packed_codes.max() <= 15
    assert torch.equal(scale * (packed_codes - zero_point), qdq)

    quant_func, registered_name = get_quant_func("int_sym_xh", bits=4, sym=True, disable_opt_rtn=True)
    assert quant_func is quant_tensor_sym_xh
    assert registered_name == "int_sym_xh"
    with pytest.raises(ValueError, match="requires symmetric quantization"):
        get_quant_func("int_sym_xh", bits=4, sym=False, disable_opt_rtn=True)


def test_int_sym_xh_preserves_autoround_minmax_and_rounding_gradients():
    torch = pytest.importorskip("torch")
    from auto_round.data_type.int import quant_tensor_sym_xh

    weight = torch.tensor(
        [[-1.0, 2.0, 0.5, -0.25], [-2.0, 1.0, -0.5, 0.25]],
        dtype=torch.float32,
    )
    min_scale = torch.tensor(1.0, requires_grad=True)
    max_scale = torch.tensor(1.0, requires_grad=True)
    rounding = torch.zeros_like(weight, requires_grad=True)

    qdq, _, _ = quant_tensor_sym_xh(
        weight,
        bits=4,
        group_size=4,
        v=rounding,
        min_scale=min_scale,
        max_scale=max_scale,
        scale_dtype=torch.float32,
    )
    qdq.square().sum().backward()

    assert min_scale.grad is not None and torch.isfinite(min_scale.grad)
    assert max_scale.grad is not None and torch.isfinite(max_scale.grad)
    assert rounding.grad is not None and torch.isfinite(rounding.grad).all()


@pytest.mark.parametrize("data_type", ["int_sym_gptq", "int_sym_xh"])
def test_symmetric_gptq_dtypes_skip_layers_that_cannot_be_packed(data_type):
    torch = pytest.importorskip("torch")
    from auto_round.formats import _check_divisible_by_32
    from auto_round.schemes import QuantizationScheme

    autoround = SimpleNamespace(
        scheme=QuantizationScheme(
            bits=4,
            group_size=64,
            sym=True,
            data_type=data_type,
            act_bits=16,
        ),
        model=torch.nn.Sequential(torch.nn.Linear(64, 1, bias=False)),
        supported_types=(torch.nn.Linear,),
        inner_supported_types=(),
        layer_config=None,
    )

    _check_divisible_by_32(autoround)

    assert autoround.layer_config["0"]["bits"] == 16
    assert autoround.layer_config["0"]["data_type"] == "fp"
    assert autoround.layer_config["0"]["fixed_by_user"] is True
