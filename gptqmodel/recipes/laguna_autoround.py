"""Laguna-S-2.1 QuaRot + AutoRound recipe API.

This API owns the maintained Laguna script lifecycle while giving downstream
workflows a stable keyword-only entrypoint. Routed experts use W4 while the
remaining eligible linear layers use W8 with symmetric group-size-64 weights.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any


_BACKEND = "third_party/auto-round/scripts_laguna/quantize_laguna.py"


@dataclass(frozen=True)
class LagunaAutoRoundRecipeResult:
    output_dir: str
    method: str
    artifact_format: str
    model_family: str
    rotation: str
    precision: dict[str, Any]
    provenance: dict[str, Any]


def quantize_laguna_autoround(
    *,
    model_dir: str,
    output_dir: str,
    rest_bits: int = 8,
    expert_bits: int = 4,
    group_size: int = 64,
    sym: bool = True,
    iters: int = 200,
    nsamples: int = 128,
    seqlen: int = 2048,
    batch_size: int = 1,
    gradient_accumulate_steps: int = 1,
    dataset: str = "NeelNanda/pile-10k",
    device_map: str = "0,1,2,3",
    format: str = "auto_round",
    seed: int = 42,
    rotation: str | bool | None = "hadamard",
    rotation_device: str = "cuda:0",
    rotation_headwise_v_o: bool = True,
    rotation_chunk_rows: int = 1024,
    rotation_chunk_columns: int = 1024,
    rotation_matrix_batch: int = 4,
    rotation_compute_dtype: str = "float64",
    low_gpu_mem_usage: bool = True,
    low_cpu_mem_usage: bool = True,
    quant_lm_head: bool = False,
    save_rotated: str | None = None,
    python: str | None = None,
    extra_args: Sequence[str] | None = None,
    dry_run: bool = False,
    env: Mapping[str, str] | None = None,
) -> LagunaAutoRoundRecipeResult:
    """Quantize Laguna-S-2.1 through the maintained mixed-bit backend."""

    rotation = _normalize_rotation(rotation)
    rest_bits = int(rest_bits)
    expert_bits = int(expert_bits)
    group_size = int(group_size)
    if (rest_bits, expert_bits, group_size, bool(sym)) != (8, 4, 64, True):
        raise ValueError(
            "Laguna AutoRound supports only rest W8, routed experts W4, group_size=64, and symmetric quantization"
        )
    if str(format) != "auto_round":
        raise ValueError(
            f"Laguna mixed-bit AutoRound requires format='auto_round', got {format!r}"
        )
    if str(rotation_compute_dtype) not in {"float32", "float64"}:
        raise ValueError("rotation_compute_dtype must be 'float32' or 'float64'")
    if extra_args is not None and isinstance(extra_args, (str, bytes)):
        raise TypeError("extra_args must be a sequence of command arguments")

    iters = _positive_int(iters, "iters")
    nsamples = _positive_int(nsamples, "nsamples")
    seqlen = _positive_int(seqlen, "seqlen")
    batch_size = _positive_int(batch_size, "batch_size")
    gradient_accumulate_steps = _positive_int(
        gradient_accumulate_steps,
        "gradient_accumulate_steps",
    )
    rotation_chunk_rows = _positive_int(rotation_chunk_rows, "rotation_chunk_rows")
    rotation_chunk_columns = _positive_int(
        rotation_chunk_columns, "rotation_chunk_columns"
    )
    rotation_matrix_batch = _positive_int(
        rotation_matrix_batch, "rotation_matrix_batch"
    )
    python = str(python or os.environ.get("GPTQMODEL_PYTHON") or sys.executable)

    command = _build_command(
        python=python,
        model_dir=model_dir,
        output_dir=output_dir,
        rest_bits=rest_bits,
        expert_bits=expert_bits,
        group_size=group_size,
        sym=bool(sym),
        iters=iters,
        nsamples=nsamples,
        seqlen=seqlen,
        batch_size=batch_size,
        gradient_accumulate_steps=gradient_accumulate_steps,
        dataset=dataset,
        device_map=device_map,
        format=str(format),
        seed=int(seed),
        rotation=rotation,
        rotation_device=rotation_device,
        rotation_headwise_v_o=bool(rotation_headwise_v_o),
        rotation_chunk_rows=rotation_chunk_rows,
        rotation_chunk_columns=rotation_chunk_columns,
        rotation_matrix_batch=rotation_matrix_batch,
        rotation_compute_dtype=str(rotation_compute_dtype),
        low_gpu_mem_usage=bool(low_gpu_mem_usage),
        low_cpu_mem_usage=bool(low_cpu_mem_usage),
        quant_lm_head=bool(quant_lm_head),
        save_rotated=save_rotated,
        extra_args=extra_args,
    )
    precision = {
        "routed_experts": "W4",
        "other_quantized_linears": "W8",
        "group_size": 64,
        "symmetric": True,
    }
    provenance = {
        "recipe": "gptqmodel.recipes.laguna_autoround.quantize_laguna_autoround",
        "backend": _BACKEND,
        "command": command,
        "log_file": _normalize_path(_default_log_file(output_dir)),
        "dry_run": bool(dry_run),
    }
    if not dry_run:
        run_env = _build_environment(env)
        _run_command_with_log(
            command,
            cwd=_repo_root(),
            env=run_env,
            log_file=provenance["log_file"],
        )
        _write_recipe_provenance(output_dir, provenance)

    return LagunaAutoRoundRecipeResult(
        output_dir=_normalize_path(output_dir),
        method="autoround",
        artifact_format="gptqmodel_hf",
        model_family="laguna",
        rotation=rotation,
        precision=precision,
        provenance=provenance,
    )


def _build_command(
    *,
    python: str,
    model_dir: str,
    output_dir: str,
    rest_bits: int,
    expert_bits: int,
    group_size: int,
    sym: bool,
    iters: int,
    nsamples: int,
    seqlen: int,
    batch_size: int,
    gradient_accumulate_steps: int,
    dataset: str,
    device_map: str,
    format: str,
    seed: int,
    rotation: str,
    rotation_device: str,
    rotation_headwise_v_o: bool,
    rotation_chunk_rows: int,
    rotation_chunk_columns: int,
    rotation_matrix_batch: int,
    rotation_compute_dtype: str,
    low_gpu_mem_usage: bool,
    low_cpu_mem_usage: bool,
    quant_lm_head: bool,
    save_rotated: str | None,
    extra_args: Sequence[str] | None,
) -> list[str]:
    script = _repo_root() / _BACKEND
    if not script.is_file():
        raise FileNotFoundError(f"Laguna AutoRound backend does not exist: {script}")
    command = [
        str(python),
        str(script),
        "--model",
        str(model_dir),
        "--output-dir",
        str(output_dir),
        "--expert-bits",
        str(expert_bits),
        "--rest-bits",
        str(rest_bits),
        "--group-size",
        str(group_size),
        "--iters",
        str(iters),
        "--nsamples",
        str(nsamples),
        "--seqlen",
        str(seqlen),
        "--batch-size",
        str(batch_size),
        "--gradient-accumulate-steps",
        str(gradient_accumulate_steps),
        "--dataset",
        str(dataset),
        "--device-map",
        str(device_map),
        "--format",
        str(format),
        "--seed",
        str(seed),
        "--rotation",
        str(rotation),
        "--rotation-device",
        str(rotation_device),
        "--rotation-chunk-rows",
        str(rotation_chunk_rows),
        "--rotation-chunk-columns",
        str(rotation_chunk_columns),
        "--rotation-matrix-batch",
        str(rotation_matrix_batch),
        "--rotation-compute-dtype",
        str(rotation_compute_dtype),
    ]
    command.append("--sym" if sym else "--no-sym")
    command.append(
        "--rotation-headwise-v-o"
        if rotation_headwise_v_o
        else "--no-rotation-headwise-v-o"
    )
    command.append(
        "--low-gpu-mem-usage" if low_gpu_mem_usage else "--no-low-gpu-mem-usage"
    )
    command.append(
        "--low-cpu-mem-usage" if low_cpu_mem_usage else "--no-low-cpu-mem-usage"
    )
    if quant_lm_head:
        command.append("--quant-lm-head")
    if save_rotated:
        command.extend(["--save-rotated", str(save_rotated)])
    if extra_args:
        command.extend(str(item) for item in extra_args)
    return command


def _normalize_rotation(value: str | bool | None) -> str:
    if value in (None, False):
        return "none"
    text = str(value).strip().lower().replace("-", "_")
    if text in {"", "none", "null", "false", "off"}:
        return "none"
    if text != "hadamard":
        raise ValueError(
            f"Laguna AutoRound rotation must be none or hadamard, got {value!r}"
        )
    return text


def _positive_int(value: int, name: str) -> int:
    result = int(value)
    if result <= 0:
        raise ValueError(f"{name} must be positive")
    return result


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _autoround_root() -> Path:
    return _repo_root() / "third_party" / "auto-round"


def _build_environment(overrides: Mapping[str, str] | None) -> dict[str, str]:
    run_env = os.environ.copy()
    if overrides:
        run_env.update({str(key): str(value) for key, value in overrides.items()})
    python_paths = [str(_autoround_root()), str(_repo_root())]
    if run_env.get("PYTHONPATH"):
        python_paths.append(run_env["PYTHONPATH"])
    run_env["PYTHONPATH"] = os.pathsep.join(python_paths)
    run_env.setdefault("TOKENIZERS_PARALLELISM", "false")
    return run_env


def _normalize_path(path: str | os.PathLike[str]) -> str:
    return os.path.abspath(os.path.normpath(str(path)))


def _default_log_file(output_dir: str | os.PathLike[str]) -> str:
    return str(Path(output_dir) / "laguna_autoround_quantize.log")


def _write_recipe_provenance(output_dir: str, provenance: Mapping[str, Any]) -> None:
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    (path / "laguna_autoround_recipe_provenance.json").write_text(
        json.dumps(dict(provenance), ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _run_command_with_log(
    command: Sequence[str],
    *,
    cwd: Path,
    env: Mapping[str, str],
    log_file: str | os.PathLike[str],
) -> None:
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8", errors="replace") as log:
        process = subprocess.Popen(
            list(command),
            cwd=str(cwd),
            env=dict(env),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            errors="replace",
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            log.write(line)
            log.flush()
            sys.stdout.write(line)
            sys.stdout.flush()
        return_code = process.wait()
    if return_code:
        raise subprocess.CalledProcessError(return_code, list(command))


__all__ = ["LagunaAutoRoundRecipeResult", "quantize_laguna_autoround"]
