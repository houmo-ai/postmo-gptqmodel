"""Qwen3.5/Qwen3.6 AutoRound mode1 recipe API.

This API preserves the maintained AutoRound wrapper behavior while giving
xh2modelzoo a stable Python entrypoint. It intentionally builds the same
command shape as the direct scripts in dry-run mode; full execution is still
owned by GPTQModel/AutoRound, not xh2modelzoo.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class Qwen35AutoRoundRecipeResult:
    output_dir: str
    method: str
    artifact_format: str
    topology: str
    mode: str
    provenance: dict[str, Any]


def quantize_qwen35_autoround(
    *,
    model_dir: str,
    output_dir: str,
    topology: str,
    mode: str = "llm-only",
    bits: int = 4,
    group_size: int = 64,
    sym: bool = True,
    data_type: str = "int_sym_gptq",
    iters: int = 400,
    nsamples: int = 256,
    seqlen: int = 2048,
    batch_size: int | None = None,
    dataset: str = "NeelNanda/pile-10k",
    device: str = "cuda:0",
    device_map: str | None = None,
    low_gpu_mem_usage: bool | None = None,
    gradient_accumulate_steps: int | None = None,
    attn_bits: int | None = None,
    shared_expert_bits: int | None = None,
    expert_bits: int | None = None,
    expert_up_gate_bits: int | None = None,
    expert_down_bits: int | None = None,
    format: str | None = None,
    seed: int = 42,
    deterministic: bool = False,
    save_rotated: str | None = None,
    extra_args: Sequence[str] | None = None,
    dry_run: bool = False,
    env: Mapping[str, str] | None = None,
    head: Mapping[str, Any] | None = None,
    auto_mixed_precision: bool | Mapping[str, Any] | None = None,
    mtp: Mapping[str, Any] | None = None,
    mtp_replay_device: str | None = None,
    optimization_mode: str = "standard",
    xh_hessian_config: Mapping[str, Any] | None = None,
) -> Qwen35AutoRoundRecipeResult:
    if data_type == "int_sym_xh_hessian":
        from auto_round.xh_hessian import resolve_xh_data_type
        optimization_mode = resolve_xh_data_type(data_type, optimization_mode)[1]
    if optimization_mode not in {"standard", "xh_hessian"}:
        raise ValueError(f"Unknown optimization_mode: {optimization_mode!r}")
    if optimization_mode == "xh_hessian":
        topology = _normalize_topology(topology)
        mode = str(mode or "llm-only").replace("_", "-")
        if topology != "dense" or mode != "llm-only" or data_type not in {"int_sym_xh", "int_sym_xh_hessian"} or bits != 4 or group_size != 64 or not sym or iters <= 0:
            raise ValueError("xh_hessian requires dense llm-only int_sym_xh W4/G64 and iters > 0")
        if mtp is not None or head is not None or auto_mixed_precision not in (None, False):
            raise ValueError("xh_hessian v1 excludes MTP, head and mixed-precision extensions")
        from auto_round.xh_hessian import XHHessianConfig
        from dataclasses import asdict
        xh_hessian_config = asdict(XHHessianConfig(**dict(xh_hessian_config or {})))
        if nsamples - xh_hessian_config["validation_samples"] < (batch_size or 8) * (gradient_accumulate_steps or 1):
            raise ValueError("xh_hessian training split must contain a full optimization batch")
    elif xh_hessian_config is not None:
        raise ValueError("xh_hessian_config requires optimization_mode='xh_hessian'")
    if mtp is not None:
        options = locals().copy()
        from .qwen35_mtp import quantize_with_mtp
        return quantize_with_mtp(quantize_qwen35_autoround, options, backbone_algorithm="autoround")
    if head is not None:
        options = locals().copy()
        from ..quantization.lm_head import HeadQuantizationConfig
        if HeadQuantizationConfig(**head).bits < 8:
            from .qwen35_head import quantize_with_head
            return quantize_with_head(quantize_qwen35_autoround, options, backbone_algorithm="autoround")
    topology = _normalize_topology(topology)
    mode = str(mode or "llm-only").replace("_", "-")
    if mode != "llm-only":
        raise ValueError("Qwen3.5 AutoRound API currently exposes mode='llm-only' only")
    if int(group_size) != 64:
        raise ValueError(f"Qwen3.5 AutoRound group_size must be 64, got {group_size!r}")
    data_type = str(data_type)
    if data_type in {"int_sym_gptq", "int_sym_xh"} and not sym:
        raise ValueError(
            f"data_type={data_type!r} requires sym=True; use data_type='int' for asymmetric quantization"
        )

    if topology == "moe":
        batch_size = 8 if batch_size is None else batch_size
        gradient_accumulate_steps = 1 if gradient_accumulate_steps is None else gradient_accumulate_steps
        device_map = "0" if device_map is None else device_map
        low_gpu_mem_usage = True if low_gpu_mem_usage is None else low_gpu_mem_usage
        attn_bits = 8 if attn_bits is None else attn_bits
        shared_expert_bits = 8 if shared_expert_bits is None else shared_expert_bits
        format = format or "auto_round:gptqmodel"
    else:
        batch_size = 8 if batch_size is None else batch_size
        format = format or "auto_gptq"

    if topology == "dense":
        _reject_dense_moe_overrides(
            attn_bits=attn_bits,
            shared_expert_bits=shared_expert_bits,
            expert_bits=expert_bits,
            expert_up_gate_bits=expert_up_gate_bits,
            expert_down_bits=expert_down_bits,
        )

    search_options = None
    if auto_mixed_precision is not None and auto_mixed_precision is not False:
        from gptqmodel.quantization.auto_mixed_precision import recipe_search_options, validate_search_paths
        search_options = recipe_search_options(
            auto_mixed_precision, topology=topology, rotation=bool(save_rotated),
            incompatible=not sym or data_type not in {"int_sym_xh", "int_sym_gptq"}
            or format != "auto_round:gptqmodel")

        if search_options is not None:
            validate_search_paths(model_dir, output_dir)
            if search_options["device"] is None:
                search_options["device"] = device or "cuda:0"

    command = _build_command(
        model_dir=model_dir,
        output_dir=output_dir,
        topology=topology,
        mode=mode,
        bits=bits,
        group_size=group_size,
        sym=sym,
        data_type=data_type,
        iters=iters,
        nsamples=nsamples,
        seqlen=seqlen,
        batch_size=batch_size,
        dataset=dataset,
        device=device,
        device_map=device_map,
        low_gpu_mem_usage=low_gpu_mem_usage,
        gradient_accumulate_steps=gradient_accumulate_steps,
        attn_bits=attn_bits,
        shared_expert_bits=shared_expert_bits,
        expert_bits=expert_bits,
        expert_up_gate_bits=expert_up_gate_bits,
        expert_down_bits=expert_down_bits,
        format=format,
        seed=seed,
        deterministic=deterministic,
        save_rotated=save_rotated,
        extra_args=extra_args,
    )
    if optimization_mode == "xh_hessian":
        command.extend(["--optimization_mode", optimization_mode, "--xh_hessian_config", json.dumps(xh_hessian_config)])
    if search_options is not None:
        command.extend(["--auto-mixed-precision", json.dumps(search_options)])

    provenance = {
        "recipe": "gptqmodel.recipes.qwen35_autoround.quantize_qwen35_autoround",
        "backend": "third_party/auto-round/scripts/quantize.py"
        if topology == "dense"
        else "third_party/auto-round/scripts_qwen35moe/quantize_moe.py",
        "command": command,
        "data_type": str(data_type),
        "log_file": _normalize_path(_default_log_file(output_dir)),
        "dry_run": bool(dry_run),
    }
    if optimization_mode == "xh_hessian":
        provenance.update(optimization_mode=optimization_mode, xh_hessian_config=xh_hessian_config)
        provenance["storage_data_type"] = "int_sym_xh"
    if not dry_run:
        run_env = os.environ.copy()
        if env:
            run_env.update({str(k): str(v) for k, v in env.items()})
        _run_command_with_log(command, cwd=_repo_root(), env=run_env, log_file=provenance["log_file"])
        if search_options is not None:
            from gptqmodel.quantization.auto_mixed_precision import read_precision_plan
            provenance["auto_mixed_precision"] = read_precision_plan(output_dir)
        _write_recipe_provenance(output_dir, provenance)
    return Qwen35AutoRoundRecipeResult(
        output_dir=_normalize_path(output_dir),
        method="autoround",
        artifact_format="gptqmodel_hf",
        topology=topology,
        mode=mode,
        provenance=provenance,
    )


def _reject_dense_moe_overrides(**values: int | None) -> None:
    provided = [name for name, value in values.items() if value is not None]
    if provided:
        joined = ", ".join(provided)
        raise ValueError(f"{joined} are only supported for topology='moe'")


def _build_command(
    *,
    model_dir: str,
    output_dir: str,
    topology: str,
    mode: str,
    bits: int,
    group_size: int,
    sym: bool,
    data_type: str,
    iters: int,
    nsamples: int,
    seqlen: int,
    batch_size: int,
    dataset: str,
    device: str,
    device_map: str | None,
    low_gpu_mem_usage: bool | None,
    gradient_accumulate_steps: int | None,
    attn_bits: int | None,
    shared_expert_bits: int | None,
    expert_bits: int | None,
    expert_up_gate_bits: int | None,
    expert_down_bits: int | None,
    format: str,
    seed: int,
    deterministic: bool,
    save_rotated: str | None,
    extra_args: Sequence[str] | None,
) -> list[str]:
    script = _autoround_root() / (
        "scripts/quantize.py" if topology == "dense" else "scripts_qwen35moe/quantize_moe.py"
    )
    command = [
        sys.executable,
        str(script),
        "--model",
        str(model_dir),
        "--mode",
        mode,
        "--output_dir",
        str(output_dir),
        "--llm_bits",
        str(int(bits)),
        "--llm_group_size",
        str(int(group_size)),
        "--data_type",
        str(data_type),
        "--seed",
        str(int(seed)),
        "--iters",
        str(int(iters)),
        "--nsamples",
        str(int(nsamples)),
        "--seqlen",
        str(int(seqlen)),
        "--batch_size",
        str(int(batch_size)),
        "--dataset",
        str(dataset),
        "--format",
        str(format),
    ]
    if topology == "moe" and device_map:
        command.extend(["--device_map", str(device_map)])
    elif topology == "dense" and device:
        command.extend(["--device", str(device)])
    command.append("--sym" if sym else "--asym")
    if low_gpu_mem_usage:
        command.append("--low_gpu_mem_usage")
    if gradient_accumulate_steps is not None:
        command.extend(["--gradient_accumulate_steps", str(int(gradient_accumulate_steps))])
    if topology == "moe":
        if attn_bits is not None:
            command.extend(["--attn_bits", str(int(attn_bits))])
        if shared_expert_bits is not None:
            command.extend(["--shared_expert_bits", str(int(shared_expert_bits))])
        if expert_bits is not None:
            command.extend(["--expert_bits", str(int(expert_bits))])
        if expert_up_gate_bits is not None:
            command.extend(["--expert_up_gate_bits", str(int(expert_up_gate_bits))])
        if expert_down_bits is not None:
            command.extend(["--expert_down_bits", str(int(expert_down_bits))])
    if deterministic:
        command.append("--deterministic")
    if save_rotated:
        command.extend(["--save_rotated", str(save_rotated)])
    if extra_args:
        command.extend(str(item) for item in extra_args)
    return command


def _normalize_topology(topology: str) -> str:
    value = str(topology).lower()
    if value not in {"dense", "moe"}:
        raise ValueError(f"topology must be dense or moe, got {topology!r}")
    return value


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _autoround_root() -> Path:
    return _repo_root() / "third_party" / "auto-round"


def _normalize_path(path: str | os.PathLike[str]) -> str:
    return os.path.abspath(os.path.normpath(str(path)))


def _write_recipe_provenance(output_dir: str, provenance: Mapping[str, Any]) -> None:
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    (path / "qwen35_autoround_recipe_provenance.json").write_text(
        json.dumps(dict(provenance), ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _default_log_file(output_dir: str | os.PathLike[str]) -> str:
    return str(Path(output_dir) / "qwen35_autoround_quantize.log")


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


__all__ = ["Qwen35AutoRoundRecipeResult", "quantize_qwen35_autoround"]
