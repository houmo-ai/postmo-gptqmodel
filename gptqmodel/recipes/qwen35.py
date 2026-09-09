"""Qwen3.5/Qwen3.6 Dense and MoE quantization recipe API.

This module is intentionally a thin API wrapper over the maintained Qwen3.5
example scripts.  It gives downstream callers such as xh2modelzoo a stable
``gptqmodel.recipes.qwen35.quantize_qwen35`` import without duplicating the
actual GPTQ/rotation/visual/MTP/MoE logic.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


_SUPPORTED_METHODS = {"gptq"}
_SUPPORTED_PRESETS = {"llm_only", "full_vlm", "full_vlm_rotate", "rotated_fp_only"}
_SUPPORTED_ROTATIONS = {None, "hadamard", "block_hadamard", "random"}
_SUPPORTED_ARTIFACT_FORMATS = {"gptqmodel_hf"}
_DENSE_CALIBRATION_JSONL = "gptqmodel/quantization/calibration/dense_ivsg/gen_data/Qwen3.5-27B.jsonl"
_MOE_CALIBRATION_JSONL = "gptqmodel/quantization/calibration/moe_ebss/gen_data/Qwen3-Next-80B-A3B-Instruct.jsonl"


@dataclass(frozen=True)
class Qwen35QuantRecipeResult:
    output_dir: str
    method: str
    artifact_format: str
    model_family: str
    topology: str
    preset: str
    rotation: str | None
    has_visual: bool
    has_mtp: bool
    provenance: dict[str, Any]


def quantize_qwen35(
    *,
    model_dir: str,
    output_dir: str,
    method: str = "gptq",
    preset: str = "full_vlm",
    rotation: str | bool | None = False,
    artifact_format: str = "gptqmodel_hf",
    topology: str | None = None,
    group_size: int = 64,
    bits: int = 4,
    sym: bool | None = None,
    iters: int | None = None,
    seed: int = 42,
    quant_nontext_module: bool | None = None,
    device: str | None = None,
    device_map: str | None = "auto",
    trust_remote_code: bool = True,
    batch_size: int = 1,
    low_gpu_mem_usage: bool | None = None,
    offload_to_disk: bool = False,
    offload_path: str | None = None,
    hessian_mse: bool = True,
    wait_for_submodule_finalizers: bool = True,
    mse: float | None = None,
    calibration_jsonl: str | None = None,
    calibration_text_key: str = "text",
    calibration_dataset: str | None = None,
    calibration_split: str | None = None,
    nsamples: int | None = None,
    seqlen: int | None = None,
    check_quant_vision_demo: bool = True,
    check_rotation_ppl: bool = False,
    max_quant_layers: int | None = None,
    moe: Mapping[str, Any] | None = None,
    self_attn_bits: int | None = None,
    shared_expert_bits: int | None = None,
    expert_bits: int | None = None,
    expert_down_bits: int | None = None,
    run_inference: bool | None = None,
    skip_inference: bool | None = None,
    dry_run: bool = False,
    extra_args: list[str] | None = None,
    env: Mapping[str, str] | None = None,
    head: Mapping[str, Any] | None = None,
) -> Qwen35QuantRecipeResult:
    """Quantize Qwen3.5/Qwen3.6 via the maintained example scripts.

    Args mirror the stable downstream recipe contract.  The current backend is
    the GPTQModel Qwen3.5 script implementation; ``method`` is retained as the
    public algorithm selector so AutoRound/GPTQ behavior can be split inside
    GPTQModel without changing xh2modelzoo.

    ``dry_run=True`` validates arguments, writes no files, and returns the exact
    command in provenance.  This is useful for lightweight integration tests and
    Gerrit command-shape checks.
    """

    if head is not None:
        options = locals().copy()
        from ..quantization.lm_head import HeadQuantizationConfig
        if HeadQuantizationConfig(**head).bits < 8:
            from .qwen35_head import quantize_with_head
            return quantize_with_head(quantize_qwen35, options, backbone_algorithm="gptq")

    method = str(method or "gptq").lower()
    preset = str(preset or "full_vlm").lower()
    rotation = _normalize_rotation(rotation)
    artifact_format = str(artifact_format or "gptqmodel_hf").lower()
    topology = _resolve_topology(model_dir=model_dir, topology=topology)

    if calibration_jsonl is None and calibration_dataset is None:
        calibration_jsonl = _DENSE_CALIBRATION_JSONL if topology == "dense" else _MOE_CALIBRATION_JSONL
    if nsamples is None:
        nsamples = 256 if topology == "dense" else 512
    if seqlen is None:
        seqlen = 1024
    if topology == "moe":
        self_attn_bits = 8 if self_attn_bits is None else self_attn_bits
        shared_expert_bits = 8 if shared_expert_bits is None else shared_expert_bits
        expert_bits = 4 if expert_bits is None else expert_bits
        expert_down_bits = 5 if expert_down_bits is None else expert_down_bits

    _validate_recipe_args(
        method=method,
        preset=preset,
        rotation=rotation,
        artifact_format=artifact_format,
        group_size=group_size,
        offload_to_disk=offload_to_disk,
    )

    if preset in {"full_vlm_rotate", "rotated_fp_only"} and not rotation:
        raise ValueError(f"preset={preset!r} requires a non-null rotation")

    command = _build_command(
        model_dir=model_dir,
        output_dir=output_dir,
        topology=topology,
        preset=preset,
        rotation=rotation,
        bits=bits,
        group_size=group_size,
        seed=seed,
        device=device,
        device_map=device_map,
        trust_remote_code=trust_remote_code,
        batch_size=batch_size,
        offload_to_disk=offload_to_disk,
        offload_path=offload_path,
        hessian_mse=hessian_mse,
        wait_for_submodule_finalizers=wait_for_submodule_finalizers,
        mse=mse,
        calibration_jsonl=calibration_jsonl,
        calibration_text_key=calibration_text_key,
        nsamples=nsamples,
        seqlen=seqlen,
        check_quant_vision_demo=check_quant_vision_demo,
        check_rotation_ppl=check_rotation_ppl,
        max_quant_layers=max_quant_layers,
        self_attn_bits=self_attn_bits,
        shared_expert_bits=shared_expert_bits,
        expert_bits=expert_bits,
        expert_down_bits=expert_down_bits,
        run_inference=run_inference,
        skip_inference=skip_inference,
        extra_args=extra_args,
    )

    provenance = {
        "recipe": "gptqmodel.recipes.qwen35.quantize_qwen35",
        "backend": "examples/quantization/examples/example_qwen35dense.py"
        if topology == "dense"
        else "examples/quantization/examples/example_qwen35moe.py",
        "command": command,
        "dry_run": bool(dry_run),
        "ignored_compat_args": {
            "sym": sym,
            "iters": iters,
            "quant_nontext_module": quant_nontext_module,
            "low_gpu_mem_usage": low_gpu_mem_usage,
            "calibration_dataset": calibration_dataset,
            "calibration_split": calibration_split,
            "moe": dict(moe or {}),
        },
    }

    if not dry_run:
        run_env = os.environ.copy()
        if env:
            run_env.update({str(key): str(value) for key, value in env.items()})
        subprocess.run(command, cwd=str(_repo_root()), env=run_env, check=True)
        _write_recipe_provenance(output_dir, provenance)

    return Qwen35QuantRecipeResult(
        output_dir=_normalize_path(output_dir),
        method=method,
        artifact_format=artifact_format,
        model_family=_infer_model_family(model_dir),
        topology=topology,
        preset=preset,
        rotation=rotation,
        has_visual=preset in {"full_vlm", "full_vlm_rotate"},
        has_mtp=preset in {"full_vlm", "full_vlm_rotate"},
        provenance=provenance,
    )


def _build_command(
    *,
    model_dir: str,
    output_dir: str,
    topology: str,
    preset: str,
    rotation: str | None,
    bits: int,
    group_size: int,
    seed: int,
    device: str | None,
    device_map: str | None,
    trust_remote_code: bool,
    batch_size: int,
    offload_to_disk: bool,
    offload_path: str | None,
    hessian_mse: bool,
    wait_for_submodule_finalizers: bool,
    mse: float | None,
    calibration_jsonl: str | None,
    calibration_text_key: str,
    nsamples: int,
    seqlen: int,
    check_quant_vision_demo: bool,
    check_rotation_ppl: bool,
    max_quant_layers: int | None,
    self_attn_bits: int | None,
    shared_expert_bits: int | None,
    expert_bits: int | None,
    expert_down_bits: int | None,
    run_inference: bool | None,
    skip_inference: bool | None,
    extra_args: list[str] | None,
) -> list[str]:
    script = _examples_dir() / ("example_qwen35dense.py" if topology == "dense" else "example_qwen35moe.py")
    command = [
        sys.executable,
        str(script),
        "--model",
        str(model_dir),
        "--out",
        str(output_dir),
        "--bits",
        str(int(bits)),
        "--group-size",
        str(int(group_size)),
        "--batch-size",
        str(int(batch_size)),
        "--nsamples",
        str(int(nsamples)),
        "--seqlen",
        str(int(seqlen)),
        "--seed",
        str(int(seed)),
    ]

    if rotation:
        command.extend(["--rotation", rotation])
    if trust_remote_code:
        command.append("--trust-remote-code")
    if device_map:
        command.extend(["--device-map", str(device_map)])
    elif device:
        command.extend(["--device", str(device)])
    if offload_to_disk:
        command.append("--offload-to-disk")
        if offload_path:
            command.extend(["--offload-path", str(offload_path)])
    else:
        command.append("--no-offload-to-disk")
    if hessian_mse:
        command.append("--hessian-mse")
    else:
        command.append("--no-hessian-mse")
    if wait_for_submodule_finalizers:
        command.append("--wait-for-submodule-finalizers")
    else:
        command.append("--no-wait-for-submodule-finalizers")
    if mse is not None and topology == "dense":
        command.extend(["--mse", str(float(mse))])
    if calibration_jsonl:
        command.extend(["--calibration-jsonl", str(calibration_jsonl)])
    if calibration_text_key:
        command.extend(["--calibration-text-key", str(calibration_text_key)])
    if check_rotation_ppl:
        command.append("--check-rotation-ppl")
    if check_quant_vision_demo:
        command.append("--check-quant-vision-demo")
    else:
        command.append("--no-check-quant-vision-demo")
    if max_quant_layers is not None:
        command.extend(["--max-quant-layers", str(int(max_quant_layers))])

    if topology == "moe":
        if self_attn_bits is not None:
            command.extend(["--self-attn-bits", str(int(self_attn_bits))])
        if shared_expert_bits is not None:
            command.extend(["--shared-expert-bits", str(int(shared_expert_bits))])
        if expert_bits is not None:
            command.extend(["--expert-bits", str(int(expert_bits))])
        if expert_down_bits is not None:
            command.extend(["--expert-down-bits", str(int(expert_down_bits))])

    if preset == "rotated_fp_only":
        command.extend(["--rotated-out", str(output_dir), "--stop-after-rotate"])

    if _should_skip_inference(run_inference=run_inference, skip_inference=skip_inference, check_quant_vision_demo=check_quant_vision_demo):
        command.append("--skip-inference")
    if extra_args:
        command.extend(str(item) for item in extra_args)
    return command


def _validate_recipe_args(
    *,
    method: str,
    preset: str,
    rotation: str | None,
    artifact_format: str,
    group_size: int,
    offload_to_disk: bool,
) -> None:
    if method not in _SUPPORTED_METHODS:
        raise ValueError(f"method must be one of {sorted(_SUPPORTED_METHODS)}, got {method!r}")
    if preset not in _SUPPORTED_PRESETS:
        raise ValueError(f"preset must be one of {sorted(_SUPPORTED_PRESETS)}, got {preset!r}")
    if rotation not in _SUPPORTED_ROTATIONS:
        raise ValueError("rotation must be one of hadamard/block_hadamard/random/null, " f"got {rotation!r}")
    if artifact_format not in _SUPPORTED_ARTIFACT_FORMATS:
        raise ValueError(f"artifact_format must be 'gptqmodel_hf', got {artifact_format!r}")
    if int(group_size) != 64:
        raise ValueError(f"Qwen3.5/Qwen3.6 group_size must be 64, got {group_size!r}")
    if rotation and offload_to_disk:
        raise ValueError("Qwen3.5/Qwen3.6 rotated quantization requires offload_to_disk=False")


def _resolve_topology(*, model_dir: str, topology: str | None) -> str:
    if topology:
        value = str(topology).lower()
        if value not in {"dense", "moe"}:
            raise ValueError(f"topology must be dense or moe, got {topology!r}")
        return value
    config_path = Path(model_dir) / "config.json"
    if config_path.is_file():
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
            text = json.dumps(config).lower()
            if "moe" in text or "num_experts" in text or "num_routed_experts" in text:
                return "moe"
        except Exception:
            pass
    return "dense"


def _normalize_rotation(rotation: str | bool | None) -> str | None:
    if rotation is None or rotation is False:
        return None
    if rotation is True:
        return "hadamard"
    value = str(rotation).strip().lower()
    if value in {"", "none", "null", "false"}:
        return None
    return value


def _should_skip_inference(*, run_inference: bool | None, skip_inference: bool | None, check_quant_vision_demo: bool) -> bool:
    if skip_inference is not None:
        return bool(skip_inference)
    if run_inference is not None:
        return not bool(run_inference)
    return not bool(check_quant_vision_demo)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _examples_dir() -> Path:
    return _repo_root() / "examples" / "quantization" / "examples"


def _normalize_path(path: str | os.PathLike[str]) -> str:
    return os.path.abspath(os.path.normpath(str(path)))


def _infer_model_family(model_dir: str) -> str:
    name = Path(model_dir).name.lower()
    if "qwen3.6" in name or "qwen36" in name:
        return "qwen3_6"
    return "qwen3_5"


def _write_recipe_provenance(output_dir: str, provenance: Mapping[str, Any]) -> None:
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    payload = dict(provenance)
    payload["command"] = list(provenance.get("command", []))
    (path / "qwen35_quant_recipe_provenance.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )


__all__ = ["Qwen35QuantRecipeResult", "quantize_qwen35"]
