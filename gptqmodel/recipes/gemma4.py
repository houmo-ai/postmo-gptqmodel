"""Gemma4 Series quantization recipe API.

This module is a thin stable wrapper over the maintained Gemma4 GPTQ example
script.  Downstream workflows such as xh2modelzoo can import
``gptqmodel.recipes.gemma4.quantize_gemma4`` without duplicating GPTQModel
loading, calibration, quantization, saving, or validation logic.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


_SUPPORTED_METHODS = {"gptq"}
_SUPPORTED_PRESETS = {"llm_only", "full_vlm", "full_multimodal"}
_SUPPORTED_ARTIFACT_FORMATS = {"gptqmodel_hf"}
_SUPPORTED_TOPOLOGIES = {"dense", "moe"}

_GEMMA4_TEXT_MODULES_IN_BLOCK_TO_QUANTIZE = [[
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]]
_GEMMA4_BLOCK_NAME_TO_QUANTIZE = "model.language_model.layers"


@dataclass(frozen=True)
class Gemma4QuantRecipeResult:
    output_dir: str
    method: str
    artifact_format: str
    model_family: str
    topology: str
    preset: str
    rotation: str | None
    variant: str
    capabilities: dict[str, bool]
    provenance: dict[str, Any]


def quantize_gemma4(
    *,
    model_dir: str,
    output_dir: str,
    method: str = "gptq",
    preset: str = "full_multimodal",
    rotation: str | None = None,
    artifact_format: str = "gptqmodel_hf",
    variant: str | None = None,
    topology: str | None = None,
    capabilities: Mapping[str, bool] | None = None,
    export_subgraphs: Mapping[str, bool] | None = None,
    context_max_length: int = 2048,
    prefill_chunk_length: int = 256,
    group_size: int = 64,
    bits: int = 4,
    sym: bool | None = None,
    iters: int | None = None,
    seed: int = 42,
    quant_nontext_module: bool | None = None,
    device: str | None = None,
    device_map: str | None = None,
    trust_remote_code: bool = True,
    batch_size: int = 1,
    low_gpu_mem_usage: bool | None = None,
    offload_to_disk: bool = False,
    offload_path: str | None = None,
    calibration_jsonl: str | None = None,
    calibration_text_key: str = "text",
    calibration_dataset: str | None = None,
    calibration_split: str | None = None,
    nsamples: int = 256,
    seqlen: int = 2048,
    check_quant_text_demo: bool = True,
    check_quant_image_demo: bool = False,
    check_quant_video_demo: bool = False,
    check_quant_audio_demo: bool = False,
    check_rotation_ppl: bool = False,
    max_quant_layers: int | None = None,
    moe: Mapping[str, Any] | None = None,
    self_attn_bits: int | None = None,
    shared_expert_bits: int | None = None,
    expert_bits: int | None = None,
    expert_down_bits: int | None = None,
    audio: Mapping[str, Any] | None = None,
    audio_max_duration_seconds: int | None = None,
    run_inference: bool | None = None,
    skip_inference: bool | None = None,
    dry_run: bool = False,
    extra_args: list[str] | None = None,
    env: Mapping[str, str] | None = None,
) -> Gemma4QuantRecipeResult:
    """Quantize Gemma4 via the maintained GPTQ example script.

    ``dry_run=True`` validates arguments and returns the exact subprocess
    command in provenance without writing model artifacts.  The current Gemma4
    backend is the checked-in GPTQ script; AutoRound and rotation are not
    claimed here because the existing Gemma4 recipe path does not implement
    them.
    """

    method = str(method or "gptq").lower()
    preset = _normalize_preset(preset)
    rotation = _normalize_rotation(rotation)
    artifact_format = str(artifact_format or "gptqmodel_hf").lower()
    topology = _resolve_topology(model_dir=model_dir, topology=topology)
    variant = str(variant or _infer_variant(model_dir=model_dir)).lower()
    caps = _normalize_capabilities(capabilities, preset=preset)

    _validate_recipe_args(
        method=method,
        preset=preset,
        rotation=rotation,
        artifact_format=artifact_format,
        topology=topology,
        group_size=group_size,
        check_rotation_ppl=check_rotation_ppl,
    )

    command = _build_command(
        model_dir=model_dir,
        output_dir=output_dir,
        bits=bits,
        group_size=group_size,
        seed=seed,
        device=device,
        device_map=device_map,
        trust_remote_code=trust_remote_code,
        batch_size=batch_size,
        offload_to_disk=offload_to_disk,
        offload_path=offload_path,
        calibration_jsonl=calibration_jsonl,
        calibration_text_key=calibration_text_key,
        nsamples=nsamples,
        seqlen=seqlen,
        run_inference=run_inference,
        skip_inference=skip_inference,
        check_quant_text_demo=check_quant_text_demo,
        moe=moe,
        extra_args=extra_args,
    )

    provenance = {
        "recipe": "gptqmodel.recipes.gemma4.quantize_gemma4",
        "backend": "examples/quantization/examples/example_gemma4.py",
        "command": command,
        "dry_run": bool(dry_run),
        "context_max_length": int(context_max_length),
        "prefill_chunk_length": int(prefill_chunk_length),
        "export_subgraphs": dict(export_subgraphs or {}),
        "moe": dict(moe or {}),
        "ignored_compat_args": {
            "sym": sym,
            "iters": iters,
            "quant_nontext_module": quant_nontext_module,
            "low_gpu_mem_usage": low_gpu_mem_usage,
            "calibration_dataset": calibration_dataset,
            "calibration_split": calibration_split,
            "check_quant_image_demo": check_quant_image_demo,
            "check_quant_video_demo": check_quant_video_demo,
            "check_quant_audio_demo": check_quant_audio_demo,
            "max_quant_layers": max_quant_layers,
            "self_attn_bits": self_attn_bits,
            "shared_expert_bits": shared_expert_bits,
            "expert_bits": expert_bits,
            "expert_down_bits": expert_down_bits,
            "audio": dict(audio or {}),
            "audio_max_duration_seconds": audio_max_duration_seconds,
        },
    }

    if not dry_run:
        run_env = os.environ.copy()
        if env:
            run_env.update({str(key): str(value) for key, value in env.items()})
        subprocess.run(command, cwd=str(_repo_root()), env=run_env, check=True)
        _copy_hf_auxiliary_files(model_dir, output_dir)
        _restore_source_config_fields(model_dir, output_dir)
        _patch_gemma4_quantization_config(output_dir)
        _write_recipe_provenance(output_dir, provenance)

    return Gemma4QuantRecipeResult(
        output_dir=_normalize_path(output_dir),
        method=method,
        artifact_format=artifact_format,
        model_family="gemma4",
        topology=topology,
        preset=preset,
        rotation=rotation,
        variant=variant,
        capabilities=caps,
        provenance=provenance,
    )


def _build_command(
    *,
    model_dir: str,
    output_dir: str,
    bits: int,
    group_size: int,
    seed: int,
    device: str | None,
    device_map: str | None,
    trust_remote_code: bool,
    batch_size: int,
    offload_to_disk: bool,
    offload_path: str | None,
    calibration_jsonl: str | None,
    calibration_text_key: str,
    nsamples: int,
    seqlen: int,
    run_inference: bool | None,
    skip_inference: bool | None,
    check_quant_text_demo: bool,
    moe: Mapping[str, Any] | None,
    extra_args: list[str] | None,
) -> list[str]:
    command = [
        sys.executable,
        str(_examples_dir() / "example_gemma4.py"),
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
    if trust_remote_code:
        command.append("--trust-remote-code")
    if device_map:
        command.extend(["--device-map", str(device_map)])
    elif device:
        command.extend(["--device", str(device), "--infer-device", str(device)])
    if offload_to_disk:
        command.append("--offload-to-disk")
        if offload_path:
            command.extend(["--offload-path", str(offload_path)])
    else:
        command.append("--no-offload-to-disk")
    if calibration_jsonl:
        command.extend(["--calibration-jsonl", str(calibration_jsonl)])
    if calibration_text_key:
        command.extend(["--calibration-text-key", str(calibration_text_key)])
    if _should_skip_eval(
        run_inference=run_inference,
        skip_inference=skip_inference,
        check_quant_text_demo=check_quant_text_demo,
    ):
        command.append("--skip-eval")
    command.extend(_build_moe_routing_args(moe))
    if extra_args:
        command.extend(str(item) for item in extra_args)
    return command


def _build_moe_routing_args(moe: Mapping[str, Any] | None) -> list[str]:
    if not moe:
        return []

    routing = moe.get("routing", moe.get("moe_routing"))
    if routing is None:
        return []
    if isinstance(routing, bool):
        routing = "bypass" if routing else "none"
    routing = str(routing).strip().lower().replace("-", "_")
    if routing in {"", "none", "null", "false"}:
        return ["--moe-routing", "none"]
    if routing not in {"bypass", "override"}:
        raise ValueError(f"Unsupported Gemma4 MoE routing mode: {routing!r}")

    args = ["--moe-routing", routing]
    batch_size = (
        moe.get("routing_batch_size")
        or moe.get("moe_routing_batch_size")
        or moe.get("batch_size")
    )
    if batch_size is not None:
        args.extend(["--moe-routing-batch-size", str(int(batch_size))])
    experts_per_tok = moe.get("num_experts_per_tok", moe.get("moe_num_experts_per_tok"))
    if routing == "override" and experts_per_tok is not None:
        args.extend(["--moe-num-experts-per-tok", str(experts_per_tok)])
    return args


def _validate_recipe_args(
    *,
    method: str,
    preset: str,
    rotation: str | None,
    artifact_format: str,
    topology: str,
    group_size: int,
    check_rotation_ppl: bool,
) -> None:
    if method not in _SUPPORTED_METHODS:
        raise ValueError(f"Gemma4 recipe method must be 'gptq', got {method!r}")
    if preset not in _SUPPORTED_PRESETS:
        raise ValueError(f"Gemma4 recipe preset must be one of {sorted(_SUPPORTED_PRESETS)}, got {preset!r}")
    if rotation is not None:
        raise ValueError("Gemma4 GPTQ recipe does not support rotation; set quant.rotation to null")
    if check_rotation_ppl:
        raise ValueError("Gemma4 GPTQ recipe does not support rotation PPL checks")
    if artifact_format not in _SUPPORTED_ARTIFACT_FORMATS:
        raise ValueError(f"artifact_format must be 'gptqmodel_hf', got {artifact_format!r}")
    if topology not in _SUPPORTED_TOPOLOGIES:
        raise ValueError(f"topology must be dense or moe, got {topology!r}")
    if int(group_size) != 64:
        raise ValueError(f"Gemma4 group_size must be 64, got {group_size!r}")


def _normalize_preset(preset: str | None) -> str:
    value = str(preset or "full_multimodal").lower()
    if value == "full_vlm":
        return "full_multimodal"
    return value


def _normalize_rotation(rotation: str | None) -> str | None:
    if rotation is None:
        return None
    value = str(rotation).strip().lower()
    if value in {"", "none", "null", "false"}:
        return None
    return value


def _normalize_capabilities(capabilities: Mapping[str, bool] | None, *, preset: str) -> dict[str, bool]:
    if capabilities:
        return {str(key): bool(value) for key, value in capabilities.items()}
    multimodal = preset == "full_multimodal"
    return {"text": True, "image": multimodal, "video": multimodal, "audio": False}


def _resolve_topology(*, model_dir: str, topology: str | None) -> str:
    if topology:
        value = str(topology).lower()
        if value not in _SUPPORTED_TOPOLOGIES:
            raise ValueError(f"topology must be dense or moe, got {topology!r}")
        return value
    config = _load_hf_config(model_dir)
    text_config = config.get("text_config", {}) if isinstance(config, Mapping) else {}
    return "moe" if isinstance(text_config, Mapping) and text_config.get("enable_moe_block") else "dense"


def _infer_variant(*, model_dir: str) -> str:
    config = _load_hf_config(model_dir)
    text_config = config.get("text_config", {}) if isinstance(config, Mapping) else {}
    if config.get("model_type") == "gemma4_unified":
        return "12b_unified"
    if isinstance(text_config, Mapping) and text_config.get("enable_moe_block"):
        return "26b_a4b"
    if isinstance(config.get("audio_config"), Mapping):
        return "e4b"
    name = Path(model_dir).name.lower()
    if "e4b" in name:
        return "e4b"
    if "26b" in name or "a4b" in name:
        return "26b_a4b"
    if "31b" in name:
        return "31b"
    return "unknown"


def _load_hf_config(model_dir: str) -> dict[str, Any]:
    config_path = Path(model_dir) / "config.json"
    if not config_path.is_file():
        return {}
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return config if isinstance(config, dict) else {}


def _should_skip_eval(*, run_inference: bool | None, skip_inference: bool | None, check_quant_text_demo: bool) -> bool:
    if skip_inference is not None:
        return bool(skip_inference)
    if run_inference is not None:
        return not bool(run_inference)
    return not bool(check_quant_text_demo)


def _copy_hf_auxiliary_files(model_dir: str, output_dir: str) -> None:
    """Copy processor/tokenizer sidecar files that GPTQModel.save may omit.

    Multimodal Gemma4 export reloads the quantized directory as a normal HF
    artifact.  Keep the quantized config/weights from GPTQModel, but bring over
    source-side processor files such as processor_config.json so image/video/audio
    preprocessing can be reconstructed from the quant directory alone.
    """

    src = Path(model_dir)
    dst = Path(output_dir)
    if not src.is_dir() or not dst.is_dir():
        return
    excluded_suffixes = (".safetensors", ".bin", ".pt", ".pth", ".onnx")
    excluded_names = {
        "config.json",
        "generation_config.json",
        "model.safetensors.index.json",
        "pytorch_model.bin.index.json",
        "quantize_config.json",
    }
    for path in src.iterdir():
        if not path.is_file():
            continue
        if path.name in excluded_names or path.name.startswith("model-") or path.name.startswith("pytorch_model-"):
            continue
        if path.suffix in excluded_suffixes:
            continue
        target = dst / path.name
        if target.exists():
            continue
        shutil.copy2(path, target)


def _restore_source_config_fields(model_dir: str, output_dir: str) -> None:
    """Keep checkpoint-specific config fields that HF save may omit.

    Some custom multimodal configs do not declare every serialized checkpoint
    field as a constructor parameter. ``save_pretrained`` then drops fields
    such as Unified vision ``model_patch_size`` and audio
    ``audio_samples_per_token``. Restore only those known source-only fields;
    all other serialized values remain owned by the quantized output.
    """

    source_path = Path(model_dir) / "config.json"
    output_path = Path(output_dir) / "config.json"
    if not source_path.is_file() or not output_path.is_file():
        return

    try:
        source_config = json.loads(source_path.read_text(encoding="utf-8"))
        output_config = json.loads(output_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return
    if not isinstance(source_config, dict) or not isinstance(output_config, dict):
        return

    source_only_fields = {
        "vision_config": ("model_patch_size",),
        "audio_config": ("audio_samples_per_token",),
    }
    changed = False
    for section_name, field_names in source_only_fields.items():
        source_section = source_config.get(section_name)
        output_section = output_config.get(section_name)
        if not isinstance(source_section, Mapping):
            continue
        if output_section is None:
            output_section = {}
            output_config[section_name] = output_section
        if not isinstance(output_section, dict):
            continue
        for field_name in field_names:
            if field_name in output_section or field_name not in source_section:
                continue
            output_section[field_name] = source_section[field_name]
            changed = True

    if changed:
        output_path.write_text(
            json.dumps(output_config, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )


def _patch_gemma4_quantization_config(output_dir: str) -> None:
    """Keep HF GPTQ reload from converting Gemma4 E4B per-layer PLE linears.

    GPTQModel intentionally quantizes only the text block projection/MLP
    linears.  Gemma4 E4B also has per_layer_input_gate and
    per_layer_projection inside each text block; they remain BF16 in the saved
    checkpoint.  Transformers/optimum re-creates GPTQ linears for every Linear
    in the block unless modules_in_block_to_quantize is persisted, which turns
    those BF16 PLE weights into missing/uninitialized qlinear buffers.
    """

    root = Path(output_dir)
    for config_name in ("config.json", "quantize_config.json"):
        config_path = root / config_name
        if not config_path.is_file():
            continue
        try:
            payload = json.loads(config_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        quant_config = payload.get("quantization_config") if config_name == "config.json" else payload
        if not isinstance(quant_config, dict):
            continue
        if str(quant_config.get("quant_method", "")).lower() != "gptq":
            continue
        quant_config["block_name_to_quantize"] = _GEMMA4_BLOCK_NAME_TO_QUANTIZE
        quant_config["modules_in_block_to_quantize"] = _GEMMA4_TEXT_MODULES_IN_BLOCK_TO_QUANTIZE
        config_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _examples_dir() -> Path:
    return _repo_root() / "examples" / "quantization" / "examples"


def _normalize_path(path: str | os.PathLike[str]) -> str:
    return os.path.abspath(os.path.normpath(str(path)))


def _write_recipe_provenance(output_dir: str, provenance: Mapping[str, Any]) -> None:
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    payload = dict(provenance)
    payload["command"] = list(provenance.get("command", []))
    (path / "gemma4_quant_recipe_provenance.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )


__all__ = ["Gemma4QuantRecipeResult", "quantize_gemma4"]
