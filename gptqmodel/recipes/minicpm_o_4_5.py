"""Stable GPTQ recipe for MiniCPM-o-4.5's Qwen3 language backbone.

MiniCPM-o-4.5 wraps an unmodified ``Qwen3ForCausalLM`` as ``self.llm``.
``MiniCPMO45QModel`` extracts that submodule into a standalone Qwen3 GPTQ
checkpoint. This recipe owns the GPTQ lifecycle so downstream projects such as
xh2modelzoo only resolve repository resources and translate workflow config,
rather than duplicating GPTQModel loading, calibration, quantization, or
saving.
"""

from __future__ import annotations

import json
import random
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class MiniCPMO45QuantRecipeResult:
    output_dir: str
    method: str
    artifact_format: str
    model_family: str
    precision: dict[str, Any]
    calibration: dict[str, Any]
    provenance: dict[str, Any]


def quantize_minicpm_o_4_5(
    *,
    model_dir: str,
    output_dir: str,
    method: str = "gptq",
    artifact_format: str = "gptqmodel_hf",
    bits: int = 4,
    group_size: int = 64,
    sym: bool = True,
    batch_size: int = 1,
    nsamples: int = 128,
    seqlen: int = 1024,
    calibration_jsonl: str | None = None,
    calibration_text_key: str = "text",
    calibration_data: Iterable[str | Mapping[str, Any]] | None = None,
    damp_percent: float = 0.01,
    device: str = "cuda:0",
    trust_remote_code: bool = True,
    offload_to_disk: bool = True,
    offload_to_disk_path: str | None = None,
    seed: int = 42,
    dry_run: bool = False,
) -> MiniCPMO45QuantRecipeResult:
    """Quantize the MiniCPM-o-4.5 LLM backbone with standard GPTQModel.

    Disk offload defaults to enabled to match the historical MiniCPM-o-4.5
    workflow. The caller may provide ``calibration_data`` directly or point to
    a JSONL calibration file through ``calibration_jsonl``.
    """

    method = str(method).lower().replace("-", "_")
    artifact_format = str(artifact_format).lower()
    _validate_recipe_args(
        method=method,
        artifact_format=artifact_format,
        bits=bits,
        group_size=group_size,
        batch_size=batch_size,
        nsamples=nsamples,
        seqlen=seqlen,
        damp_percent=damp_percent,
    )

    model_path = Path(model_dir).expanduser().resolve()
    output_path = Path(output_dir).expanduser().resolve()
    calibration_path = (
        None
        if calibration_jsonl in (None, "")
        else Path(str(calibration_jsonl)).expanduser().resolve()
    )
    resolved_offload_path = _resolve_offload_path(
        output_path=output_path,
        offload_to_disk=offload_to_disk,
        offload_to_disk_path=offload_to_disk_path,
    )

    precision = {
        "backbone": "qwen3",
        "bits": int(bits),
        "group_size": int(group_size),
        "symmetric": bool(sym),
        "desc_act": False,
        "damp_percent": float(damp_percent),
    }
    calibration = {
        "jsonl": None if calibration_path is None else str(calibration_path),
        "text_key": str(calibration_text_key),
        "nsamples": int(nsamples),
        "seqlen": int(seqlen),
    }
    provenance = {
        "recipe": "gptqmodel.recipes.minicpm_o_4_5.quantize_minicpm_o_4_5",
        "backend": "direct_gptqmodel_api",
        "dry_run": bool(dry_run),
        "offload_to_disk": bool(offload_to_disk),
        "offload_to_disk_path": resolved_offload_path,
        "trust_remote_code": bool(trust_remote_code),
        "batch_size": int(batch_size),
        "seed": int(seed),
    }
    result = MiniCPMO45QuantRecipeResult(
        output_dir=str(output_path),
        method=method,
        artifact_format=artifact_format,
        model_family="minicpm_o_4_5",
        precision=precision,
        calibration=calibration,
        provenance=provenance,
    )
    if dry_run:
        return result

    if output_path.exists() and any(output_path.iterdir()):
        raise FileExistsError(
            f"Refusing to overwrite non-empty output directory: {output_path}"
        )

    import torch

    from gptqmodel import GPTQModel, QuantizeConfig

    random.seed(seed)
    torch.manual_seed(seed)
    texts = _load_calibration_texts(
        calibration_data=calibration_data,
        calibration_jsonl=calibration_path,
        calibration_text_key=calibration_text_key,
        nsamples=nsamples,
    )

    quantize_config = QuantizeConfig(
        bits=int(bits),
        group_size=int(group_size),
        sym=bool(sym),
        desc_act=False,
        damp_percent=float(damp_percent),
        device=str(device),
        offload_to_disk=bool(offload_to_disk),
        offload_to_disk_path=resolved_offload_path,
    )
    model = GPTQModel.load(
        str(model_path),
        quantize_config=quantize_config,
        trust_remote_code=bool(trust_remote_code),
    )
    model.quantize(
        texts,
        batch_size=int(batch_size),
        calibration_concat_size=int(seqlen),
    )
    model.save(str(output_path))
    _write_provenance(output_path, result)
    return result


def _validate_recipe_args(
    *,
    method: str,
    artifact_format: str,
    bits: int,
    group_size: int,
    batch_size: int,
    nsamples: int,
    seqlen: int,
    damp_percent: float,
) -> None:
    if method != "gptq":
        raise ValueError("MiniCPM-o-4.5 recipe method must be 'gptq'")
    if artifact_format != "gptqmodel_hf":
        raise ValueError("artifact_format must be 'gptqmodel_hf'")
    if int(bits) <= 0:
        raise ValueError("bits must be positive")
    if int(group_size) <= 0:
        raise ValueError("group_size must be positive")
    if int(batch_size) <= 0:
        raise ValueError("batch_size must be positive")
    if int(nsamples) <= 0:
        raise ValueError("nsamples must be positive")
    if int(seqlen) <= 0:
        raise ValueError("seqlen must be positive")
    if float(damp_percent) < 0:
        raise ValueError("damp_percent must be non-negative")


def _resolve_offload_path(
    *,
    output_path: Path,
    offload_to_disk: bool,
    offload_to_disk_path: str | None,
) -> str | None:
    if not offload_to_disk:
        return None
    path = (
        Path(offload_to_disk_path).expanduser()
        if offload_to_disk_path
        else output_path.parent / f"{output_path.name}_offload"
    )
    return str(path.resolve())


def _load_calibration_texts(
    *,
    calibration_data: Iterable[str | Mapping[str, Any]] | None,
    calibration_jsonl: Path | None,
    calibration_text_key: str,
    nsamples: int,
) -> list[str]:
    if calibration_data is not None:
        source = calibration_data
    elif calibration_jsonl is not None:
        if not calibration_jsonl.is_file():
            raise FileNotFoundError(f"Calibration JSONL not found: {calibration_jsonl}")

        def _jsonl_records():
            with calibration_jsonl.open(encoding="utf-8") as stream:
                for line in stream:
                    yield json.loads(line)

        source = _jsonl_records()
    else:
        raise ValueError(
            "Provide calibration_data or calibration_jsonl for MiniCPM-o-4.5 GPTQ"
        )

    texts: list[str] = []
    for row_index, record in enumerate(source, 1):
        text = record if isinstance(record, str) else record.get(calibration_text_key)
        if not isinstance(text, str) or not text.strip():
            raise ValueError(
                f"Invalid calibration field {calibration_text_key!r} at row {row_index}"
            )
        texts.append(text)
        if len(texts) >= int(nsamples):
            break
    if len(texts) < int(nsamples):
        raise RuntimeError(
            f"Calibration source yielded {len(texts)} samples, need {int(nsamples)}"
        )
    return texts


def _write_provenance(
    output_path: Path,
    result: MiniCPMO45QuantRecipeResult,
) -> None:
    output_path.mkdir(parents=True, exist_ok=True)
    payload = {
        "method": result.method,
        "artifact_format": result.artifact_format,
        "model_family": result.model_family,
        "precision": result.precision,
        "calibration": result.calibration,
        "provenance": result.provenance,
    }
    (output_path / "minicpm_o_4_5_quant_recipe_provenance.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )


__all__ = [
    "MiniCPMO45QuantRecipeResult",
    "quantize_minicpm_o_4_5",
]
