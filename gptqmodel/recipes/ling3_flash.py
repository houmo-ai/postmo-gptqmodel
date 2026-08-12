"""Stable GPTQ recipe API for Ling-3-Flash checkpoints.

The recipe owns the Ling-specific W8/W4 profile, calibration tokenization,
MoE routing, GPTQModel load/quantize/save lifecycle, and the conservative
``offload_to_disk=False`` default. Downstream projects should call
``quantize_ling3_flash`` instead of rebuilding ``QuantizeConfig`` themselves.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping


_EXPERT_PATTERN = r".*\.mlp\.experts\.\d+\.(gate_proj|up_proj|down_proj)$"
_QUANTIZATION_ATTENTION_BACKEND = "eager"


@dataclass(frozen=True)
class Ling3FlashQuantRecipeResult:
    output_dir: str
    method: str
    artifact_format: str
    model_family: str
    offload_to_disk: bool
    precision: dict[str, Any]
    calibration: dict[str, Any]
    provenance: dict[str, Any]


def build_ling3_flash_dynamic(
    *,
    num_layers: int | None = None,
    expert_bits: int = 4,
    group_size: int = 64,
    max_quant_layers: int | None = None,
) -> dict[str, dict[str, Any]]:
    """Build routed-expert W4 overrides and trailing block exclusions."""

    dynamic: dict[str, dict[str, Any]] = {
        _EXPERT_PATTERN: {
            "bits": int(expert_bits),
            "group_size": int(group_size),
            "sym": True,
        }
    }
    if max_quant_layers is None:
        return dynamic
    if num_layers is None:
        raise ValueError("num_layers is required when max_quant_layers is set")
    if not 0 <= int(max_quant_layers) <= int(num_layers):
        raise ValueError(f"max_quant_layers must be in [0, {num_layers}]")
    return {
        **{
            rf"-:.*\.layers\.{index}\..*": {}
            for index in range(int(max_quant_layers), int(num_layers))
        },
        **dynamic,
    }


def tokenize_ling3_flash_calibration(
    texts: list[str],
    *,
    tokenizer,
    seqlen: int,
) -> list[dict[str, list[int]]]:
    """Tokenize and hard-truncate calibration rows to the recipe sequence length."""

    encoded = tokenizer(
        texts,
        add_special_tokens=True,
        truncation=True,
        max_length=int(seqlen),
        padding=False,
    )
    input_ids = encoded["input_ids"]
    attention_masks = encoded.get("attention_mask")
    if attention_masks is None:
        attention_masks = [[1] * len(row) for row in input_ids]
    if len(input_ids) != len(texts) or len(attention_masks) != len(texts):
        raise RuntimeError("Tokenizer returned a different number of calibration rows")

    calibration: list[dict[str, list[int]]] = []
    for row_index, (row_ids, row_mask) in enumerate(
        zip(input_ids, attention_masks, strict=True)
    ):
        if len(row_ids) > seqlen:
            raise RuntimeError(
                f"Tokenizer did not truncate calibration row {row_index} to {seqlen}"
            )
        if len(row_ids) != len(row_mask):
            raise RuntimeError(
                f"Calibration row {row_index} has mismatched ids and attention mask"
            )
        calibration.append(
            {"input_ids": list(row_ids), "attention_mask": list(row_mask)}
        )
    return calibration


def quantize_ling3_flash(
    *,
    model_dir: str,
    output_dir: str,
    method: str = "gptq",
    artifact_format: str = "gptqmodel_hf",
    bits: int = 8,
    expert_bits: int = 4,
    group_size: int = 64,
    sym: bool = True,
    batch_size: int = 1,
    nsamples: int = 128,
    seqlen: int = 2048,
    calibration_jsonl: str | None = None,
    calibration_text_key: str = "text",
    calibration_dataset: str | None = "NeelNanda/pile-10k",
    calibration_data: Iterable[str | Mapping[str, Any]] | None = None,
    device: str = "cuda:0",
    trust_remote_code: bool = True,
    offload_to_disk: bool = False,
    offload_path: str | None = None,
    hessian_mse: bool = True,
    wait_for_submodule_finalizers: bool = True,
    moe_routing_batch_size: int | None = None,
    auto_forward_data_parallel: bool = False,
    max_quant_layers: int | None = None,
    seed: int = 42,
    dry_run: bool = False,
) -> Ling3FlashQuantRecipeResult:
    """Quantize Ling-3-Flash with routed experts W4 and other linears W8.

    Disk offload is intentionally opt-in. Ling follows the same hybrid
    attention/MoE family as Qwen3.5, whose offload path has previously produced
    invalid quantized checkpoints. Callers must explicitly set
    ``offload_to_disk=True`` when investigating that path.
    """

    method = str(method).lower().replace("-", "_")
    artifact_format = str(artifact_format).lower()
    _validate_recipe_args(
        method=method,
        artifact_format=artifact_format,
        bits=bits,
        expert_bits=expert_bits,
        group_size=group_size,
        sym=sym,
        batch_size=batch_size,
        nsamples=nsamples,
        seqlen=seqlen,
        moe_routing_batch_size=moe_routing_batch_size,
        max_quant_layers=max_quant_layers,
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
        offload_path=offload_path,
    )
    decoder_layers: int | None = None
    total_layers: int | None = None
    config_path = model_path / "config.json"
    if config_path.is_file():
        decoder_layers, total_layers = _read_ling_layer_counts(model_path)
    elif max_quant_layers is not None:
        # Preserve the previous actionable error for explicit partial runs.
        _read_ling_layer_counts(model_path)

    effective_max_quant_layers = max_quant_layers
    if decoder_layers is not None:
        if effective_max_quant_layers is None:
            effective_max_quant_layers = decoder_layers
        elif int(effective_max_quant_layers) > decoder_layers:
            raise ValueError(
                "max_quant_layers cannot include Ling MTP blocks: "
                f"got {effective_max_quant_layers}, decoder_layers={decoder_layers}"
            )
    dynamic = build_ling3_flash_dynamic(
        num_layers=total_layers,
        expert_bits=expert_bits,
        group_size=group_size,
        max_quant_layers=effective_max_quant_layers,
    )
    precision = {
        "routed_experts": f"W{expert_bits}",
        "all_other_quantized_linears": f"W{bits}",
        "kda_decay_and_beta": "BF16",
        "mtp_layers": "BF16",
        "group_size": int(group_size),
        "symmetric": True,
    }
    calibration = {
        "jsonl": None if calibration_path is None else str(calibration_path),
        "dataset": calibration_dataset,
        "text_key": str(calibration_text_key),
        "nsamples": int(nsamples),
        "seqlen": int(seqlen),
    }
    provenance = {
        "recipe": "gptqmodel.recipes.ling3_flash.quantize_ling3_flash",
        "backend": "direct_gptqmodel_api",
        "dry_run": bool(dry_run),
        "offload_to_disk": bool(offload_to_disk),
        "offload_path": resolved_offload_path,
        "wait_for_submodule_finalizers": bool(wait_for_submodule_finalizers),
        "hessian_mse": bool(hessian_mse),
        "moe_routing": "bypass",
        "moe_routing_batch_size": moe_routing_batch_size,
        "auto_forward_data_parallel": bool(auto_forward_data_parallel),
        "attention_backend": _QUANTIZATION_ATTENTION_BACKEND,
        "max_quant_layers": max_quant_layers,
        "effective_max_quant_layers": effective_max_quant_layers,
        "decoder_layers": decoder_layers,
        "mtp_layers": (
            None
            if decoder_layers is None or total_layers is None
            else total_layers - decoder_layers
        ),
        "dynamic": dynamic,
    }
    result = Ling3FlashQuantRecipeResult(
        output_dir=str(output_path),
        method=method,
        artifact_format=artifact_format,
        model_family="ling_3_flash",
        offload_to_disk=bool(offload_to_disk),
        precision=precision,
        calibration=calibration,
        provenance=provenance,
    )
    if dry_run:
        return result

    if output_path.exists() and any(output_path.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output directory: {output_path}")

    import torch
    from transformers import AutoTokenizer

    from gptqmodel import GPTQModel, QuantizeConfig
    from gptqmodel.models.definitions.ling_3_flash import (
        patch_ling_3_flash_remote_code,
    )
    from gptqmodel.quantization.config import ExpertsRoutingBypass, MoEConfig

    random.seed(seed)
    torch.manual_seed(seed)
    texts = _load_calibration_texts(
        calibration_data=calibration_data,
        calibration_jsonl=calibration_path,
        calibration_dataset=calibration_dataset,
        calibration_text_key=calibration_text_key,
        nsamples=nsamples,
    )
    patch_ling_3_flash_remote_code()
    quantize_config = QuantizeConfig(
        bits=int(bits),
        dynamic=dynamic,
        group_size=int(group_size),
        sym=True,
        desc_act=False,
        device=str(device),
        hessian_mse=bool(hessian_mse),
        wait_for_submodule_finalizers=bool(wait_for_submodule_finalizers),
        moe=MoEConfig(
            routing=ExpertsRoutingBypass(batch_size=moe_routing_batch_size)
        ),
        offload_to_disk=bool(offload_to_disk),
        offload_to_disk_path=resolved_offload_path,
        auto_forward_data_parallel=bool(auto_forward_data_parallel),
    )
    model = GPTQModel.load(
        str(model_path),
        quantize_config=quantize_config,
        trust_remote_code=bool(trust_remote_code),
        attn_implementation=_QUANTIZATION_ATTENTION_BACKEND,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path),
        trust_remote_code=bool(trust_remote_code),
    )
    tokenized = tokenize_ling3_flash_calibration(
        texts,
        tokenizer=tokenizer,
        seqlen=seqlen,
    )
    patch_ling_3_flash_remote_code()
    model.quantize(tokenized, batch_size=int(batch_size))
    model.save(str(output_path))
    _write_provenance(output_path, result)
    return result


def _validate_recipe_args(
    *,
    method: str,
    artifact_format: str,
    bits: int,
    expert_bits: int,
    group_size: int,
    sym: bool,
    batch_size: int,
    nsamples: int,
    seqlen: int,
    moe_routing_batch_size: int | None,
    max_quant_layers: int | None,
) -> None:
    if method != "gptq":
        raise ValueError("Ling-3-Flash recipe method must be 'gptq'")
    if artifact_format != "gptqmodel_hf":
        raise ValueError("artifact_format must be 'gptqmodel_hf'")
    if (int(bits), int(expert_bits), int(group_size)) != (8, 4, 64):
        raise ValueError(
            "Ling-3-Flash profile requires base W8, routed experts W4, and group_size=64"
        )
    if sym is not True:
        raise ValueError("Ling-3-Flash profile requires symmetric quantization")
    if int(batch_size) <= 0 or int(nsamples) <= 0 or int(seqlen) <= 0:
        raise ValueError("batch_size, nsamples, and seqlen must be positive")
    if moe_routing_batch_size is not None and int(moe_routing_batch_size) <= 0:
        raise ValueError("moe_routing_batch_size must be positive when specified")
    if max_quant_layers is not None and int(max_quant_layers) < 0:
        raise ValueError("max_quant_layers must be non-negative")


def _resolve_offload_path(
    *,
    output_path: Path,
    offload_to_disk: bool,
    offload_path: str | None,
) -> str | None:
    if not offload_to_disk:
        return None
    path = (
        Path(offload_path).expanduser()
        if offload_path
        else output_path.parent / f"{output_path.name}_offload"
    )
    return str(path.resolve())


def _read_ling_layer_counts(model_path: Path) -> tuple[int, int]:
    config_path = model_path / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(
            f"config.json is required for max_quant_layers: {config_path}"
        )
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    text_config = payload.get("text_config") or {}
    value = payload.get("num_hidden_layers", text_config.get("num_hidden_layers"))
    if value is None:
        raise ValueError(f"num_hidden_layers is missing from {config_path}")
    decoder_layers = int(value)
    mtp_layers = int(
        payload.get(
            "num_nextn_predict_layers",
            text_config.get("num_nextn_predict_layers", 0),
        )
    )
    if decoder_layers <= 0 or mtp_layers < 0:
        raise ValueError(
            f"Invalid Ling layer counts in {config_path}: "
            f"decoder_layers={decoder_layers}, mtp_layers={mtp_layers}"
        )
    return decoder_layers, decoder_layers + mtp_layers


def _load_calibration_texts(
    *,
    calibration_data: Iterable[str | Mapping[str, Any]] | None,
    calibration_jsonl: Path | None,
    calibration_dataset: str | None,
    calibration_text_key: str,
    nsamples: int,
) -> list[str]:
    texts: list[str] = []
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
        if not calibration_dataset:
            raise ValueError(
                "Provide calibration_data, calibration_jsonl, or calibration_dataset"
            )
        from datasets import load_dataset

        source = load_dataset(calibration_dataset, split="train")

    for row_index, record in enumerate(source, 1):
        text = record if isinstance(record, str) else record.get(calibration_text_key)
        if not isinstance(text, str) or not text.strip():
            raise ValueError(
                f"Invalid calibration field {calibration_text_key!r} at row {row_index}"
            )
        texts.append(text)
        if len(texts) >= nsamples:
            break
    if len(texts) < nsamples:
        raise RuntimeError(
            f"Calibration source yielded {len(texts)} samples, need {nsamples}"
        )
    return texts


def _write_provenance(
    output_path: Path,
    result: Ling3FlashQuantRecipeResult,
) -> None:
    output_path.mkdir(parents=True, exist_ok=True)
    payload = {
        "method": result.method,
        "artifact_format": result.artifact_format,
        "model_family": result.model_family,
        "offload_to_disk": result.offload_to_disk,
        "precision": result.precision,
        "calibration": result.calibration,
        "provenance": result.provenance,
    }
    (output_path / "ling3_flash_quant_recipe_provenance.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )


__all__ = [
    "Ling3FlashQuantRecipeResult",
    "build_ling3_flash_dynamic",
    "quantize_ling3_flash",
    "tokenize_ling3_flash_calibration",
]
