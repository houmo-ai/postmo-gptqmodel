"""Stable GPTQ and AutoRound recipe for Xing4.0 MLA+MoE.

Xing4.0-29B-A4B is a remote-code MLA+MoE checkpoint with dense early decoder
layers, routed ModuleList experts, shared experts, and HyperConnection
residuals. This recipe owns the GPTQModel lifecycle (loading, calibration,
quantization, and saving) so downstream projects such as xh2modelzoo only
resolve repository resources and translate workflow config, rather than
duplicating GPTQModel internals.
"""

from __future__ import annotations

import json
import random
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

_SUPPORTED_METHODS = {"gptq", "auto_round"}
_SELF_ATTN_PATTERN = (
    r"model\.layers\.\d+\.self_attn\."
    r"(q_a_proj|q_b_proj|kv_a_proj_with_mqa|kv_b_proj|o_proj)$"
)
_DENSE_MLP_PATTERN = r"model\.layers\.\d+\.mlp\.(gate_proj|up_proj|down_proj)$"
_EXPERT_PATTERN = (
    r"model\.layers\.\d+\.mlp\.experts\.\d+\.(gate_proj|up_proj|down_proj)$"
)
_SHARED_EXPERT_PATTERN = (
    r"model\.layers\.\d+\.mlp\.shared_experts\.(gate_proj|up_proj|down_proj)$"
)


@dataclass(frozen=True)
class Xing4_0QuantRecipeResult:
    output_dir: str
    method: str
    artifact_format: str
    model_family: str
    precision: dict[str, Any]
    calibration: dict[str, Any]
    provenance: dict[str, Any]


def quantize_xing4_0(
    *,
    model_dir: str,
    output_dir: str,
    method: str = "gptq",
    artifact_format: str = "gptqmodel_hf",
    bits: int = 4,
    self_attn_bits: int = 8,
    dense_mlp_bits: int = 8,
    shared_expert_bits: int = 8,
    expert_bits: int = 4,
    group_size: int = 64,
    sym: bool = True,
    batch_size: int = 1,
    nsamples: int = 128,
    seqlen: int = 1024,
    calibration_jsonl: str | None = None,
    calibration_text_key: str = "text",
    calibration_dataset: str | None = "wikitext",
    calibration_dataset_config: str | None = "wikitext-2-raw-v1",
    calibration_split: str = "train",
    calibration_cache_dir: str | None = None,
    calibration_data: Iterable[str | Mapping[str, Any]] | None = None,
    device: str = "cuda:0",
    dtype: str | None = None,
    trust_remote_code: bool = True,
    offload_to_disk: bool = True,
    offload_to_disk_path: str | None = None,
    hessian_mse: bool = True,
    damp_percent: float = 0.01,
    moe_routing: str = "bypass",
    moe_routing_batch_size: int | None = None,
    moe_num_experts_per_tok: int | str | None = None,
    auto_round_version: str = "v1",
    auto_round_iters: int = 200,
    auto_round_lr: float | None = None,
    auto_round_minmax_lr: float | None = None,
    max_shard_size: str = "4GB",
    max_quant_layers: int | None = None,
    seed: int = 42,
    dry_run: bool = False,
) -> Xing4_0QuantRecipeResult:
    """Quantize a Xing4_0 HF checkpoint with GPTQ or AutoRound.

    ``calibration_data`` accepts plain text records; the recipe tokenizes them
    into the contiguous fixed-length WikiText blocks used by Xing4_0.
    Alternatively pass ``calibration_jsonl`` or ``calibration_dataset``.
    """

    normalized_method = _normalize_method(method)
    normalized_format = str(artifact_format).lower()
    _validate_recipe_args(
        method=normalized_method,
        artifact_format=normalized_format,
        bits=bits,
        self_attn_bits=self_attn_bits,
        dense_mlp_bits=dense_mlp_bits,
        shared_expert_bits=shared_expert_bits,
        expert_bits=expert_bits,
        group_size=group_size,
        sym=sym,
        batch_size=batch_size,
        nsamples=nsamples,
        seqlen=seqlen,
        moe_routing=moe_routing,
        moe_routing_batch_size=moe_routing_batch_size,
        moe_num_experts_per_tok=moe_num_experts_per_tok,
        auto_round_version=auto_round_version,
        auto_round_iters=auto_round_iters,
        auto_round_lr=auto_round_lr,
        auto_round_minmax_lr=auto_round_minmax_lr,
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
        offload_path=offload_to_disk_path,
    )
    num_layers = (
        _read_num_hidden_layers(model_path) if max_quant_layers is not None else None
    )
    dynamic = build_xing4_0_dynamic(
        self_attn_bits=self_attn_bits,
        dense_mlp_bits=dense_mlp_bits,
        shared_expert_bits=shared_expert_bits,
        expert_bits=expert_bits,
        group_size=group_size,
        sym=sym,
        num_layers=num_layers,
        max_quant_layers=max_quant_layers,
    )

    precision = {
        "self_attn": f"W{int(self_attn_bits)}",
        "dense_mlp": f"W{int(dense_mlp_bits)}",
        "shared_experts": f"W{int(shared_expert_bits)}",
        "routed_experts": f"W{int(expert_bits)}",
        "base_bits": int(bits),
        "group_size": int(group_size),
        "symmetric": bool(sym),
        "damp_percent": float(damp_percent),
    }
    calibration = {
        "jsonl": None if calibration_path is None else str(calibration_path),
        "dataset": calibration_dataset,
        "dataset_config": calibration_dataset_config,
        "split": str(calibration_split),
        "cache_dir": calibration_cache_dir,
        "text_key": str(calibration_text_key),
        "nsamples": int(nsamples),
        "seqlen": int(seqlen),
    }
    provenance = {
        "recipe": "gptqmodel.recipes.xing4_0.quantize_xing4_0",
        "backend": f"direct_gptqmodel_{normalized_method}_api",
        "dry_run": bool(dry_run),
        "offload_to_disk": bool(offload_to_disk),
        "offload_to_disk_path": resolved_offload_path,
        "hessian_mse": bool(hessian_mse),
        "trust_remote_code": bool(trust_remote_code),
        "batch_size": int(batch_size),
        "max_shard_size": str(max_shard_size),
        "moe_routing": moe_routing,
        "moe_routing_batch_size": moe_routing_batch_size,
        "moe_num_experts_per_tok": moe_num_experts_per_tok,
        "auto_round": {
            "enabled": normalized_method == "auto_round",
            "version": str(auto_round_version).lower(),
            "iters": int(auto_round_iters),
            "lr": auto_round_lr,
            "minmax_lr": auto_round_minmax_lr,
        },
        "dynamic": dynamic,
        "seed": int(seed),
    }
    result = Xing4_0QuantRecipeResult(
        output_dir=str(output_path),
        method=normalized_method,
        artifact_format=normalized_format,
        model_family="xing4_0",
        precision=precision,
        calibration=calibration,
        provenance=provenance,
    )
    if dry_run:
        return result

    if not model_path.is_dir():
        raise FileNotFoundError(f"Xing4_0 model directory not found: {model_path}")
    if output_path.exists() and any(output_path.iterdir()):
        raise FileExistsError(
            f"Refusing to overwrite non-empty output directory: {output_path}"
        )

    import torch
    from transformers import AutoTokenizer

    from gptqmodel import GPTQModel, QuantizeConfig
    from gptqmodel.quantization.config import (
        AutoRoundVersion,
        ExpertsRoutingBypass,
        ExpertsRoutingOverride,
        MoEConfig,
    )

    random.seed(seed)
    torch.manual_seed(seed)
    texts = _load_calibration_texts(
        calibration_data=calibration_data,
        calibration_jsonl=calibration_path,
        calibration_dataset=calibration_dataset,
        calibration_dataset_config=calibration_dataset_config,
        calibration_split=calibration_split,
        calibration_cache_dir=calibration_cache_dir,
        calibration_text_key=calibration_text_key,
        nsamples=nsamples,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path),
        trust_remote_code=bool(trust_remote_code),
        use_fast=True,
    )
    calibration = tokenize_xing4_0_calibration(
        texts,
        tokenizer=tokenizer,
        seqlen=seqlen,
        nsamples=nsamples,
    )

    auto_round_enabled = normalized_method == "auto_round"
    moe = None
    if moe_routing == "bypass":
        moe = MoEConfig(routing=ExpertsRoutingBypass(batch_size=moe_routing_batch_size))
    elif moe_routing == "override":
        moe = MoEConfig(
            routing=ExpertsRoutingOverride(
                num_experts_per_tok=(
                    "all"
                    if moe_num_experts_per_tok is None
                    else moe_num_experts_per_tok
                )
            )
        )
    torch_dtype = _resolve_dtype(dtype, auto_round_enabled=auto_round_enabled)
    quantize_config = QuantizeConfig(
        bits=int(bits),
        dynamic=dynamic,
        group_size=int(group_size),
        sym=bool(sym),
        desc_act=False,
        device=str(device),
        hessian_mse=bool(hessian_mse),
        wait_for_submodule_finalizers=True,
        damp_percent=float(damp_percent),
        moe=moe,
        offload_to_disk=bool(offload_to_disk),
        offload_to_disk_path=resolved_offload_path,
        auto_round_enabled=auto_round_enabled,
        auto_round_version=(
            AutoRoundVersion(str(auto_round_version).lower())
            if auto_round_enabled
            else None
        ),
        iters=int(auto_round_iters),
        lr=auto_round_lr if auto_round_enabled else None,
        minmax_lr=auto_round_minmax_lr if auto_round_enabled else None,
    )
    model = GPTQModel.load(
        str(model_path),
        quantize_config=quantize_config,
        trust_remote_code=bool(trust_remote_code),
        dtype=torch_dtype,
    )
    model.quantize(calibration, batch_size=int(batch_size))
    model.save(str(output_path), max_shard_size=str(max_shard_size))
    _write_provenance(output_path, result)
    return result


def build_xing4_0_dynamic(
    *,
    self_attn_bits: int = 8,
    dense_mlp_bits: int = 8,
    shared_expert_bits: int = 8,
    expert_bits: int = 4,
    group_size: int = 64,
    sym: bool = True,
    num_layers: int | None = None,
    max_quant_layers: int | None = None,
) -> dict[str, dict[str, Any]]:
    """Build the Xing4_0 mixed-precision dynamic overrides."""

    dynamic: dict[str, dict[str, Any]] = {
        _SELF_ATTN_PATTERN: {
            "bits": int(self_attn_bits),
            "group_size": int(group_size),
            "sym": bool(sym),
        },
        _DENSE_MLP_PATTERN: {
            "bits": int(dense_mlp_bits),
            "group_size": int(group_size),
            "sym": bool(sym),
        },
        _EXPERT_PATTERN: {
            "bits": int(expert_bits),
            "group_size": int(group_size),
            "sym": bool(sym),
        },
        _SHARED_EXPERT_PATTERN: {
            "bits": int(shared_expert_bits),
            "group_size": int(group_size),
            "sym": bool(sym),
        },
    }
    if max_quant_layers is None:
        return dynamic
    if num_layers is None:
        raise ValueError("num_layers is required when max_quant_layers is set")
    exclusions = {
        rf"-:model\.layers\.{layer_index}\..*": {}
        for layer_index in range(int(max_quant_layers), int(num_layers))
    }
    return {**exclusions, **dynamic}


def tokenize_xing4_0_calibration(
    texts: list[str],
    *,
    tokenizer: Any,
    seqlen: int,
    nsamples: int,
) -> list[dict[str, list[int]]]:
    """Tokenize WikiText-style rows into contiguous fixed-length blocks."""

    encoded = tokenizer("\n\n".join(texts))
    input_ids = list(encoded["input_ids"])
    attention_mask = list(encoded["attention_mask"])
    sample_count = min(int(nsamples), len(input_ids) // int(seqlen))
    if sample_count < 1:
        raise RuntimeError(
            "Xing4_0 calibration data does not contain one complete sequence"
        )
    return [
        {
            "input_ids": input_ids[start : start + int(seqlen)],
            "attention_mask": attention_mask[start : start + int(seqlen)],
        }
        for start in range(0, sample_count * int(seqlen), int(seqlen))
    ]


def _validate_recipe_args(
    *,
    method: str,
    artifact_format: str,
    bits: int,
    self_attn_bits: int,
    dense_mlp_bits: int,
    shared_expert_bits: int,
    expert_bits: int,
    group_size: int,
    sym: bool,
    batch_size: int,
    nsamples: int,
    seqlen: int,
    moe_routing: str,
    moe_routing_batch_size: int | None,
    moe_num_experts_per_tok: int | str | None,
    auto_round_version: str,
    auto_round_iters: int,
    auto_round_lr: float | None,
    auto_round_minmax_lr: float | None,
    max_quant_layers: int | None,
) -> None:
    if method not in _SUPPORTED_METHODS:
        raise ValueError("Xing4_0 recipe method must be 'gptq' or 'auto_round'")
    if artifact_format != "gptqmodel_hf":
        raise ValueError("artifact_format must be 'gptqmodel_hf'")
    if int(group_size) != 64:
        raise ValueError("Xing4_0 recipe requires group_size=64")
    for value, name in (
        (bits, "bits"),
        (self_attn_bits, "self_attn_bits"),
        (dense_mlp_bits, "dense_mlp_bits"),
        (shared_expert_bits, "shared_expert_bits"),
        (expert_bits, "expert_bits"),
    ):
        if int(value) not in {2, 3, 4, 5, 8}:
            raise ValueError(f"{name} must be one of [2, 3, 4, 5, 8]")
    if sym is not True:
        raise ValueError("Xing4_0 recipe requires symmetric quantization")
    if int(batch_size) <= 0 or int(nsamples) <= 0 or int(seqlen) <= 0:
        raise ValueError("batch_size, nsamples, and seqlen must be positive")
    if moe_routing not in {"none", "bypass", "override"}:
        raise ValueError("moe_routing must be one of: none, bypass, override")
    if moe_routing_batch_size is not None and int(moe_routing_batch_size) <= 0:
        raise ValueError("moe_routing_batch_size must be positive when specified")
    if (
        moe_num_experts_per_tok is not None
        and moe_num_experts_per_tok != "all"
        and int(moe_num_experts_per_tok) <= 0
    ):
        raise ValueError("moe_num_experts_per_tok must be positive or 'all'")
    if str(auto_round_version).lower() not in {"v1", "v2"}:
        raise ValueError("auto_round_version must be 'v1' or 'v2'")
    if int(auto_round_iters) < 0:
        raise ValueError("auto_round_iters must be non-negative")
    if auto_round_lr is not None and float(auto_round_lr) <= 0:
        raise ValueError("auto_round_lr must be positive when specified")
    if auto_round_minmax_lr is not None and float(auto_round_minmax_lr) <= 0:
        raise ValueError("auto_round_minmax_lr must be positive when specified")
    if max_quant_layers is not None and int(max_quant_layers) < 0:
        raise ValueError("max_quant_layers must be non-negative")


def _normalize_method(method: str) -> str:
    normalized = str(method).strip().lower().replace("-", "_")
    if normalized == "autoround":
        normalized = "auto_round"
    return normalized


def _read_num_hidden_layers(model_path: Path) -> int:
    if not model_path.is_dir():
        raise FileNotFoundError(f"Xing4_0 model directory not found: {model_path}")
    config_path = model_path / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Xing4_0 config not found: {config_path}")
    with config_path.open(encoding="utf-8") as stream:
        config = json.load(stream)
    num_layers = int(config.get("num_hidden_layers", 0))
    if num_layers <= 0:
        raise ValueError("num_hidden_layers must be positive in Xing4_0 config")
    return num_layers


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


def _resolve_dtype(dtype: str | None, *, auto_round_enabled: bool) -> Any:
    if dtype:
        import torch

        resolved = getattr(torch, str(dtype).lstrip("torch."), None)
        if resolved is None:
            raise ValueError(f"Unsupported dtype: {dtype!r}")
        return resolved
    import torch

    return torch.float16 if auto_round_enabled else torch.bfloat16


def _load_calibration_texts(
    *,
    calibration_data: Iterable[str | Mapping[str, Any]] | None,
    calibration_jsonl: Path | None,
    calibration_dataset: str | None,
    calibration_dataset_config: str | None,
    calibration_split: str,
    calibration_cache_dir: str | None,
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
        if not calibration_dataset:
            raise ValueError(
                "Provide calibration_data, calibration_jsonl, or calibration_dataset"
            )

        source = _load_dataset_with_fallback(
            calibration_dataset=calibration_dataset,
            calibration_dataset_config=calibration_dataset_config,
            calibration_split=calibration_split,
            calibration_cache_dir=calibration_cache_dir,
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


def _load_dataset_with_fallback(
    *,
    calibration_dataset: str,
    calibration_dataset_config: str | None,
    calibration_split: str,
    calibration_cache_dir: str | None,
) -> Iterable[Mapping[str, Any]]:
    from datasets import load_dataset

    dataset_kwargs: dict[str, Any] = {
        "split": calibration_split,
        "cache_dir": calibration_cache_dir,
        "verification_mode": "no_checks",
    }
    if calibration_dataset_config:
        args: tuple[str, ...] = (calibration_dataset, calibration_dataset_config)
    else:
        args = (calibration_dataset,)
    try:
        dataset = load_dataset(*args, **dataset_kwargs)
    except TypeError as exc:
        if "verification_mode" not in str(exc):
            raise
        dataset_kwargs.pop("verification_mode")
        dataset = load_dataset(*args, **dataset_kwargs)
    return dataset


def _write_provenance(
    output_path: Path,
    result: Xing4_0QuantRecipeResult,
) -> None:
    output_path.mkdir(parents=True, exist_ok=True)
    (output_path / "xing4_0_quant_recipe_provenance.json").write_text(
        json.dumps(asdict(result), ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )


__all__ = [
    "Xing4_0QuantRecipeResult",
    "build_xing4_0_dynamic",
    "quantize_xing4_0",
    "tokenize_xing4_0_calibration",
]
