#!/usr/bin/env python3
"""Audit every packed DeepSeek-V4 GPTQ tensor against its BF16 source.

The audit is deliberately independent of ``GPTQModel.load``.  It reads the
safetensor index directly, resolves the native DeepSeek tensor names to their
Transformers/GPTQ names, decodes GPTQ-v1 qzeros, dequantizes each Linear, and
computes one exact set of error metrics per tensor.  This makes it useful for
separating checkpoint/packing errors from runtime kernel or graph errors.

Example (two GPUs, exact full-model audit)::

    python audit_deepseek_v4_packed_checkpoint.py \
      --quant-model work_dirs/...-repacked-clamped-v2 \
      --bf16-model work_dirs/deepseek-v4-flash-bf16-full \
      --output work_dirs/...-repacked-clamped-v2/full_weight_audit \
      --devices cuda:6,cuda:7

The BF16 staging checkpoint uses official/native names (``w1/w2/w3``), while
the packed checkpoint uses Transformers names (``gate/down/up_proj``).  The
mapping is imported from the checkpoint repair utility so repair and audit
cannot silently disagree about tensor identity.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import multiprocessing as mp
import os
import re
import statistics
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open

from repair_deepseek_v4_packed_checkpoint import quant_module_source_key


_MASK32 = 0xFFFF_FFFF
_COMPANIONS = ("qweight", "qzeros", "scales", "g_idx")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quant-model", type=Path, required=True)
    parser.add_argument("--bf16-model", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output prefix; writes .csv, .json, and temporary .part-*.jsonl files.",
    )
    parser.add_argument(
        "--devices",
        default="cuda:0",
        help="Comma-separated workers, for example cuda:6,cuda:7 or cpu.",
    )
    parser.add_argument(
        "--qzero-source-offset",
        choices=("module", "global"),
        default="module",
        help=(
            "GPTQ-v1 packed-one offset used by the artifact. Normal artifacts use "
            "'module'. The known pre-repair mixed W8/W4 artifact used 'global'."
        ),
    )
    parser.add_argument(
        "--module-regex",
        help="Optional re.search filter over packed module names.",
    )
    parser.add_argument(
        "--max-modules",
        type=int,
        help="Optional deterministic limit after sorting/filtering (for smoke audits).",
    )
    parser.add_argument(
        "--sample-in-features",
        type=int,
        help="Audit only the first N input features of every tensor (quick triage).",
    )
    parser.add_argument(
        "--sample-out-features",
        type=int,
        help="Audit only the first N output features of every tensor (quick triage).",
    )
    parser.add_argument(
        "--aux-only",
        action="store_true",
        help="Validate mapping/shapes/scales/qzeros/g_idx without reading weight values.",
    )
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument(
        "--keep-parts",
        action="store_true",
        help="Keep worker JSONL files after CSV/JSON merge.",
    )
    return parser.parse_args()


def load_index(model_dir: Path) -> dict[str, Any]:
    path = model_dir / "model.safetensors.index.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def load_quant_config(model_dir: Path) -> dict[str, Any]:
    config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    qconfig = config.get("quantization_config")
    if not isinstance(qconfig, dict):
        quantize_path = model_dir / "quantize_config.json"
        if not quantize_path.is_file():
            raise KeyError("No quantization_config or quantize_config.json found")
        qconfig = json.loads(quantize_path.read_text(encoding="utf-8"))
    return qconfig


def module_scheme(module_name: str, qconfig: dict[str, Any]) -> dict[str, Any]:
    """Resolve bits/group/sym exactly like QuantizeConfig.dynamic_get (re.match)."""

    resolved = {
        "bits": int(qconfig["bits"]),
        "group_size": int(qconfig["group_size"]),
        "sym": bool(qconfig.get("sym", True)),
        "dynamic_pattern": None,
    }
    for raw_pattern, overrides in (qconfig.get("dynamic") or {}).items():
        if raw_pattern.startswith("-:"):
            if re.match(raw_pattern[2:], module_name):
                raise ValueError(f"Module {module_name} is dynamically excluded")
            continue
        pattern = raw_pattern[2:] if raw_pattern.startswith("+:") else raw_pattern
        if re.match(pattern, module_name):
            for key in ("bits", "group_size", "sym"):
                if key in overrides:
                    resolved[key] = overrides[key]
            resolved["bits"] = int(resolved["bits"])
            resolved["group_size"] = int(resolved["group_size"])
            resolved["sym"] = bool(resolved["sym"])
            resolved["dynamic_pattern"] = raw_pattern
            break
    return resolved


def packed_one_offset(bits: int) -> int:
    if bits not in (2, 4, 8):
        raise ValueError(f"Packed-one decode currently supports 2/4/8 bits, got {bits}")
    return sum(1 << (slot * bits) for slot in range(32 // bits))


def unpack_qzeros_v1(
    qzeros: torch.Tensor,
    *,
    bits: int,
    source_offset_bits: int,
) -> torch.Tensor:
    """Decode GPTQ-v1 words to logical unsigned zero points.

    GPTQ-v1 serializes a v2 word minus a packed word of ones.  Addition is
    performed on the complete uint32 word (including carries), matching the
    runtime v1-to-v2 conversion rather than adding one independently per slot.
    """

    packed_v2 = (
        (qzeros.to(torch.int64) & _MASK32) + packed_one_offset(source_offset_bits)
    ) & _MASK32
    mask = (1 << bits) - 1
    shifts = torch.arange(0, 32, bits, dtype=torch.int64)
    return (
        (packed_v2.unsqueeze(-1) >> shifts) & mask
    ).reshape(qzeros.shape[0], -1).to(torch.int32)


class TensorStore:
    """Index-aware safetensor reader with a bounded mmap-reader LRU."""

    def __init__(
        self,
        model_dir: Path,
        weight_map: dict[str, str],
        *,
        max_open_shards: int = 8,
    ) -> None:
        self.model_dir = model_dir
        self.weight_map = weight_map
        self.max_open_shards = max_open_shards
        self.readers: OrderedDict[str, tuple[Any, Any]] = OrderedDict()

    def __enter__(self) -> "TensorStore":
        return self

    def __exit__(self, *exc: object) -> None:
        while self.readers:
            _, (context, _) = self.readers.popitem(last=False)
            context.__exit__(None, None, None)

    def tensor(self, key: str) -> torch.Tensor:
        shard = self.weight_map[key]
        entry = self.readers.pop(shard, None)
        if entry is None:
            context = safe_open(
                self.model_dir / shard, framework="pt", device="cpu"
            )
            reader = context.__enter__()
            entry = (context, reader)
        self.readers[shard] = entry
        while len(self.readers) > self.max_open_shards:
            _, (old_context, _) = self.readers.popitem(last=False)
            old_context.__exit__(None, None, None)
        _, reader = entry
        return reader.get_tensor(key)


def module_sort_key(module_name: str) -> tuple[int, int, int, str]:
    layer_match = re.search(r"\.layers\.(\d+)\.", module_name)
    expert_match = re.search(r"\.experts\.(\d+)\.", module_name)
    layer = int(layer_match.group(1)) if layer_match else -1
    expert = int(expert_match.group(1)) if expert_match else -1
    if ".self_attn." in module_name:
        family = 0
    elif ".shared_experts." in module_name:
        family = 1
    else:
        family = 2
    return layer, family, expert, module_name


def module_labels(module_name: str) -> dict[str, Any]:
    layer_match = re.search(r"\.layers\.(\d+)\.", module_name)
    expert_match = re.search(r"\.experts\.(\d+)\.", module_name)
    if ".self_attn." in module_name:
        family = "attention"
    elif ".shared_experts." in module_name:
        family = "shared_expert"
    else:
        family = "routed_expert"
    return {
        "layer": int(layer_match.group(1)) if layer_match else -1,
        "family": family,
        "expert": int(expert_match.group(1)) if expert_match else "",
        "projection": module_name.rsplit(".", 1)[-1],
    }


@torch.inference_mode()
def audit_module(
    module_name: str,
    source_key: str,
    scheme: dict[str, Any],
    qstore: TensorStore,
    bstore: TensorStore,
    device: torch.device,
    qzero_source_offset: str,
    global_bits: int,
    aux_only: bool,
    sample_in_features: int | None,
    sample_out_features: int | None,
) -> dict[str, Any]:
    started = time.perf_counter()
    bits = int(scheme["bits"])
    group_size = int(scheme["group_size"])
    pack_factor = 32 // bits
    prefix = module_name + "."
    qweight = qstore.tensor(prefix + "qweight")
    qzeros = qstore.tensor(prefix + "qzeros")
    scales = qstore.tensor(prefix + "scales")
    g_idx = qstore.tensor(prefix + "g_idx")
    source = bstore.tensor(source_key)

    if source.ndim != 2:
        raise ValueError(f"{source_key}: expected matrix, got {tuple(source.shape)}")
    out_features, in_features = map(int, source.shape)
    groups = math.ceil(in_features / group_size)
    expected_shapes = {
        "qweight": (in_features // pack_factor, out_features),
        "qzeros": (groups, out_features // pack_factor),
        "scales": (groups, out_features),
        "g_idx": (in_features,),
    }
    actual_shapes = {
        "qweight": tuple(qweight.shape),
        "qzeros": tuple(qzeros.shape),
        "scales": tuple(scales.shape),
        "g_idx": tuple(g_idx.shape),
    }
    structural_errors = [
        f"{name}:expected={expected_shapes[name]},actual={actual_shapes[name]}"
        for name in expected_shapes
        if expected_shapes[name] != actual_shapes[name]
    ]
    source_offset_bits = global_bits if qzero_source_offset == "global" else bits
    zeros = unpack_qzeros_v1(
        qzeros,
        bits=bits,
        source_offset_bits=source_offset_bits,
    )
    maximum = (1 << bits) - 1
    expected_zero = (1 << (bits - 1)) if scheme["sym"] else None
    zero_range_invalid = int(((zeros < 0) | (zeros > maximum)).sum().item())
    zero_sym_invalid = (
        int((zeros != expected_zero).sum().item())
        if expected_zero is not None
        else 0
    )
    scale_finite = torch.isfinite(scales)
    scale_nonfinite = int((~scale_finite).sum().item())
    scale_nonpositive = int((scale_finite & (scales <= 0)).sum().item())
    g_idx_long = g_idx.to(torch.int64)
    g_idx_invalid = int(((g_idx_long < 0) | (g_idx_long >= groups)).sum().item())
    expected_g_idx = torch.arange(in_features, dtype=torch.int64) // group_size
    g_idx_layout_mismatch = (
        int((g_idx_long != expected_g_idx).sum().item())
        if tuple(g_idx_long.shape) == (in_features,)
        else in_features
    )
    if zero_range_invalid or scale_nonfinite or g_idx_invalid:
        structural_errors.append("invalid quantization auxiliary values")
    if structural_errors:
        raise ValueError("; ".join(structural_errors))

    selected_in = min(in_features, sample_in_features or in_features)
    selected_out = min(out_features, sample_out_features or out_features)
    metric_mode = (
        "aux_only"
        if aux_only
        else "exact"
        if selected_in == in_features and selected_out == out_features
        else f"prefix_sample_{selected_out}x{selected_in}"
    )
    common: dict[str, Any] = {
        "module": module_name,
        "source_key": source_key,
        **module_labels(module_name),
        "bits": bits,
        "scheme": f"W{bits}",
        "group_size": group_size,
        "sym": bool(scheme["sym"]),
        "dynamic_pattern": scheme["dynamic_pattern"] or "",
        "qzero_source_offset": qzero_source_offset,
        "source_offset_bits": source_offset_bits,
        "out_features": out_features,
        "in_features": in_features,
        "full_numel": int(source.numel()),
        "metric_mode": metric_mode,
        "selected_out_features": 0 if aux_only else selected_out,
        "selected_in_features": 0 if aux_only else selected_in,
        "source_dtype": str(source.dtype).removeprefix("torch."),
        "scale_dtype": str(scales.dtype).removeprefix("torch."),
        "qweight_shape": "x".join(map(str, qweight.shape)),
        "qzeros_shape": "x".join(map(str, qzeros.shape)),
        "scales_shape": "x".join(map(str, scales.shape)),
        "g_idx_shape": "x".join(map(str, g_idx.shape)),
        "scale_nonfinite": scale_nonfinite,
        "scale_nonpositive": scale_nonpositive,
        "scale_min": float(scales.float().min().item()),
        "scale_max": float(scales.float().max().item()),
        "qzero_expected": expected_zero if expected_zero is not None else "",
        "qzero_min": int(zeros.min().item()),
        "qzero_max": int(zeros.max().item()),
        "qzero_range_invalid": zero_range_invalid,
        "qzero_sym_invalid": zero_sym_invalid,
        "g_idx_min": int(g_idx_long.min().item()),
        "g_idx_max": int(g_idx_long.max().item()),
        "g_idx_invalid": g_idx_invalid,
        "g_idx_layout_mismatch": g_idx_layout_mismatch,
    }
    if aux_only:
        common.update(
            numel=0,
            elapsed_seconds=time.perf_counter() - started,
            status="ok",
            error="",
        )
        return common

    # Packed qweight rows are [packed_input, output].  Slot-major expansion
    # followed by reshape reconstructs [input, output] in original row order.
    shifts = torch.arange(0, 32, bits, dtype=torch.int32, device=device)
    selected_packed_in = math.ceil(selected_in / pack_factor)
    qword = qweight[:selected_packed_in, :selected_out].to(
        device=device, dtype=torch.int32, non_blocking=False
    )
    weight_int = (
        (qword.unsqueeze(1) >> shifts.reshape(1, pack_factor, 1)) & maximum
    ).reshape(selected_packed_in * pack_factor, selected_out)[:selected_in]
    groups_device = g_idx_long[:selected_in].to(device=device)
    scales_device = scales[:, :selected_out].to(device=device, dtype=torch.float32)
    zeros_device = zeros[:, :selected_out].to(device=device, dtype=torch.float32)
    dequant = (
        weight_int.to(torch.float32) - zeros_device.index_select(0, groups_device)
    ) * scales_device.index_select(0, groups_device)
    reference = source[:selected_out, :selected_in].to(device=device, dtype=torch.float32)
    dequant_out_in = dequant.transpose(0, 1)
    error = dequant_out_in - reference

    # FP32 reduction is ample for these bounded weights and is much faster than
    # A100 FP64.  All scalar values are transferred in one synchronization.
    error_sq = torch.sum(error * error)
    ref_sq = torch.sum(reference * reference)
    deq_sq = torch.sum(dequant_out_in * dequant_out_in)
    dot = torch.sum(dequant_out_in * reference)
    scalars = torch.stack(
        (
            error_sq,
            ref_sq,
            deq_sq,
            dot,
            torch.sum(error),
            torch.max(torch.abs(error)),
            torch.max(torch.abs(reference)),
            torch.max(torch.abs(dequant_out_in)),
        )
    ).cpu().tolist()
    error_sq_v, ref_sq_v, deq_sq_v, dot_v, error_sum, max_error, ref_max, deq_max = (
        float(value) for value in scalars
    )
    numel = int(reference.numel())
    tiny = 1e-30
    rel_rmse = math.sqrt(error_sq_v / max(ref_sq_v, tiny))
    rmse = math.sqrt(error_sq_v / numel)
    snr_db = 20.0 * math.log10(
        math.sqrt(max(ref_sq_v, tiny)) / max(math.sqrt(error_sq_v), tiny)
    )
    cosine = dot_v / max(math.sqrt(ref_sq_v * deq_sq_v), tiny)
    row: dict[str, Any] = {
        **common,
        "numel": numel,
        "rmse": rmse,
        "rel_rmse": rel_rmse,
        "snr_db": snr_db,
        "cosine": cosine,
        "mean_bias": error_sum / numel,
        "max_abs_error": max_error,
        "reference_abs_max": ref_max,
        "dequant_abs_max": deq_max,
        "reference_l2": math.sqrt(ref_sq_v),
        "error_l2": math.sqrt(error_sq_v),
        "elapsed_seconds": time.perf_counter() - started,
        "status": "ok",
        "error": "",
    }
    del qword, weight_int, groups_device, scales_device, zeros_device
    del dequant, reference, dequant_out_in, error
    return row


def error_row(
    module_name: str,
    source_key: str,
    scheme: dict[str, Any],
    exc: BaseException,
    elapsed: float,
) -> dict[str, Any]:
    return {
        "module": module_name,
        "source_key": source_key,
        **module_labels(module_name),
        "bits": int(scheme["bits"]),
        "scheme": f"W{int(scheme['bits'])}",
        "group_size": int(scheme["group_size"]),
        "sym": bool(scheme["sym"]),
        "dynamic_pattern": scheme["dynamic_pattern"] or "",
        "elapsed_seconds": elapsed,
        "status": "error",
        "error": f"{type(exc).__name__}: {exc}",
    }


def worker_main(
    worker_id: int,
    device_name: str,
    quant_model: str,
    bf16_model: str,
    quant_weight_map: dict[str, str],
    bf16_weight_map: dict[str, str],
    work: list[tuple[str, str, dict[str, Any]]],
    qzero_source_offset: str,
    global_bits: int,
    aux_only: bool,
    sample_in_features: int | None,
    sample_out_features: int | None,
    part_path: str,
    progress_every: int,
) -> None:
    # Small auxiliary reductions otherwise make every spawned worker claim the
    # host's full OpenMP pool, which hurts both mmap throughput and co-located
    # inference jobs.  CUDA does the full-weight arithmetic.
    torch.set_num_threads(int(os.environ.get("DEEPSEEK_AUDIT_CPU_THREADS", "4")))
    torch.set_num_interop_threads(1)
    device = torch.device(device_name)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    path = Path(part_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    errors = 0
    with (
        TensorStore(Path(quant_model), quant_weight_map) as qstore,
        TensorStore(Path(bf16_model), bf16_weight_map) as bstore,
        path.open("w", encoding="utf-8") as output,
    ):
        for index, (module_name, source_key, scheme) in enumerate(work, start=1):
            item_started = time.perf_counter()
            try:
                row = audit_module(
                    module_name,
                    source_key,
                    scheme,
                    qstore,
                    bstore,
                    device,
                    qzero_source_offset,
                    global_bits,
                    aux_only,
                    sample_in_features,
                    sample_out_features,
                )
            except BaseException as exc:
                errors += 1
                row = error_row(
                    module_name,
                    source_key,
                    scheme,
                    exc,
                    time.perf_counter() - item_started,
                )
            output.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            output.flush()
            if index % progress_every == 0 or index == len(work):
                elapsed = time.perf_counter() - started
                rate = index / max(elapsed, 1e-9)
                print(
                    f"[audit worker={worker_id} device={device_name}] "
                    f"{index}/{len(work)} errors={errors} rate={rate:.2f} tensor/s "
                    f"last={module_name}",
                    flush=True,
                )


def quantile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def metric_distribution(rows: list[dict[str, Any]], field: str) -> dict[str, Any]:
    values = [
        float(row[field])
        for row in rows
        if row.get("status") == "ok" and field in row
    ]
    if not values:
        return {"count": 0}
    return {
        "count": len(values),
        "min": min(values),
        "mean": statistics.fmean(values),
        "p50": quantile(values, 0.50),
        "p90": quantile(values, 0.90),
        "p95": quantile(values, 0.95),
        "p99": quantile(values, 0.99),
        "max": max(values),
    }


def merge_outputs(
    part_paths: list[Path],
    output_prefix: Path,
    metadata: dict[str, Any],
) -> tuple[Path, Path, dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in part_paths:
        with path.open(encoding="utf-8") as source:
            rows.extend(json.loads(line) for line in source if line.strip())
    rows.sort(key=lambda row: module_sort_key(row["module"]))
    csv_path = output_prefix.with_suffix(".csv")
    json_path = output_prefix.with_suffix(".json")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    preferred = [
        "module", "source_key", "layer", "family", "expert", "projection",
        "bits", "scheme", "group_size", "sym", "numel", "rel_rmse", "rmse",
        "snr_db", "cosine", "mean_bias", "max_abs_error", "reference_abs_max",
        "dequant_abs_max", "scale_nonfinite", "scale_nonpositive", "scale_min",
        "scale_max", "qzero_expected", "qzero_min", "qzero_max",
        "qzero_range_invalid", "qzero_sym_invalid", "g_idx_invalid",
        "g_idx_layout_mismatch", "status", "error", "elapsed_seconds",
    ]
    all_fields = set().union(*(row.keys() for row in rows)) if rows else set()
    fieldnames = preferred + sorted(all_fields - set(preferred))
    with csv_path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    ok_rows = [row for row in rows if row.get("status") == "ok"]
    error_rows = [row for row in rows if row.get("status") != "ok"]
    by_scheme: dict[str, Any] = {}
    for scheme in sorted({row["scheme"] for row in rows}):
        subset = [row for row in ok_rows if row["scheme"] == scheme]
        by_scheme[scheme] = {
            "tensors": len(subset),
            "elements": sum(int(row.get("numel", 0)) for row in subset),
            "rel_rmse": metric_distribution(subset, "rel_rmse"),
            "snr_db": metric_distribution(subset, "snr_db"),
            "cosine": metric_distribution(subset, "cosine"),
            "max_abs_error": metric_distribution(subset, "max_abs_error"),
        }
    top_rel = sorted(
        (row for row in ok_rows if "rel_rmse" in row),
        key=lambda row: float(row["rel_rmse"]),
        reverse=True,
    )[:100]
    invalid_counts = {
        field: sum(int(row.get(field, 0)) for row in ok_rows)
        for field in (
            "scale_nonfinite", "scale_nonpositive", "qzero_range_invalid",
            "qzero_sym_invalid", "g_idx_invalid", "g_idx_layout_mismatch",
        )
    }
    focus_names = (
        "model.layers.29.mlp.experts.126.gate_proj",
        "model.layers.29.mlp.experts.126.up_proj",
        "model.layers.29.mlp.experts.126.down_proj",
    )
    summary = {
        **metadata,
        "audited_tensors": len(rows),
        "successful_tensors": len(ok_rows),
        "failed_tensors": len(error_rows),
        "audited_elements": sum(int(row.get("numel", 0)) for row in ok_rows),
        "by_scheme": by_scheme,
        "invalid_auxiliary_value_counts": invalid_counts,
        "errors": error_rows[:100],
        "top_100_rel_rmse": top_rel,
        "layer29_expert126": [row for row in rows if row["module"] in focus_names],
        "csv": str(csv_path.resolve()),
    }
    json_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return csv_path, json_path, summary


def main() -> None:
    args = parse_args()
    quant_model = args.quant_model.expanduser().resolve()
    bf16_model = args.bf16_model.expanduser().resolve()
    output_prefix = args.output.expanduser().resolve()
    for model_dir in (quant_model, bf16_model):
        if not model_dir.is_dir():
            raise FileNotFoundError(model_dir)
    quant_index = load_index(quant_model)
    bf16_index = load_index(bf16_model)
    quant_weight_map = quant_index["weight_map"]
    bf16_weight_map = bf16_index["weight_map"]
    qconfig = load_quant_config(quant_model)
    global_bits = int(qconfig["bits"])

    all_modules = sorted(
        (key[: -len(".qweight")] for key in quant_weight_map if key.endswith(".qweight")),
        key=module_sort_key,
    )
    companion_missing = {
        module: [suffix for suffix in _COMPANIONS if module + "." + suffix not in quant_weight_map]
        for module in all_modules
    }
    companion_missing = {key: value for key, value in companion_missing.items() if value}
    if companion_missing:
        raise KeyError(f"Incomplete quant modules: {list(companion_missing.items())[:10]}")

    mapping_errors: list[dict[str, str]] = []
    work: list[tuple[str, str, dict[str, Any]]] = []
    for module in all_modules:
        try:
            source_key = quant_module_source_key(module)
        except KeyError as exc:
            mapping_errors.append({"module": module, "error": str(exc)})
            continue
        if source_key not in bf16_weight_map:
            mapping_errors.append({"module": module, "source_key": source_key, "error": "missing BF16 source"})
            continue
        work.append((module, source_key, module_scheme(module, qconfig)))
    if mapping_errors:
        raise KeyError(f"Native-to-packed mapping errors: {mapping_errors[:10]}")
    mapped_before_filter = len(work)
    if args.module_regex:
        regex = re.compile(args.module_regex)
        work = [item for item in work if regex.search(item[0])]
    if args.max_modules is not None:
        if args.max_modules <= 0:
            raise ValueError("--max-modules must be positive")
        work = work[: args.max_modules]
    for name, value in (
        ("--sample-in-features", args.sample_in_features),
        ("--sample-out-features", args.sample_out_features),
    ):
        if value is not None and value <= 0:
            raise ValueError(f"{name} must be positive")
    if not work:
        raise ValueError("No modules selected")

    devices = [item.strip() for item in args.devices.split(",") if item.strip()]
    if not devices:
        raise ValueError("--devices must contain at least one device")
    workers = min(len(devices), len(work))
    # Contiguous partitions avoid two workers reading the same large BF16 shard
    # at the same time and preserve mostly sequential mmap access.
    partitions: list[list[tuple[str, str, dict[str, Any]]]] = []
    for worker_id in range(workers):
        start = len(work) * worker_id // workers
        end = len(work) * (worker_id + 1) // workers
        partitions.append(work[start:end])
    part_paths = [
        output_prefix.parent / f"{output_prefix.name}.part-{worker_id:02d}.jsonl"
        for worker_id in range(workers)
    ]
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    context = mp.get_context("spawn")
    processes: list[mp.Process] = []
    for worker_id, partition in enumerate(partitions):
        process = context.Process(
            target=worker_main,
            args=(
                worker_id, devices[worker_id], str(quant_model), str(bf16_model),
                quant_weight_map, bf16_weight_map, partition,
                args.qzero_source_offset, global_bits,
                args.aux_only, args.sample_in_features, args.sample_out_features,
                str(part_paths[worker_id]),
                args.progress_every,
            ),
        )
        process.start()
        processes.append(process)
    for process in processes:
        process.join()
    bad_workers = [process.pid for process in processes if process.exitcode != 0]
    if bad_workers:
        raise RuntimeError(f"Audit workers failed: pids={bad_workers}")

    scheme_counts: dict[str, int] = {}
    for _, _, scheme in work:
        key = f"W{int(scheme['bits'])}"
        scheme_counts[key] = scheme_counts.get(key, 0) + 1
    metadata = {
        "quant_model": str(quant_model),
        "bf16_model": str(bf16_model),
        "exact_full_tensor_metrics": (
            not args.aux_only
            and args.sample_in_features is None
            and args.sample_out_features is None
        ),
        "aux_only": args.aux_only,
        "sample_in_features": args.sample_in_features,
        "sample_out_features": args.sample_out_features,
        "qzero_decode": (
            "GPTQ-v1 whole-word addition followed by per-slot unpack; "
            f"source_offset={args.qzero_source_offset}"
        ),
        "total_packed_qweight_tensors": len(all_modules),
        "mapped_tensors_before_filter": mapped_before_filter,
        "unmapped_tensors": len(mapping_errors),
        "selected_tensors": len(work),
        "selected_scheme_counts": scheme_counts,
        "module_regex": args.module_regex,
        "max_modules": args.max_modules,
        "devices": devices[:workers],
        "elapsed_seconds": time.perf_counter() - started,
    }
    csv_path, json_path, summary = merge_outputs(part_paths, output_prefix, metadata)
    if not args.keep_parts:
        for path in part_paths:
            path.unlink(missing_ok=True)
    print(
        json.dumps(
            {
                "csv": str(csv_path),
                "json": str(json_path),
                "audited_tensors": summary["audited_tensors"],
                "failed_tensors": summary["failed_tensors"],
                "by_scheme": summary["by_scheme"],
                "invalid_auxiliary_value_counts": summary["invalid_auxiliary_value_counts"],
                "elapsed_seconds": summary["elapsed_seconds"],
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    # Avoid inheriting a huge parent address space and CUDA context.
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    main()
