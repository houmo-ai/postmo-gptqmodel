#!/usr/bin/env python3
"""Audit DeepSeek-V4 native FP4 routed experts after INT4 requantization.

The native 0731 checkpoint stores two E2M1 values in each signed byte and one
direct E8M0 scale per 32 logical input columns.  This tool reads the checkpoint
shard by shard, exactly materializes those FP4 values, and compares:

* conventional symmetric INT4 RTN with group size 32;
* the six-rate group-32 search published by baicai1145;
* the same search with FP16 rather than BF16 stored scales; and
* conventional/search INT4 with the target group size 64 and FP16 scales.

No converted checkpoint is written.  ``--rows-per-tensor`` makes an all-tensor
survey inexpensive while retaining every layer/expert/projection in the audit;
use zero for an exact full-weight pass.  Multiple CUDA devices run independent
checkpoint shards concurrently.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import multiprocessing
import re
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
import torch
from safetensors import safe_open


FP4_TABLE = (
    0.0,
    0.5,
    1.0,
    1.5,
    2.0,
    3.0,
    4.0,
    6.0,
    0.0,
    -0.5,
    -1.0,
    -1.5,
    -2.0,
    -3.0,
    -4.0,
    -6.0,
)
DEFAULT_RATES = (6.0 / 7.0, 0.9, 0.95, 1.0, 1.05, 1.1)
EXPERT_RE = re.compile(
    r"(?:^|\.)layers\.(?P<layer>\d+)\.ffn\.experts\."
    r"(?P<expert>\d+)\.(?P<projection>w1|w2|w3)\.weight$"
)


def parse_int_set(value: str | None) -> set[int] | None:
    if value is None or value.strip().lower() in {"", "all", "*"}:
        return None
    result: set[int] = set()
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        if "-" in item:
            begin, end = (int(x) for x in item.split("-", 1))
            result.update(range(begin, end + 1))
        else:
            result.add(int(item))
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Native DeepSeek-V4 checkpoint")
    parser.add_argument("--output", required=True, help="Summary JSON output")
    parser.add_argument(
        "--tensor-records",
        default=None,
        help="Optional JSONL output with one record per audited expert tensor",
    )
    parser.add_argument("--layers", default="all", help="all, comma list, or ranges")
    parser.add_argument("--experts", default="all", help="all, comma list, or ranges")
    parser.add_argument(
        "--projections", default="w1,w2,w3", help="Comma-separated projections"
    )
    parser.add_argument(
        "--rows-per-tensor",
        type=int,
        default=64,
        help="Evenly sample this many output rows; zero audits every row",
    )
    parser.add_argument(
        "--devices",
        default="cuda:0",
        help="Comma-separated CUDA devices, or cpu",
    )
    parser.add_argument(
        "--rates",
        default=",".join(str(x) for x in DEFAULT_RATES),
        help="Scale multipliers for the per-group grid search",
    )
    parser.add_argument(
        "--lloyd-initial-rates",
        default="0.7,0.8,0.9,1.0,1.1,1.2,1.3",
        help="Comma-separated amax/7 scale multipliers for multi-start LS refinement",
    )
    parser.add_argument(
        "--lloyd-iterations",
        type=int,
        default=0,
        help="Alternating round/least-squares iterations; zero disables Lloyd metrics",
    )
    parser.add_argument("--max-tensors", type=int, default=None)
    parser.add_argument("--top-outliers", type=int, default=100)
    return parser.parse_args()


@dataclass
class Metric:
    elements: int = 0
    groups: int = 0
    signal_sq: float = 0.0
    error_sq: float = 0.0
    dot: float = 0.0
    reconstruction_sq: float = 0.0
    max_abs_error: float = 0.0
    rate_counts: list[int] = field(default_factory=list)

    def merge(self, other: "Metric") -> None:
        self.elements += other.elements
        self.groups += other.groups
        self.signal_sq += other.signal_sq
        self.error_sq += other.error_sq
        self.dot += other.dot
        self.reconstruction_sq += other.reconstruction_sq
        self.max_abs_error = max(self.max_abs_error, other.max_abs_error)
        if len(self.rate_counts) < len(other.rate_counts):
            self.rate_counts.extend([0] * (len(other.rate_counts) - len(self.rate_counts)))
        for index, count in enumerate(other.rate_counts):
            self.rate_counts[index] += count

    def as_dict(self, rates: tuple[float, ...]) -> dict:
        rel_mse = self.error_sq / self.signal_sq if self.signal_sq else 0.0
        cosine = self.dot / math.sqrt(self.signal_sq * self.reconstruction_sq) \
            if self.signal_sq and self.reconstruction_sq else 1.0
        payload = {
            "elements": self.elements,
            "groups": self.groups,
            "signal_sq": self.signal_sq,
            "error_sq": self.error_sq,
            "relative_mse": rel_mse,
            "relative_rmse": math.sqrt(rel_mse),
            "snr_db": -10.0 * math.log10(rel_mse) if rel_mse > 0 else math.inf,
            "cosine": cosine,
            "max_abs_error": self.max_abs_error,
        }
        if self.rate_counts:
            total = sum(self.rate_counts)
            payload["selected_rate_counts"] = {
                str(rate): self.rate_counts[index] for index, rate in enumerate(rates)
            }
            payload["selected_rate_fractions"] = {
                str(rate): self.rate_counts[index] / total if total else 0.0
                for index, rate in enumerate(rates)
            }
        return payload


def unpack_native_fp4(
    packed: torch.Tensor,
    source_scale: torch.Tensor,
    *,
    device: torch.device,
    row_indices: torch.Tensor | None,
) -> torch.Tensor:
    if packed.dtype != torch.int8 or packed.ndim != 2 or source_scale.ndim != 2:
        raise ValueError(
            f"expected int8 rank-2 weight and rank-2 scale, got "
            f"{packed.dtype} {tuple(packed.shape)} / {source_scale.dtype} "
            f"{tuple(source_scale.shape)}"
        )
    if row_indices is not None:
        packed = packed.index_select(0, row_indices)
        source_scale = source_scale.index_select(0, row_indices)
    logical_columns = packed.shape[1] * 2
    expected_scale_shape = (packed.shape[0], logical_columns // 32)
    if tuple(source_scale.shape) != expected_scale_shape:
        raise ValueError(
            f"native scale shape {tuple(source_scale.shape)} != {expected_scale_shape}; "
            "the 0731 routed-expert layout must have one E8M0 scale per 32 values"
        )

    packed_u8 = packed.to(device=device, non_blocking=True).view(torch.uint8)
    codes = torch.stack((packed_u8 & 0x0F, (packed_u8 >> 4) & 0x0F), dim=-1)
    codes = codes.reshape(packed.shape[0], logical_columns).long()
    table = torch.tensor(FP4_TABLE, dtype=torch.float32, device=device)
    scales = source_scale.to(device=device, dtype=torch.float32, non_blocking=True)
    scales = scales.repeat_interleave(32, dim=1)
    return table[codes] * scales


def quantize_metric(
    reference: torch.Tensor,
    *,
    group_size: int,
    rates: tuple[float, ...],
    stored_scale_dtype: torch.dtype,
) -> Metric:
    if reference.shape[1] % group_size:
        raise ValueError(f"input dim {reference.shape[1]} is not divisible by {group_size}")
    values = reference.reshape(reference.shape[0], -1, group_size)
    amax = values.abs().amax(dim=-1, keepdim=True)
    rate_tensor = torch.tensor(rates, dtype=torch.float32, device=reference.device)
    candidate_scales = amax.unsqueeze(0) * rate_tensor[:, None, None, None] / 7.0
    candidate_scales = torch.where(
        candidate_scales > 0,
        candidate_scales,
        torch.ones_like(candidate_scales),
    )
    candidate_codes = torch.round(values.unsqueeze(0) / candidate_scales).clamp_(-8, 7)
    candidate_error = (values.unsqueeze(0) - candidate_codes * candidate_scales).square()
    best = candidate_error.sum(dim=-1).argmin(dim=0)
    gather_index = best[None, ..., None]
    selected_scale = candidate_scales.gather(0, gather_index).squeeze(0)

    # Match the published procedure: choose and quantize with the FP32 working
    # scale, then account for the BF16/FP16 scale that is actually stored.
    codes = torch.round(values / selected_scale).clamp_(-8, 7)
    stored_scale = selected_scale.to(stored_scale_dtype).to(torch.float32)
    reconstructed = codes * stored_scale
    difference = reference - reconstructed.reshape_as(reference)

    counts = torch.bincount(best.reshape(-1), minlength=len(rates)).cpu().tolist()
    return Metric(
        elements=reference.numel(),
        groups=best.numel(),
        signal_sq=reference.square().sum().item(),
        error_sq=difference.square().sum().item(),
        dot=(reference * reconstructed.reshape_as(reference)).sum().item(),
        reconstruction_sq=reconstructed.square().sum().item(),
        max_abs_error=difference.abs().amax().item(),
        rate_counts=[int(x) for x in counts] if len(rates) > 1 else [],
    )


def lloyd_quantize_metric(
    reference: torch.Tensor,
    *,
    group_size: int,
    initial_rates: tuple[float, ...],
    iterations: int,
    stored_scale_dtype: torch.dtype,
) -> Metric:
    """Optimize one symmetric INT4 scale per group by alternating LS updates.

    For fixed integer codes the least-squares optimum is
    ``sum(w*q) / sum(q*q)``.  Multiple starts are required because changing the
    scale changes rounded codes and the resulting one-dimensional objective is
    piecewise quadratic.  Selection includes the final stored-scale rounding,
    which is the quantity relevant to the target FP16-scale hardware format.
    """

    if reference.shape[1] % group_size:
        raise ValueError(f"input dim {reference.shape[1]} is not divisible by {group_size}")
    values = reference.reshape(reference.shape[0], -1, group_size)
    amax = values.abs().amax(dim=-1, keepdim=True)
    rate_tensor = torch.tensor(initial_rates, dtype=torch.float32, device=reference.device)
    scales = amax.unsqueeze(0) * rate_tensor[:, None, None, None] / 7.0
    scales = torch.where(scales > 0, scales, torch.ones_like(scales))

    for _ in range(iterations):
        codes = torch.round(values.unsqueeze(0) / scales).clamp_(-8, 7)
        numerator = (values.unsqueeze(0) * codes).sum(dim=-1, keepdim=True)
        denominator = codes.square().sum(dim=-1, keepdim=True)
        updated = numerator / denominator.clamp_min_(1.0)
        scales = torch.where(denominator > 0, updated, scales).clamp_min_(torch.finfo(torch.float32).tiny)

    # Quantization codes are derived from the FP32 optimized scale.  The
    # per-start objective then reflects the actual FP16/BF16 scale stored.
    codes = torch.round(values.unsqueeze(0) / scales).clamp_(-8, 7)
    stored_scales = scales.to(stored_scale_dtype).to(torch.float32)
    reconstructed_candidates = codes * stored_scales
    errors = (values.unsqueeze(0) - reconstructed_candidates).square().sum(dim=-1)
    best = errors.argmin(dim=0)
    gather_index = best[None, ..., None].expand(-1, -1, -1, group_size)
    reconstructed = reconstructed_candidates.gather(0, gather_index).squeeze(0)
    difference = reference - reconstructed.reshape_as(reference)
    counts = torch.bincount(best.reshape(-1), minlength=len(initial_rates)).cpu().tolist()
    return Metric(
        elements=reference.numel(),
        groups=best.numel(),
        signal_sq=reference.square().sum().item(),
        error_sq=difference.square().sum().item(),
        dot=(reference * reconstructed.reshape_as(reference)).sum().item(),
        reconstruction_sq=reconstructed.square().sum().item(),
        max_abs_error=difference.abs().amax().item(),
        rate_counts=[int(x) for x in counts] if len(initial_rates) > 1 else [],
    )


def metric_record(metric: Metric, rates: tuple[float, ...]) -> dict:
    return metric.as_dict(rates)


def audit_tensor(
    packed: torch.Tensor,
    scale: torch.Tensor,
    *,
    device: torch.device,
    rows_per_tensor: int,
    rates: tuple[float, ...],
    lloyd_initial_rates: tuple[float, ...],
    lloyd_iterations: int,
) -> dict[str, Metric]:
    row_indices = None
    if rows_per_tensor > 0 and rows_per_tensor < packed.shape[0]:
        row_indices = torch.linspace(
            0, packed.shape[0] - 1, rows_per_tensor, dtype=torch.float64
        ).round().to(torch.long)
    reference = unpack_native_fp4(
        packed, scale, device=device, row_indices=row_indices
    )
    metrics = {
        "rtn_g32_fp16_scale": quantize_metric(
            reference, group_size=32, rates=(1.0,), stored_scale_dtype=torch.float16
        ),
        "grid_g32_bf16_scale": quantize_metric(
            reference, group_size=32, rates=rates, stored_scale_dtype=torch.bfloat16
        ),
        "grid_g32_fp16_scale": quantize_metric(
            reference, group_size=32, rates=rates, stored_scale_dtype=torch.float16
        ),
        "rtn_g64_fp16_scale": quantize_metric(
            reference, group_size=64, rates=(1.0,), stored_scale_dtype=torch.float16
        ),
        "grid_g64_fp16_scale": quantize_metric(
            reference, group_size=64, rates=rates, stored_scale_dtype=torch.float16
        ),
    }
    if lloyd_iterations > 0:
        metrics.update({
            "lloyd_g32_fp16_scale": lloyd_quantize_metric(
                reference,
                group_size=32,
                initial_rates=lloyd_initial_rates,
                iterations=lloyd_iterations,
                stored_scale_dtype=torch.float16,
            ),
            "lloyd_g64_fp16_scale": lloyd_quantize_metric(
                reference,
                group_size=64,
                initial_rates=lloyd_initial_rates,
                iterations=lloyd_iterations,
                stored_scale_dtype=torch.float16,
            ),
        })
    return metrics


def worker(
    worker_index: int,
    model_path: str,
    entries: list[dict],
    device_name: str,
    rows_per_tensor: int,
    rates: tuple[float, ...],
    lloyd_initial_rates: tuple[float, ...],
    lloyd_iterations: int,
) -> dict:
    device = torch.device(device_name)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    records: list[dict] = []
    by_shard: dict[str, list[dict]] = defaultdict(list)
    for entry in entries:
        by_shard[entry["shard"]].append(entry)

    started = time.time()
    completed = 0
    for shard_index, (shard, shard_entries) in enumerate(sorted(by_shard.items())):
        with safe_open(Path(model_path) / shard, framework="pt", device="cpu") as handle:
            for entry in shard_entries:
                weight = handle.get_tensor(entry["key"])
                scale = handle.get_tensor(entry["scale_key"])
                metrics = audit_tensor(
                    weight,
                    scale,
                    device=device,
                    rows_per_tensor=rows_per_tensor,
                    rates=rates,
                    lloyd_initial_rates=lloyd_initial_rates,
                    lloyd_iterations=lloyd_iterations,
                )
                records.append(
                    {
                        **entry,
                        "weight_shape_packed": list(weight.shape),
                        "scale_shape": list(scale.shape),
                        "scale_dtype": str(scale.dtype),
                        "sampled_rows": min(rows_per_tensor, weight.shape[0])
                        if rows_per_tensor > 0 else weight.shape[0],
                        "schemes": {
                            name: metric_record(
                                metric,
                                lloyd_initial_rates if name.startswith("lloyd_") else rates,
                            )
                            for name, metric in metrics.items()
                        },
                    }
                )
                completed += 1
                del weight, scale, metrics
        if device.type == "cuda":
            torch.cuda.empty_cache()
        print(
            f"worker={worker_index} device={device_name} shards="
            f"{shard_index + 1}/{len(by_shard)} tensors={completed}/{len(entries)} "
            f"elapsed={time.time() - started:.1f}s",
            flush=True,
        )
    return {"records": records, "elapsed_seconds": time.time() - started}


def aggregate_records(records: list[dict], rates: tuple[float, ...]) -> dict:
    # Aggregate directly from additive quantities.  Store dot/reconstruction_sq
    # in tensor records internally before this function if exact global cosine
    # becomes important; relative MSE and SNR are the primary audit metrics.
    buckets: dict[str, dict[str, Metric]] = {
        "global": defaultdict(Metric),
        "by_layer": defaultdict(Metric),
        "by_projection": defaultdict(Metric),
    }
    for record in records:
        for scheme, payload in record["schemes"].items():
            for category, key in (
                ("global", scheme),
                ("by_layer", f"{record['layer']}::{scheme}"),
                ("by_projection", f"{record['projection']}::{scheme}"),
            ):
                metric = buckets[category][key]
                metric.elements += payload["elements"]
                metric.groups += payload["groups"]
                metric.signal_sq += payload["signal_sq"]
                metric.error_sq += payload["error_sq"]
                metric.max_abs_error = max(metric.max_abs_error, payload["max_abs_error"])
                # Candidate-choice histograms are meaningful in per-tensor
                # records.  Global summaries may combine schemes with distinct
                # candidate sets (grid versus Lloyd), so omit them here.

    def serialize(category: str, *, split: bool) -> dict:
        result: dict = {}
        for key, metric in buckets[category].items():
            if split:
                group, scheme = key.split("::", 1)
                result.setdefault(group, {})[scheme] = metric.as_dict(rates)
            else:
                result[key] = metric.as_dict(rates)
            # Aggregate cosine is not reconstructed from rounded JSON fields.
            if split:
                result[group][scheme].pop("cosine", None)
            else:
                result[key].pop("cosine", None)
        return result

    return {
        "global": serialize("global", split=False),
        "by_layer": serialize("by_layer", split=True),
        "by_projection": serialize("by_projection", split=True),
    }


def main() -> None:
    args = parse_args()
    model_path = Path(args.model).resolve()
    output_path = Path(args.output).resolve()
    records_path = Path(args.tensor_records).resolve() if args.tensor_records else None
    rates = tuple(float(value) for value in args.rates.split(","))
    lloyd_initial_rates = tuple(
        float(value) for value in args.lloyd_initial_rates.split(",")
    )
    if args.lloyd_iterations < 0:
        raise ValueError("--lloyd-iterations must be non-negative")
    layers = parse_int_set(args.layers)
    experts = parse_int_set(args.experts)
    projections = {value.strip() for value in args.projections.split(",") if value.strip()}
    devices = [value.strip() for value in args.devices.split(",") if value.strip()]
    if not devices:
        raise ValueError("at least one device is required")

    with (model_path / "model.safetensors.index.json").open(encoding="utf-8") as handle:
        weight_map = json.load(handle)["weight_map"]
    entries: list[dict] = []
    for key, shard in weight_map.items():
        match = EXPERT_RE.search(key)
        if not match:
            continue
        layer = int(match.group("layer"))
        expert = int(match.group("expert"))
        projection = match.group("projection")
        if layers is not None and layer not in layers:
            continue
        if experts is not None and expert not in experts:
            continue
        if projection not in projections:
            continue
        scale_key = key[:-len("weight")] + "scale"
        if scale_key not in weight_map:
            raise KeyError(f"missing native FP4 scale for {key}: {scale_key}")
        if weight_map[scale_key] != shard:
            raise ValueError(f"weight and scale are in different shards for {key}")
        entries.append(
            {
                "key": key,
                "scale_key": scale_key,
                "shard": shard,
                "layer": layer,
                "expert": expert,
                "projection": projection,
            }
        )
    entries.sort(key=lambda item: (item["shard"], item["key"]))
    if args.max_tensors is not None:
        entries = entries[: args.max_tensors]
    if not entries:
        raise ValueError("no routed-expert tensors matched the requested filters")

    # Keep every shard on one worker so each safetensor is opened exactly once.
    shards = sorted({entry["shard"] for entry in entries})
    shard_owner = {shard: index % len(devices) for index, shard in enumerate(shards)}
    partitions = [[] for _ in devices]
    for entry in entries:
        partitions[shard_owner[entry["shard"]]].append(entry)

    started = time.time()
    results: list[dict] = []
    if len(devices) == 1:
        results.append(
            worker(
                0,
                str(model_path),
                partitions[0],
                devices[0],
                args.rows_per_tensor,
                rates,
                lloyd_initial_rates,
                args.lloyd_iterations,
            )
        )
    else:
        context = multiprocessing.get_context("spawn")
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=len(devices), mp_context=context
        ) as executor:
            futures = [
                executor.submit(
                    worker,
                    index,
                    str(model_path),
                    partition,
                    devices[index],
                    args.rows_per_tensor,
                    rates,
                    lloyd_initial_rates,
                    args.lloyd_iterations,
                )
                for index, partition in enumerate(partitions)
                if partition
            ]
            for future in concurrent.futures.as_completed(futures):
                results.append(future.result())

    records = [record for result in results for record in result["records"]]
    records.sort(key=lambda record: (record["layer"], record["expert"], record["projection"]))
    aggregate = aggregate_records(records, rates)
    outlier_schemes = list(aggregate["global"])
    top_outliers = {
        scheme: sorted(
            (
                {
                    "key": record["key"],
                    "relative_mse": record["schemes"][scheme]["relative_mse"],
                    "snr_db": record["schemes"][scheme]["snr_db"],
                }
                for record in records
            ),
            key=lambda item: item["relative_mse"],
            reverse=True,
        )[: args.top_outliers]
        for scheme in outlier_schemes
    }
    payload = {
        "model": str(model_path),
        "native_layout": {
            "storage": "signed int8 bytes, low E2M1 nibble first",
            "logical_values_per_byte": 2,
            "scale_dtype_observed": sorted({record["scale_dtype"] for record in records}),
            "scale_semantics": "direct E8M0 multiplier",
            "scale_group_size": 32,
            "fp4_table": list(FP4_TABLE),
        },
        "published_grid_rates": list(rates),
        "lloyd": {
            "initial_rates": list(lloyd_initial_rates),
            "iterations": args.lloyd_iterations,
            "selection_includes_stored_fp16_scale_rounding": True,
        },
        "selection_semantics": (
            "MSE selection and INT4 codes use FP32 working scale; reconstructed error "
            "includes the configured BF16/FP16 stored-scale rounding"
        ),
        "filters": {
            "layers": args.layers,
            "experts": args.experts,
            "projections": sorted(projections),
            "rows_per_tensor": args.rows_per_tensor,
            "devices": devices,
        },
        "tensor_count": len(records),
        "elapsed_seconds": time.time() - started,
        "worker_elapsed_seconds": [result["elapsed_seconds"] for result in results],
        **aggregate,
        "top_outliers": top_outliers,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, allow_nan=True)
    if records_path:
        records_path.parent.mkdir(parents=True, exist_ok=True)
        with records_path.open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False, allow_nan=True) + "\n")
    print(json.dumps({
        "output": str(output_path),
        "tensor_records": str(records_path) if records_path else None,
        "tensor_count": len(records),
        "elapsed_seconds": payload["elapsed_seconds"],
        "global": payload["global"],
    }, indent=2, ensure_ascii=False, allow_nan=True))


if __name__ == "__main__":
    main()
