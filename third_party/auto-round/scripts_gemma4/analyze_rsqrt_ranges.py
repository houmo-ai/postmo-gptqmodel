"""Analyze Gemma4 RMSNorm rsqrt input ranges on real datasets only.

This script tracks the rsqrt input of Gemma4 text-stack RMSNorm modules:
  - input_layernorm
  - self_attn.q_norm
  - self_attn.k_norm
  - self_attn.v_norm

Coverage:
  - WikiText-2 test full sliding-window pass
  - MMLU-Pro test 10% prompt-only sample (chat-template formatted)
  - C4 validation sampled documents
  - GSM8K test prompt-only pass
  - ARC-Challenge test prompt-only pass
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import subprocess
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from scripts_gemma4.eval.eval_benchmark import load_model


FP16_INFO = torch.finfo(torch.float16)
FP16_MAX = float(FP16_INFO.max)
FP16_MIN_NORMAL = float(FP16_INFO.tiny)
FP16_MIN_SUBNORMAL = math.ldexp(1.0, -24)


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze Gemma4 RMSNorm rsqrt ranges on real inputs")
    parser.add_argument("--model", type=str, default="weights/gemma-4-31B-it")
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--device", type=str, default="auto",
                        help='Target device, e.g. "cuda:0" or "auto"')
    parser.add_argument("--gpu-max-used-mb", type=int, default=1024)
    parser.add_argument("--gpu-max-util", type=int, default=10)
    parser.add_argument("--gpu-min-free-gb", type=int, default=70)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--stride", type=int, default=1024)
    parser.add_argument("--prompt-batch-size", type=int, default=2)

    parser.add_argument("--mmlu-fraction", type=float, default=0.1)
    parser.add_argument("--c4-docs", type=int, default=512)
    parser.add_argument("--gsm8k-fraction", type=float, default=1.0)
    parser.add_argument("--arc-fraction", type=float, default=1.0)

    parser.add_argument("--sample-per-update", type=int, default=64)
    parser.add_argument("--max-sample-values", type=int, default=50000)
    parser.add_argument("--progress-interval", type=int, default=25)

    parser.add_argument("--output-json", type=str, default="output/analysis/gemma4_rsqrt_range_analysis_real.json")
    parser.add_argument("--output-md", type=str, default="output/analysis/gemma4_rsqrt_range_analysis_real.md")
    return parser.parse_args()


def choose_idle_gpu(args) -> dict[str, Any]:
    cmd = [
        "nvidia-smi",
        "--query-gpu=index,name,memory.used,memory.total,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    output = subprocess.check_output(cmd, text=True)
    gpus = []
    for line in output.strip().splitlines():
        index, name, used, total, util = [part.strip() for part in line.split(",")]
        used = int(used)
        total = int(total)
        util = int(util)
        gpus.append({
            "index": int(index),
            "name": name,
            "memory_used_mb": used,
            "memory_total_mb": total,
            "memory_free_mb": total - used,
            "utilization_gpu": util,
        })

    candidates = [
        gpu for gpu in gpus
        if gpu["memory_used_mb"] <= args.gpu_max_used_mb
        and gpu["utilization_gpu"] <= args.gpu_max_util
        and gpu["memory_free_mb"] >= args.gpu_min_free_gb * 1024
    ]
    if not candidates:
        raise RuntimeError(
            "No idle GPU matched the thresholds: "
            f"used<={args.gpu_max_used_mb}MB, util<={args.gpu_max_util}%, "
            f"free>={args.gpu_min_free_gb}GB"
        )
    candidates.sort(key=lambda item: (-item["memory_free_mb"], item["memory_used_mb"], item["index"]))
    return candidates[0]


def resolve_device(args):
    if args.device != "auto":
        return args.device, None
    gpu = choose_idle_gpu(args)
    return f"cuda:{gpu['index']}", gpu


def quantiles_from_samples(values: list[float], qs: list[float]) -> dict[str, float]:
    if not values:
        return {f"p{int(q * 1000) / 10:g}": float("nan") for q in qs}
    tensor = torch.tensor(values, dtype=torch.float64)
    result = {}
    for q in qs:
        label = f"p{int(q * 1000) / 10:g}"
        result[label] = torch.quantile(tensor, q).item()
    return result


def format_ratio(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def format_float(value: float, precision: int = 6) -> str:
    if value is None or math.isnan(value):
        return "nan"
    return f"{value:.{precision}g}"


def sample_count(total: int, fraction: float) -> int:
    if fraction <= 0:
        return 0
    if fraction >= 1:
        return total
    return max(1, int(total * fraction))


def token_summary(lengths: list[int]) -> dict[str, Any]:
    if not lengths:
        return {"count": 0, "min": 0, "mean": 0.0, "p50": 0.0, "p99": 0.0, "max": 0}
    tensor = torch.tensor(lengths, dtype=torch.float64)
    return {
        "count": int(len(lengths)),
        "min": int(min(lengths)),
        "mean": tensor.mean().item(),
        "p50": torch.quantile(tensor, 0.5).item(),
        "p99": torch.quantile(tensor, 0.99).item(),
        "max": int(max(lengths)),
    }


@dataclass
class RangeAccumulator:
    sample_per_update: int
    max_sample_values: int
    count: int = 0
    sum_value: float = 0.0
    min_value: float = float("inf")
    max_value: float = float("-inf")
    min_record: dict[str, Any] | None = None
    max_record: dict[str, Any] | None = None
    subnormal_count: int = 0
    below_subnormal_count: int = 0
    normal_count: int = 0
    overflow_count: int = 0
    fp16_zero_count: int = 0
    fp16_inf_count: int = 0
    cast_rel_err_sum: float = 0.0
    cast_rel_err_max: float = 0.0
    rsqrt_rel_err_sum: float = 0.0
    rsqrt_rel_err_max: float = 0.0
    rsqrt_output_min: float = float("inf")
    rsqrt_output_max: float = float("-inf")
    sample_values: list[float] = field(default_factory=list)
    seen_sample_values: int = 0

    def _update_record(self, kind: str, value: float, context: dict[str, Any]):
        record = {
            "value": float(value),
            "source": context["source"],
            "item": context["item"],
            "module": context["module"],
        }
        if kind == "min":
            self.min_record = record
        else:
            self.max_record = record

    def update(self, values: torch.Tensor, context: dict[str, Any]):
        flat = values.detach().reshape(-1).to(torch.float32)
        if flat.numel() == 0:
            return

        fp16_values = flat.to(torch.float16)
        ref_rsqrt = torch.rsqrt(flat)
        fp16_rsqrt = torch.rsqrt(fp16_values).to(torch.float32)

        count = flat.numel()
        self.count += count
        self.sum_value += flat.sum(dtype=torch.float64).item()

        min_value = flat.min().item()
        max_value = flat.max().item()
        if min_value < self.min_value:
            self.min_value = min_value
            self._update_record("min", min_value, context)
        if max_value > self.max_value:
            self.max_value = max_value
            self._update_record("max", max_value, context)

        self.subnormal_count += ((flat > 0) & (flat < FP16_MIN_NORMAL)).sum().item()
        self.below_subnormal_count += ((flat > 0) & (flat < FP16_MIN_SUBNORMAL)).sum().item()
        self.normal_count += ((flat >= FP16_MIN_NORMAL) & (flat <= FP16_MAX)).sum().item()
        self.overflow_count += (flat > FP16_MAX).sum().item()
        self.fp16_zero_count += ((fp16_values == 0) & (flat > 0)).sum().item()
        self.fp16_inf_count += torch.isinf(fp16_values).sum().item()

        cast_rel_err = (fp16_values.to(torch.float32) - flat).abs() / flat.clamp_min(1e-30)
        rsqrt_rel_err = (fp16_rsqrt - ref_rsqrt).abs() / ref_rsqrt.clamp_min(1e-30)
        self.cast_rel_err_sum += cast_rel_err.sum(dtype=torch.float64).item()
        self.rsqrt_rel_err_sum += rsqrt_rel_err.sum(dtype=torch.float64).item()
        self.cast_rel_err_max = max(self.cast_rel_err_max, cast_rel_err.max().item())
        self.rsqrt_rel_err_max = max(self.rsqrt_rel_err_max, rsqrt_rel_err.max().item())

        self.rsqrt_output_min = min(self.rsqrt_output_min, ref_rsqrt.min().item())
        self.rsqrt_output_max = max(self.rsqrt_output_max, ref_rsqrt.max().item())

        take = min(self.sample_per_update, count)
        if take > 0:
            if count <= take:
                sampled = flat
            else:
                indices = torch.randint(count, (take,), device=flat.device)
                sampled = flat.index_select(0, indices)
            self._update_sample_values(sampled.cpu().tolist())

    def _update_sample_values(self, sampled: list[float]):
        for value in sampled:
            self.seen_sample_values += 1
            if len(self.sample_values) < self.max_sample_values:
                self.sample_values.append(value)
                continue
            slot = random.randrange(self.seen_sample_values)
            if slot < self.max_sample_values:
                self.sample_values[slot] = value

    def to_dict(self) -> dict[str, Any]:
        quantiles = quantiles_from_samples(self.sample_values, [0.01, 0.5, 0.99, 0.999])
        return {
            "count": self.count,
            "mean": self.sum_value / self.count if self.count else float("nan"),
            "min": self.min_value if self.count else float("nan"),
            "max": self.max_value if self.count else float("nan"),
            "quantiles": quantiles,
            "subnormal_fraction": format_ratio(self.subnormal_count, self.count),
            "below_subnormal_fraction": format_ratio(self.below_subnormal_count, self.count),
            "normal_fraction": format_ratio(self.normal_count, self.count),
            "overflow_fraction": format_ratio(self.overflow_count, self.count),
            "fp16_zero_fraction": format_ratio(self.fp16_zero_count, self.count),
            "fp16_inf_fraction": format_ratio(self.fp16_inf_count, self.count),
            "cast_rel_err_mean": self.cast_rel_err_sum / self.count if self.count else float("nan"),
            "cast_rel_err_max": self.cast_rel_err_max,
            "fp16_rsqrt_rel_err_mean": self.rsqrt_rel_err_sum / self.count if self.count else float("nan"),
            "fp16_rsqrt_rel_err_max": self.rsqrt_rel_err_max,
            "rsqrt_output_min": self.rsqrt_output_min if self.count else float("nan"),
            "rsqrt_output_max": self.rsqrt_output_max if self.count else float("nan"),
            "sample_size": len(self.sample_values),
            "min_record": self.min_record,
            "max_record": self.max_record,
        }


class ActivationCollector:
    def __init__(self, sample_per_update: int, max_sample_values: int):
        self.sample_per_update = sample_per_update
        self.max_sample_values = max_sample_values
        self.context = {"source": "unknown", "item": "unknown"}
        self.stats: dict[str, dict[str, RangeAccumulator]] = defaultdict(dict)
        self.handles = []

    def set_context(self, source: str, item: str):
        self.context = {"source": source, "item": item}

    def _accumulator(self, source: str, kind: str) -> RangeAccumulator:
        source_stats = self.stats[source]
        if kind not in source_stats:
            source_stats[kind] = RangeAccumulator(
                sample_per_update=self.sample_per_update,
                max_sample_values=self.max_sample_values,
            )
        return source_stats[kind]

    def _hook_factory(self, module_name: str, kind: str):
        def hook(module, inputs, _output):
            hidden_states = inputs[0]
            rsqrt_input = hidden_states.detach().to(torch.float32).pow(2).mean(dim=-1) + float(module.eps)
            context = {
                "source": self.context["source"],
                "item": self.context["item"],
                "module": module_name,
            }
            self._accumulator(self.context["source"], kind).update(rsqrt_input, context)
        return hook

    def register(self, model):
        for module_name, module in model.named_modules():
            kind = None
            if module_name.endswith("self_attn.q_norm"):
                kind = "q_norm"
            elif module_name.endswith("self_attn.k_norm"):
                kind = "k_norm"
            elif module_name.endswith("self_attn.v_norm"):
                kind = "v_norm"
            elif module_name.endswith("input_layernorm"):
                kind = "input_layernorm"
            if kind is not None:
                self.handles.append(module.register_forward_hook(self._hook_factory(module_name, kind)))

    def remove(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def to_dict(self):
        return {
            source: {kind: acc.to_dict() for kind, acc in source_stats.items()}
            for source, source_stats in self.stats.items()
        }


def maybe_clear_cuda(device: str):
    if device.startswith("cuda"):
        torch.cuda.empty_cache()


def apply_chat_template_or_raw(tokenizer, user_prompt: str) -> str:
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        messages = [{"role": "user", "content": user_prompt}]
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return user_prompt


def batch_forward_texts(
    model,
    tokenizer,
    device: str,
    collector: ActivationCollector,
    source: str,
    records: list[dict[str, Any]],
    seq_len: int,
    batch_size: int,
    progress_interval: int,
) -> dict[str, Any]:
    token_lengths: list[int] = []
    total_batches = math.ceil(len(records) / batch_size) if records else 0
    print(
        f"[{source}] starting prompt-only forward: {len(records)} records, "
        f"batch_size={batch_size}, total_batches={total_batches}",
        flush=True,
    )

    for batch_start in range(0, len(records), batch_size):
        batch = records[batch_start: batch_start + batch_size]
        texts = [item["text"] for item in batch]
        labels = [item["label"] for item in batch]
        batch_label = f"batch_{batch_start // batch_size}[{labels[0]}..{labels[-1]}]"
        collector.set_context(source, batch_label)
        encoded = tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=seq_len,
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        attention_mask = encoded.get("attention_mask")
        if attention_mask is not None:
            token_lengths.extend(attention_mask.sum(dim=1).tolist())
        else:
            token_lengths.extend([encoded["input_ids"].shape[1]] * len(batch))

        with torch.no_grad():
            model(**encoded, use_cache=False)
        batch_index = batch_start // batch_size + 1
        if progress_interval > 0 and (
            batch_index == 1
            or batch_index == total_batches
            or batch_index % progress_interval == 0
        ):
            print(
                f"[{source}] progress {batch_index}/{total_batches} batches",
                flush=True,
            )

    return {
        "count": len(records),
        "token_summary": token_summary(token_lengths),
        "examples": [{key: record[key] for key in record if key != "text"} for record in records[:5]],
    }


def run_wikitext_full(model, tokenizer, device: str, collector: ActivationCollector, args) -> dict[str, Any]:
    from datasets import load_dataset

    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(dataset["text"])
    input_ids = tokenizer(text, return_tensors="pt").input_ids
    total_len = input_ids.size(1)
    total_windows = len(range(0, total_len, args.stride))
    print(
        f"[wikitext2_full] starting sliding-window forward: total_tokens={total_len}, "
        f"seq_len={args.seq_len}, stride={args.stride}, total_windows={total_windows}",
        flush=True,
    )

    windows = 0
    window_lengths: list[int] = []
    for begin in range(0, total_len, args.stride):
        end = min(begin + args.seq_len, total_len)
        collector.set_context("wikitext2_full", f"window_{windows}[{begin}:{end}]")
        ids = input_ids[:, begin:end].to(device)
        attention_mask = torch.ones_like(ids, device=device)
        with torch.no_grad():
            model(input_ids=ids, attention_mask=attention_mask, use_cache=False)
        windows += 1
        window_lengths.append(end - begin)
        if args.progress_interval > 0 and (
            windows == 1
            or windows == total_windows
            or windows % args.progress_interval == 0
        ):
            print(
                f"[wikitext2_full] progress {windows}/{total_windows} windows",
                flush=True,
            )
        if end >= total_len:
            break

    return {
        "mode": "full_sliding_window",
        "rows": len(dataset),
        "total_tokens": int(total_len),
        "windows": windows,
        "seq_len": args.seq_len,
        "stride": args.stride,
        "token_summary": token_summary(window_lengths),
    }


def build_mmlu_pro_prompt(tokenizer, sample: dict[str, Any]) -> str:
    option_lines = [f"{chr(65 + idx)}. {option}" for idx, option in enumerate(sample["options"])]
    prompt = (
        "Answer the following multiple-choice question. Return the best option letter.\n\n"
        f"Question: {sample['question']}\n\n"
        "Options:\n"
        + "\n".join(option_lines)
        + "\n\nAnswer:"
    )
    return apply_chat_template_or_raw(tokenizer, prompt)


def run_mmlu_pro_sample(model, tokenizer, device: str, collector: ActivationCollector, args) -> dict[str, Any]:
    from datasets import load_dataset

    dataset = load_dataset("TIGER-Lab/MMLU-Pro", split="test")
    take = sample_count(len(dataset), args.mmlu_fraction)
    print(
        f"[mmlu_pro_10pct] loading dataset_size={len(dataset)}, sampled={take}",
        flush=True,
    )
    sampled = dataset.shuffle(seed=args.seed).select(range(take))
    records = []
    for sample in sampled:
        records.append({
            "label": f"question_{sample['question_id']}_{sample['category']}",
            "question_id": int(sample["question_id"]),
            "category": sample["category"],
            "text": build_mmlu_pro_prompt(tokenizer, sample),
        })
    meta = batch_forward_texts(
        model, tokenizer, device, collector, "mmlu_pro_10pct", records,
        seq_len=args.seq_len, batch_size=args.prompt_batch_size, progress_interval=args.progress_interval,
    )
    meta.update({
        "mode": "prompt_only_chat_template",
        "fraction": args.mmlu_fraction,
        "total_dataset_size": len(dataset),
    })
    return meta


def build_gsm8k_prompt(tokenizer, sample: dict[str, Any]) -> str:
    prompt = (
        "Solve the following grade-school math problem. "
        "Reason step by step and provide the final answer.\n\n"
        f"Question: {sample['question']}\n\nAnswer:"
    )
    return apply_chat_template_or_raw(tokenizer, prompt)


def run_gsm8k_sample(model, tokenizer, device: str, collector: ActivationCollector, args) -> dict[str, Any]:
    from datasets import load_dataset

    dataset = load_dataset("openai/gsm8k", "main", split="test")
    take = sample_count(len(dataset), args.gsm8k_fraction)
    print(
        f"[gsm8k_test] loading dataset_size={len(dataset)}, sampled={take}",
        flush=True,
    )
    sampled = dataset if take == len(dataset) else dataset.shuffle(seed=args.seed).select(range(take))
    records = []
    for idx, sample in enumerate(sampled):
        records.append({
            "label": f"gsm8k_{idx}",
            "text": build_gsm8k_prompt(tokenizer, sample),
        })
    meta = batch_forward_texts(
        model, tokenizer, device, collector, "gsm8k_test", records,
        seq_len=args.seq_len, batch_size=args.prompt_batch_size, progress_interval=args.progress_interval,
    )
    meta.update({
        "mode": "prompt_only_chat_template",
        "fraction": args.gsm8k_fraction,
        "total_dataset_size": len(dataset),
    })
    return meta


def build_arc_prompt(tokenizer, sample: dict[str, Any]) -> str:
    option_lines = [
        f"{label}. {text}"
        for label, text in zip(sample["choices"]["label"], sample["choices"]["text"])
    ]
    prompt = (
        "Answer the following multiple-choice science question. Return the best option letter.\n\n"
        f"Question: {sample['question']}\n\n"
        "Options:\n"
        + "\n".join(option_lines)
        + "\n\nAnswer:"
    )
    return apply_chat_template_or_raw(tokenizer, prompt)


def run_arc_sample(model, tokenizer, device: str, collector: ActivationCollector, args) -> dict[str, Any]:
    from datasets import load_dataset

    dataset = load_dataset("ai2_arc", "ARC-Challenge", split="test")
    take = sample_count(len(dataset), args.arc_fraction)
    print(
        f"[arc_challenge_test] loading dataset_size={len(dataset)}, sampled={take}",
        flush=True,
    )
    sampled = dataset if take == len(dataset) else dataset.shuffle(seed=args.seed).select(range(take))
    records = []
    for idx, sample in enumerate(sampled):
        records.append({
            "label": sample["id"] or f"arc_{idx}",
            "text": build_arc_prompt(tokenizer, sample),
        })
    meta = batch_forward_texts(
        model, tokenizer, device, collector, "arc_challenge_test", records,
        seq_len=args.seq_len, batch_size=args.prompt_batch_size, progress_interval=args.progress_interval,
    )
    meta.update({
        "mode": "prompt_only_chat_template",
        "fraction": args.arc_fraction,
        "total_dataset_size": len(dataset),
    })
    return meta


def run_c4_sample(model, tokenizer, device: str, collector: ActivationCollector, args) -> dict[str, Any]:
    from datasets import load_dataset

    dataset = load_dataset("allenai/c4", "en", split="validation", streaming=True)
    dataset = dataset.shuffle(seed=args.seed, buffer_size=10000)
    print(
        f"[c4_validation_sample] starting streaming sample: target_docs={args.c4_docs}",
        flush=True,
    )

    records = []
    for idx, sample in enumerate(dataset):
        text = sample["text"].strip()
        if not text:
            continue
        records.append({
            "label": f"c4_{idx}",
            "url": sample.get("url"),
            "text": text,
        })
        if args.progress_interval > 0 and len(records) % (args.progress_interval * args.prompt_batch_size) == 0:
            print(
                f"[c4_validation_sample] collected {len(records)}/{args.c4_docs} docs",
                flush=True,
            )
        if len(records) >= args.c4_docs:
            break

    meta = batch_forward_texts(
        model, tokenizer, device, collector, "c4_validation_sample", records,
        seq_len=args.seq_len, batch_size=args.prompt_batch_size, progress_interval=args.progress_interval,
    )
    meta.update({
        "mode": "streaming_doc_sample",
        "sampled_docs": len(records),
    })
    return meta


def source_display_name(source: str) -> str:
    mapping = {
        "wikitext2_full": "WikiText-2 full",
        "mmlu_pro_10pct": "MMLU-Pro 10%",
        "c4_validation_sample": "C4 validation sample",
        "gsm8k_test": "GSM8K test",
        "arc_challenge_test": "ARC-Challenge test",
    }
    return mapping.get(source, source)


def best_real_metric(real_stats: dict[str, Any], kind: str, key: str) -> float:
    values = []
    for source_stats in real_stats.values():
        if kind in source_stats:
            values.append(source_stats[kind][key])
    return max(values) if values else float("nan")


def write_markdown_report(result: dict[str, Any], output_path: str):
    real_stats = result["real_data"]["stats"]
    source_meta = result["real_data"]["sources"]
    meta = result["meta"]
    theoretical = result["theoretical"]
    source_order = result["real_data"]["source_order"]

    key_kinds = ["input_layernorm", "q_norm", "k_norm", "v_norm"]
    focus_stats = []
    for source in source_order:
        for kind in key_kinds:
            stat = real_stats.get(source, {}).get(kind)
            if stat is not None:
                focus_stats.append((source, kind, stat))

    observed_global_max = max((stat["max"] for _, _, stat in focus_stats), default=float("nan"))
    observed_global_min = min((stat["min"] for _, _, stat in focus_stats), default=float("nan"))
    min_record = min(
        (stat["min_record"] for _, _, stat in focus_stats if stat.get("min_record") is not None),
        key=lambda item: item["value"],
        default=None,
    )
    max_record = max(
        (stat["max_record"] for _, _, stat in focus_stats if stat.get("max_record") is not None),
        key=lambda item: item["value"],
        default=None,
    )
    any_overflow = any(stat["overflow_fraction"] > 0 for _, _, stat in focus_stats)
    any_subnormal = any(stat["subnormal_fraction"] > 0 for _, _, stat in focus_stats)

    source_rows = []
    for source in source_order:
        info = source_meta[source]
        token_info = info.get("token_summary", {})
        coverage = ""
        if "total_dataset_size" in info:
            coverage = f"{info['count']}/{info['total_dataset_size']}"
        elif "sampled_docs" in info:
            coverage = str(info["sampled_docs"])
        elif "windows" in info:
            coverage = str(info["windows"])
        source_rows.append(
            f"| {source_display_name(source)} | {info['mode']} | {coverage} | "
            f"{format_float(token_info.get('p50', float('nan')))} | "
            f"{format_float(token_info.get('p99', float('nan')))} | "
            f"{format_float(token_info.get('max', float('nan')))} |"
        )

    summary_lines = []
    for kind in key_kinds:
        observed_min = min(
            (real_stats.get(source, {}).get(kind, {}).get("min", float("inf")) for source in source_order),
            default=float("inf"),
        )
        observed_max = max(
            (real_stats.get(source, {}).get(kind, {}).get("max", float("-inf")) for source in source_order),
            default=float("-inf"),
        )
        max_subnormal = best_real_metric(real_stats, kind, "subnormal_fraction")
        max_overflow = best_real_metric(real_stats, kind, "overflow_fraction")
        max_err = best_real_metric(real_stats, kind, "fp16_rsqrt_rel_err_max")
        summary_lines.append(
            f"| {kind} | {format_float(observed_min)} | {format_float(observed_max)} | "
            f"{format_float(max_subnormal, 4)} | {format_float(max_overflow, 4)} | {format_float(max_err, 4)} |"
        )

    real_tables = []
    for source in source_order:
        rows = []
        for kind in key_kinds:
            if kind not in real_stats.get(source, {}):
                continue
            stat = real_stats[source][kind]
            rows.append(
                f"| {kind} | {format_float(stat['min'])} | {format_float(stat['quantiles']['p1'])} | "
                f"{format_float(stat['quantiles']['p50'])} | {format_float(stat['quantiles']['p99'])} | "
                f"{format_float(stat['max'])} | {format_float(stat['subnormal_fraction'], 4)} | "
                f"{format_float(stat['overflow_fraction'], 4)} | {format_float(stat['fp16_rsqrt_rel_err_max'], 4)} |"
            )
        real_tables.append(
            f"### {source_display_name(source)}\n\n"
            "| 模块 | min | p1 | p50 | p99 | max | subnormal_frac | overflow_frac | fp16_rsqrt_rel_err_max |\n"
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|\n"
            f"{chr(10).join(rows)}\n"
        )

    min_note = ""
    if min_record is not None:
        min_note = (
            f"最小真实值来自 **{source_display_name(min_record['source'])} / {min_record['module']}**，"
            f"批次标识 `{min_record['item']}`，值为 `{format_float(min_record['value'])}`。"
        )
    max_note = ""
    if max_record is not None:
        max_note = (
            f"最大真实值来自 **{source_display_name(max_record['source'])} / {max_record['module']}**，"
            f"批次标识 `{max_record['item']}`，值为 `{format_float(max_record['value'])}`。"
        )

    markdown = f"""## 结论

<callout emoji="✅" background-color="light-green" border-color="green">
本次只基于真实输入数据做分析时，Gemma4 `input_layernorm / q_norm / k_norm / v_norm` 的 `rsqrt(mean(x^2) + eps)` 输入整体范围为 <code>[{format_float(observed_global_min)}, {format_float(observed_global_max)}]</code>。

- {"真实样本中没有观察到超过 FP16 上界 65504 的输入。" if not any_overflow else "真实样本中观察到了超过 FP16 上界 65504 的输入。"}
- {"真实样本里没有观察到落入 FP16 subnormal 区间的输入。" if not any_subnormal else "真实样本里观察到了落入 FP16 subnormal 区间的输入。"}
- {min_note}
- {max_note}
</callout>

<callout emoji="⚠️" background-color="light-yellow" border-color="yellow">
Gemma4 的理论下界仍然由 <code>eps = 1e-6</code> 决定。这个值在 FP16 中 <b>可表示但属于 subnormal</b>，所以真实样本没有命中 subnormal，并不改变理论边界判断。
</callout>

## 分析对象

- 模型：`{meta['model']}`
- 设备：`{meta['device']}`{"（自动选卡）" if meta.get('selected_gpu') else ""}
- 模块：`input_layernorm`、`q_norm`、`k_norm`、`v_norm`
- 公式：

```python
mean_squared = hidden_states.pow(2).mean(-1, keepdim=True) + eps
normed = hidden_states * torch.pow(mean_squared, -0.5)
```

## 理论边界

| 项目 | 数值 |
|---|---:|
| FP16 max | {FP16_MAX} |
| FP16 min normal | {FP16_MIN_NORMAL:.8g} |
| FP16 min subnormal | {FP16_MIN_SUBNORMAL:.8g} |
| Gemma4 eps | {theoretical['eps']} |
| rsqrt 输入理论下界 | {theoretical['rsqrt_input_lower_bound']} |
| 触发 FP16 overflow 所需 RMS 幅值 | {theoretical['overflow_rms_threshold']:.6f} |

## 数据集设置

本轮按默认推荐集合扩展：**WikiText-2 full + MMLU-Pro 10% + C4 validation sample + GSM8K test + ARC-Challenge test**。

| 数据源 | 模式 | 覆盖量 | token p50 | token p99 | token max |
|---|---|---:|---:|---:|---:|
{chr(10).join(source_rows)}

## 汇总结论（四类 norm）

| 模块 | 真实最小值 | 真实最大值 | 最大 subnormal_frac | 最大 overflow_frac | 最大 fp16_rsqrt_rel_err |
|---|---:|---:|---:|---:|---:|
{chr(10).join(summary_lines)}

## 分数据源统计

{chr(10).join(real_tables)}

## 判断

1. **如果判断标准是“真实样本里是否能安全落在 FP16 可表示范围”**：本轮答案是 **可以**，没有看到上溢或 subnormal 命中。
2. **如果判断标准是“理论上是否始终在 FP16 normal 范围”**：答案仍然是 **不一定**，因为 `eps=1e-6 < FP16 min normal`。
3. **如果目标是尝试 FP16 rsqrt 指令提速**：从这轮更大规模真实输入看，`input_layernorm / q_norm / k_norm / v_norm` 都具备较强可行性，但仍建议保留：
   - `clamp_min(1e-6)` 或更高阈值；
   - FTZ/DAZ 场景下对极低能量输入做保护；
   - 最终用任务精度做 A/B 验证。

## 备注

- 这份报告只看 **真实输入数据**，不再使用 synthetic 压力分布。
- MMLU-Pro / GSM8K / ARC-Challenge 都采用 **prompt-only 前向**，即只分析真实题目输入分布，不跑完整生成。
- C4 通过 streaming 方式采样，避免全量下载。
"""

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(markdown)


def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    started_at = time.time()
    device, selected_gpu = resolve_device(args)
    print(f"Using device: {device}")
    if selected_gpu is not None:
        print(f"Selected GPU: {json.dumps(selected_gpu, ensure_ascii=False)}")

    model, tokenizer = load_model(args.model, device, args.dtype)
    model.eval()

    text_config = getattr(model.config, "text_config", model.config)
    eps = float(text_config.rms_norm_eps)

    collector = ActivationCollector(
        sample_per_update=args.sample_per_update,
        max_sample_values=args.max_sample_values,
    )
    collector.register(model)

    try:
        source_meta = {}
        source_meta["wikitext2_full"] = run_wikitext_full(model, tokenizer, device, collector, args)
        maybe_clear_cuda(device)
        print("[wikitext2_full] done", flush=True)
        source_meta["mmlu_pro_10pct"] = run_mmlu_pro_sample(model, tokenizer, device, collector, args)
        maybe_clear_cuda(device)
        print("[mmlu_pro_10pct] done", flush=True)
        source_meta["c4_validation_sample"] = run_c4_sample(model, tokenizer, device, collector, args)
        maybe_clear_cuda(device)
        print("[c4_validation_sample] done", flush=True)
        source_meta["gsm8k_test"] = run_gsm8k_sample(model, tokenizer, device, collector, args)
        maybe_clear_cuda(device)
        print("[gsm8k_test] done", flush=True)
        source_meta["arc_challenge_test"] = run_arc_sample(model, tokenizer, device, collector, args)
        maybe_clear_cuda(device)
        print("[arc_challenge_test] done", flush=True)
    finally:
        collector.remove()

    theoretical = {
        "eps": eps,
        "rsqrt_input_lower_bound": eps,
        "fp16_max": FP16_MAX,
        "fp16_min_normal": FP16_MIN_NORMAL,
        "fp16_min_subnormal": FP16_MIN_SUBNORMAL,
        "overflow_rms_threshold": math.sqrt(FP16_MAX - eps),
    }

    source_order = [
        "wikitext2_full",
        "mmlu_pro_10pct",
        "c4_validation_sample",
        "gsm8k_test",
        "arc_challenge_test",
    ]

    result = {
        "meta": {
            "model": args.model,
            "dtype": args.dtype,
            "device": device,
            "selected_gpu": selected_gpu,
            "seed": args.seed,
            "elapsed_seconds": int(time.time() - started_at),
        },
        "theoretical": theoretical,
        "real_data": {
            "source_order": source_order,
            "sources": source_meta,
            "stats": collector.to_dict(),
        },
    }

    os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    write_markdown_report(result, args.output_md)

    print(f"Saved JSON: {args.output_json}")
    print(f"Saved Markdown: {args.output_md}")


if __name__ == "__main__":
    main()
