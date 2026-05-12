import argparse
import copy
import gc
import json
import os
import random
import time
from pathlib import Path
from typing import Dict, List

import torch
from datasets import load_dataset
from transformers import AutoTokenizer

from gptqmodel import BACKEND, GPTQModel, QuantizeConfig
from gptqmodel.quantization import FORMAT
from gptqmodel.quantization.config import AutoRoundVersion
from gptqmodel.quantization.mix_precision import run_mix_precision_pipeline
from gptqmodel.utils.calibration import prepare_calibration_dataset
from gptqmodel.utils.perplexity import Perplexity


os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def log(message: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run one mix-precision search and reuse the allocation for GPTQ and AutoRound quantization."
    )
    parser.add_argument("--model", type=str, default="/data02/datasets/Qwen3-8B", help="native model path")
    parser.add_argument(
        "--output-root",
        type=str,
        default="output/qwen3_8b_mix_precision_full",
        help="directory used for search artifacts, quantized models, and summary json",
    )
    parser.add_argument("--dataset-path", type=str, default="wikitext", help="dataset path")
    parser.add_argument("--dataset-name", type=str, default="wikitext-2-raw-v1", help="dataset name")
    parser.add_argument("--calibration-split", type=str, default="train", help="split used for quantization calibration")
    parser.add_argument("--ppl-split", type=str, default="test", help="split used for perplexity evaluation")
    parser.add_argument("--text-column", type=str, default="text", help="dataset text column")
    parser.add_argument(
        "--calibration-samples",
        type=int,
        default=64,
        help="number of raw calibration rows; set to 0 to use the whole split after filtering",
    )
    parser.add_argument("--calibration-min-chars", type=int, default=200, help="minimum chars kept per raw row")
    parser.add_argument("--calibration-concat-size", type=int, default=512, help="target concat token length")
    parser.add_argument(
        "--calibration-sort",
        type=str,
        choices=["asc", "desc", "shuffle"],
        default="desc",
        help="sort order before batching calibration data",
    )
    parser.add_argument("--calibration-batch-size", type=int, default=4, help="batch size for search and quantization")
    parser.add_argument("--calibration-data-min-length", type=int, default=10, help="minimum token length kept")
    parser.add_argument("--bits", type=int, default=4, choices=[2, 3, 4, 8], help="base quantization bit width")
    parser.add_argument("--group-size", type=int, default=64, help="group size")
    parser.add_argument("--mix-weight-bits", nargs="+", type=int, default=[4, 8], help="candidate weight bits")
    parser.add_argument("--mix-policy", type=str, choices=["topk", "threshold"], default="topk")
    parser.add_argument("--mix-topk", type=float, default=0.1)
    parser.add_argument("--mix-threshold", type=float, default=1e-6)
    parser.add_argument(
        "--mix-metric",
        type=str,
        choices=["kl", "sqnr", "l1", "mse", "ppl", "perplexity"],
        default="ppl",
        help="sensitivity metric for mix precision search",
    )
    parser.add_argument(
        "--mix-output-reduction",
        type=str,
        choices=["last_token", "mean_tokens"],
        default="last_token",
        help="only used for non-ppl search metrics",
    )
    parser.add_argument("--max-search-batches", type=int, default=8, help="limit search calibration batches")
    parser.add_argument("--search-device", type=str, default="cuda:0", help="device used for mix precision search")
    parser.add_argument("--quant-device", type=str, default="cuda:0", help="device used for quantization")
    parser.add_argument("--infer-device", type=str, default="cuda:0", help="device used for perplexity evaluation")
    parser.add_argument("--backend", type=str, default="torch", choices=[b.value for b in BACKEND])
    parser.add_argument("--sym", action=argparse.BooleanOptionalAction, default=True, help="use symmetric quantization")
    parser.add_argument("--desc-act", action=argparse.BooleanOptionalAction, default=False, help="activation ordering flag")
    parser.add_argument("--offload-to-disk", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--offload-root", type=str, default=None, help="optional offload root")
    parser.add_argument("--autoround-iters", type=int, default=200, help="AutoRound optimization steps")
    parser.add_argument("--autoround-version", type=str, choices=["v1", "v2"], default="v1")
    parser.add_argument(
        "--pipelines",
        nargs="+",
        choices=["gptq", "autoround"],
        default=["gptq", "autoround"],
        help="which quantization pipelines to run",
    )
    parser.add_argument(
        "--dynamic-json",
        type=str,
        default=None,
        help="reuse an existing mix-precision dynamic json and skip the search step",
    )
    parser.add_argument(
        "--autoround-hessian-mse",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="enable Hessian-MSE initialization for AutoRound",
    )
    parser.add_argument("--ppl-n-ctx", type=int, default=512, help="perplexity context size")
    parser.add_argument("--ppl-n-batch", type=int, default=512, help="perplexity batch size")
    parser.add_argument("--trust-remote-code", action="store_true", default=False)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def resolve_device(device: str) -> str:
    if device.startswith("cuda") and not torch.cuda.is_available():
        log(f"CUDA is unavailable, falling back from {device} to cpu")
        return "cpu"
    return device


def save_json(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def clear_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def build_tokenizer(args) -> AutoTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        use_fast=True,
        trust_remote_code=args.trust_remote_code,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    return tokenizer


def load_calibration_texts(args) -> List[str]:
    dataset = load_dataset(args.dataset_path, args.dataset_name, split=args.calibration_split)
    texts = []
    target_count = None if args.calibration_samples == 0 else args.calibration_samples
    for row in dataset:
        text = row[args.text_column].strip()
        if len(text) < args.calibration_min_chars:
            continue
        texts.append(text)
        if target_count is not None and len(texts) >= target_count:
            break

    if target_count is not None and len(texts) < target_count:
        raise RuntimeError(
            f"Only found {len(texts)} calibration rows with >= {args.calibration_min_chars} chars, "
            f"expected {target_count}."
        )

    return texts


def average_ppl(model, tokenizer, args) -> float:
    ppl = Perplexity(
        model=model,
        tokenizer=tokenizer,
        dataset_path=args.dataset_path,
        dataset_name=args.dataset_name,
        split=args.ppl_split,
        text_column=args.text_column,
    )
    values = ppl.calculate(n_ctx=args.ppl_n_ctx, n_batch=args.ppl_n_batch)
    return float(sum(values) / len(values))


def build_mix_precision_config(args, output_path: Path) -> Dict:
    config = {
        "enabled": True,
        "weight_bits": args.mix_weight_bits,
        "policy": args.mix_policy,
        "topk": args.mix_topk,
        "threshold": args.mix_threshold,
        "metric": args.mix_metric,
        "store_baseline_on_cpu": True,
        "max_calibration_batches": args.max_search_batches,
        "output_path": str(output_path),
    }
    if args.mix_metric not in {"ppl", "perplexity"}:
        config["output_reduction"] = args.mix_output_reduction
    return config


def bit_distribution(dynamic: Dict[str, Dict[str, int]]) -> Dict[str, int]:
    distribution: Dict[str, int] = {}
    for config in dynamic.values():
        bit = str(config.get("bits"))
        distribution[bit] = distribution.get(bit, 0) + 1
    return distribution


def run_mix_precision_search(args, calibration_texts: List[str], output_root: Path) -> Dict[str, Dict[str, int]]:
    search_device = resolve_device(args.search_device)
    search_output_path = output_root / "mix_precision_search.json"
    dynamic_output_path = output_root / "mix_precision_dynamic.json"

    quant_config = QuantizeConfig(
        bits=args.bits,
        group_size=args.group_size,
        sym=args.sym,
        desc_act=args.desc_act,
        mix_precision_search=True,
        device=search_device,
        offload_to_disk=False,
    )
    quant_config.extension_set("mix_precision", build_mix_precision_config(args, search_output_path))

    log(f"[mix-search] loading native model from {args.model}")
    model = GPTQModel.load(
        args.model,
        quantize_config=quant_config,
        trust_remote_code=args.trust_remote_code,
    )

    log(f"[mix-search] moving model to {search_device}")
    model.model = model.model.to(search_device)

    calibration_batches = prepare_calibration_dataset(
        qmodel=model,
        calibration_dataset=calibration_texts,
        calibration_dataset_concat_size=args.calibration_concat_size,
        calibration_dataset_sort=args.calibration_sort,
        batch_size=args.calibration_batch_size,
        calibration_data_min_length=args.calibration_data_min_length,
    )

    log(f"[mix-search] running search on {len(calibration_batches)} prepared batches")
    allocation = run_mix_precision_pipeline(
        model=model.model,
        quantize_config=quant_config,
        calibration_data=calibration_batches,
        device=search_device,
        mix_precision_config_template=model.mix_precision_config_template(),
    )

    dynamic = copy.deepcopy(quant_config.dynamic or {})
    if not dynamic:
        raise RuntimeError("Mix precision search finished without producing any dynamic overrides.")

    payload = json.loads(search_output_path.read_text(encoding="utf-8"))
    save_json(dynamic_output_path, dynamic)

    search_summary = {
        "search_device": search_device,
        "search_output": str(search_output_path),
        "dynamic_output": str(dynamic_output_path),
        "metric": args.mix_metric,
        "allocation_count": len(allocation or {}),
        "dynamic_count": len(dynamic),
        "bit_distribution": bit_distribution(dynamic),
        "top_sensitivities": sorted(
            payload.get("sensitivities", {}).items(),
            key=lambda item: item[1],
            reverse=True,
        )[:10],
    }

    log(f"[mix-search] dynamic overrides: {search_summary['dynamic_count']}")
    log(f"[mix-search] bit distribution: {search_summary['bit_distribution']}")

    del model
    clear_memory()

    return {
        "dynamic": dynamic,
        "summary": search_summary,
    }


def build_gptq_config(args, dynamic: Dict[str, Dict[str, int]], offload_path: Path | None) -> QuantizeConfig:
    return QuantizeConfig(
        bits=args.bits,
        group_size=args.group_size,
        sym=args.sym,
        desc_act=args.desc_act,
        dynamic=copy.deepcopy(dynamic),
        device=resolve_device(args.quant_device),
        offload_to_disk=args.offload_to_disk,
        offload_to_disk_path=str(offload_path) if offload_path is not None else None,
    )


def build_autoround_config(args, dynamic: Dict[str, Dict[str, int]], offload_path: Path | None) -> QuantizeConfig:
    return QuantizeConfig(
        auto_round_enabled=True,
        auto_round_version=AutoRoundVersion(args.autoround_version),
        bits=args.bits,
        group_size=args.group_size,
        sym=args.sym,
        desc_act=args.desc_act,
        dynamic=copy.deepcopy(dynamic),
        format=FORMAT.GPTQ,
        device=resolve_device(args.quant_device),
        iters=args.autoround_iters,
        hessian_mse=args.autoround_hessian_mse,
        offload_to_disk=args.offload_to_disk,
        offload_to_disk_path=str(offload_path) if offload_path is not None else None,
    )


def quantize_and_eval(
    args,
    label: str,
    quantize_config: QuantizeConfig,
    calibration_texts: List[str],
    tokenizer,
    output_dir: Path,
) -> Dict:
    log(f"[{label}] loading native model")
    model = GPTQModel.load(
        args.model,
        quantize_config=quantize_config,
        trust_remote_code=args.trust_remote_code,
    )

    quant_start = time.time()
    log(f"[{label}] quantizing with {len(calibration_texts)} raw calibration rows")
    model.quantize(
        calibration_texts,
        calibration_concat_size=args.calibration_concat_size,
        calibration_sort=args.calibration_sort,
        batch_size=args.calibration_batch_size,
        calibration_data_min_length=args.calibration_data_min_length,
    )
    quant_seconds = time.time() - quant_start

    output_dir.mkdir(parents=True, exist_ok=True)
    log(f"[{label}] saving quantized model to {output_dir}")
    model.save(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    del model
    clear_memory()

    infer_device = resolve_device(args.infer_device)
    log(f"[{label}] reloading quantized model on {infer_device} for perplexity")
    reloaded = GPTQModel.load(
        str(output_dir),
        backend=BACKEND(args.backend),
        device=infer_device,
        trust_remote_code=args.trust_remote_code,
    )

    ppl_start = time.time()
    ppl_value = average_ppl(reloaded, tokenizer, args)
    ppl_seconds = time.time() - ppl_start

    del reloaded
    clear_memory()

    result = {
        "output_dir": str(output_dir),
        "quant_method": quantize_config.quant_method.value,
        "quant_seconds": quant_seconds,
        "ppl": ppl_value,
        "ppl_seconds": ppl_seconds,
        "bit_distribution": bit_distribution(quantize_config.dynamic or {}),
    }
    log(f"[{label}] ppl={ppl_value:.6f}, quant_seconds={quant_seconds:.2f}, ppl_seconds={ppl_seconds:.2f}")
    return result


def main():
    args = parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    summary_path = output_root / "run_summary.json"

    log(f"output root: {output_root}")
    tokenizer = build_tokenizer(args)
    calibration_texts = load_calibration_texts(args)
    log(f"loaded {len(calibration_texts)} calibration rows from {args.dataset_path}:{args.calibration_split}")

    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    else:
        summary = {}

    summary.update(
        {
            "model": args.model,
            "output_root": str(output_root),
            "dataset": {
                "path": args.dataset_path,
                "name": args.dataset_name,
                "calibration_split": args.calibration_split,
                "ppl_split": args.ppl_split,
                "text_column": args.text_column,
            },
            "calibration": {
                "samples": args.calibration_samples,
                "min_chars": args.calibration_min_chars,
                "concat_size": args.calibration_concat_size,
                "batch_size": args.calibration_batch_size,
                "sort": args.calibration_sort,
                "data_min_length": args.calibration_data_min_length,
            },
        }
    )
    summary.setdefault("mix_precision", None)
    summary.setdefault("gptq", None)
    summary.setdefault("autoround", None)
    save_json(summary_path, summary)

    selected_pipelines = list(dict.fromkeys(args.pipelines))

    if args.dynamic_json is not None:
        dynamic_path = Path(args.dynamic_json).resolve()
        dynamic = json.loads(dynamic_path.read_text(encoding="utf-8"))
        existing_reused = []
        if isinstance(summary.get("mix_precision"), dict):
            existing_reused = list(summary["mix_precision"].get("reused_for", []))
        reused_for = list(dict.fromkeys(existing_reused + selected_pipelines))
        if summary["mix_precision"] is None:
            summary["mix_precision"] = {
                "dynamic_output": str(dynamic_path),
                "reused_for": reused_for,
            }
        else:
            summary["mix_precision"]["dynamic_output"] = str(dynamic_path)
            summary["mix_precision"]["reused_for"] = reused_for
        save_json(summary_path, summary)
    else:
        mix_precision_result = run_mix_precision_search(args, calibration_texts, output_root)
        dynamic = mix_precision_result["dynamic"]
        summary["mix_precision"] = mix_precision_result["summary"]
        summary["mix_precision"]["reused_for"] = selected_pipelines
        save_json(summary_path, summary)

    offload_root = Path(args.offload_root).resolve() if args.offload_root is not None else output_root / "offload"

    if "gptq" in selected_pipelines:
        gptq_config = build_gptq_config(
            args,
            dynamic=dynamic,
            offload_path=(offload_root / "gptq") if args.offload_to_disk else None,
        )
        summary["gptq"] = quantize_and_eval(
            args,
            label="gptq",
            quantize_config=gptq_config,
            calibration_texts=calibration_texts,
            tokenizer=tokenizer,
            output_dir=output_root / "gptq_model",
        )
        save_json(summary_path, summary)

    if "autoround" in selected_pipelines:
        autoround_config = build_autoround_config(
            args,
            dynamic=dynamic,
            offload_path=(offload_root / "autoround") if args.offload_to_disk else None,
        )
        summary["autoround"] = quantize_and_eval(
            args,
            label="autoround",
            quantize_config=autoround_config,
            calibration_texts=calibration_texts,
            tokenizer=tokenizer,
            output_dir=output_root / "autoround_model",
        )
        save_json(summary_path, summary)

    log("all pipelines completed")
    print(json.dumps(summary, indent=2, ensure_ascii=True), flush=True)


if __name__ == "__main__":
    main()
