# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

import argparse
import gc
import json
import os
import tempfile
import time
from pathlib import Path
from typing import Dict, Optional

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from gptqmodel import BACKEND, GPTQModel
from gptqmodel.quantization import FORMAT
from gptqmodel.quantization.config import AutoRoundQuantizeConfig, AutoRoundVersion
from gptqmodel.utils.perplexity import Perplexity
from gptqmodel.utils.torch import torch_empty_cache


os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True,max_split_size_mb:256,garbage_collection_threshold:0.7")


def log(message: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate Auto-Round perplexity with optional rotation for one or more AutoRound versions."
    )
    parser.add_argument("--model", type=str, default="/data02/datasets/Qwen3-4B", help="native model path")
    parser.add_argument("--output", type=str, default="./autoround_perplexity_results.json", help="JSON output path")
    parser.add_argument(
        "--artifact-root",
        type=str,
        default=None,
        help="optional directory used as the parent for temporary quantized checkpoints",
    )
    parser.add_argument("--dataset-path", type=str, default="wikitext", help="dataset path for perplexity and calibration")
    parser.add_argument("--dataset-name", type=str, default="wikitext-2-raw-v1", help="dataset name")
    parser.add_argument("--ppl-split", type=str, default="test", help="dataset split used for perplexity")
    parser.add_argument("--calibration-split", type=str, default="train", help="dataset split used for calibration")
    parser.add_argument("--text-column", type=str, default="text", help="dataset text column")
    parser.add_argument("--n-ctx", type=int, default=512, help="perplexity context size")
    parser.add_argument("--n-batch", type=int, default=512, help="perplexity batch size")
    parser.add_argument("--calibration-samples", type=int, default=64, help="number of long calibration rows to keep")
    parser.add_argument("--calibration-min-chars", type=int, default=1024, help="minimum character length per raw row")
    parser.add_argument("--calibration-concat-size", type=int, default=512, help="target concatenated calibration token length")
    parser.add_argument("--quant-batch-size", type=int, default=8, help="quantization batch size")
    parser.add_argument("--bits", type=int, default=4, choices=[2, 3, 4, 8], help="weight bits")
    parser.add_argument("--group-size", type=int, default=128, help="group size")
    parser.add_argument("--iters", type=int, default=32, help="Auto-Round optimization steps")
    parser.add_argument("--device", type=str, default="cuda:0", help="quantization device")
    parser.add_argument("--backend", choices=["auto", "torch", "triton", "cuda", "ipex", "marlin", "bitblas"], default="torch")
    parser.add_argument("--sym", action=argparse.BooleanOptionalAction, default=True, help="use symmetric quantization")
    parser.add_argument("--desc-act", action=argparse.BooleanOptionalAction, default=False, help="activation ordering flag")
    parser.add_argument(
        "--hessian-mse",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="enable Hessian-MSE scale initialization",
    )
    parser.add_argument("--offload-to-disk", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--offload-path", type=str, default=None, help="explicit offload directory")
    parser.add_argument(
        "--versions",
        nargs="+",
        default=["v1", "v2"],
        choices=["v1", "v2"],
        help="AutoRound versions to evaluate",
    )
    parser.add_argument(
        "--rotations",
        nargs="+",
        default=["none"],
        choices=["none", "hadamard", "block_hadamard", "random"],
        help="rotation modes to evaluate; 'none' disables rotation",
    )
    parser.add_argument("--use-fast-tokenizer", action="store_true", default=False)
    parser.add_argument("--trust-remote-code", action="store_true", default=False)
    return parser.parse_args()


def rotation_to_value(rotation_label: str) -> Optional[str]:
    return None if rotation_label == "none" else rotation_label


def maybe_write_results(output_path: Optional[Path], payload: Dict) -> None:
    if output_path is None:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n")


def clear_memory():
    gc.collect()
    torch_empty_cache()


def average_ppl(model, tokenizer, args) -> float:
    ppl = Perplexity(
        model=model,
        tokenizer=tokenizer,
        dataset_path=args.dataset_path,
        dataset_name=args.dataset_name,
        split=args.ppl_split,
        text_column=args.text_column,
    )
    all_ppl = ppl.calculate(n_ctx=args.n_ctx, n_batch=args.n_batch)
    return float(sum(all_ppl) / len(all_ppl))


def load_calibration_dataset(args):
    dataset = load_dataset(args.dataset_path, args.dataset_name, split=args.calibration_split)
    selected_indices = []
    for idx, row in enumerate(dataset):
        if len(row[args.text_column]) >= args.calibration_min_chars:
            selected_indices.append(idx)
            if len(selected_indices) >= args.calibration_samples:
                break
    if len(selected_indices) < args.calibration_samples:
        raise RuntimeError(
            f"Expected at least {args.calibration_samples} calibration rows with >= {args.calibration_min_chars} chars, got {len(selected_indices)}"
        )
    return dataset.select(selected_indices)


def load_tokenizer(args):
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        use_fast=args.use_fast_tokenizer,
        trust_remote_code=args.trust_remote_code,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    return tokenizer


def build_quantize_config(args, version: AutoRoundVersion, rotation_label: str):
    rotation = rotation_to_value(rotation_label)
    offload_path = args.offload_path
    if args.offload_to_disk and offload_path is not None:
        offload_path = str(Path(offload_path) / f"{version.value}-{rotation_label}")

    return AutoRoundQuantizeConfig(
        bits=args.bits,
        group_size=args.group_size,
        sym=args.sym,
        format=FORMAT.GPTQ,
        desc_act=args.desc_act,
        iters=args.iters,
        hessian_mse=args.hessian_mse,
        auto_round_version=version,
        rotation=rotation,
        device=args.device,
        offload_to_disk=args.offload_to_disk,
        offload_to_disk_path=offload_path,
    )


def evaluate_combo(args, tokenizer, calibration_dataset, version: AutoRoundVersion, rotation_label: str):
    quantize_config = build_quantize_config(args, version, rotation_label)
    combo_label = f"{version.value}:{rotation_label}"
    log(f"Loading model for Auto-Round {combo_label}")
    model = GPTQModel.load(args.model, quantize_config=quantize_config, trust_remote_code=args.trust_remote_code)

    quant_start = time.time()
    model.quantize(
        calibration_dataset,
        calibration_concat_size=args.calibration_concat_size,
        batch_size=args.quant_batch_size,
    )
    quant_seconds = time.time() - quant_start
    log(f"Quantization complete for {combo_label} in {quant_seconds:.2f}s")

    artifact_root = Path(args.artifact_root) if args.artifact_root is not None else None
    if artifact_root is not None:
        artifact_root.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix=f"autoround-{version.value}-{rotation_label}-", dir=artifact_root) as tmp_dir:
        log(f"Saving quantized model for {combo_label} to {tmp_dir}")
        model.save(tmp_dir)
        tokenizer.save_pretrained(tmp_dir)

        del model
        clear_memory()

        log(f"Reloading quantized model for {combo_label}")
        reloaded = GPTQModel.load(
            tmp_dir,
            backend=BACKEND(args.backend.lower()),
            device_map="auto",
            trust_remote_code=args.trust_remote_code,
        )

        ppl_start = time.time()
        ppl_value = average_ppl(reloaded, tokenizer, args)
        ppl_seconds = time.time() - ppl_start
        log(f"Perplexity complete for {combo_label}: {ppl_value:.6f} in {ppl_seconds:.2f}s")

    del reloaded
    clear_memory()

    return {
        "auto_round_version": version.value,
        "rotation": rotation_label,
        "iters": args.iters,
        "bits": args.bits,
        "group_size": args.group_size,
        "sym": args.sym,
        "desc_act": args.desc_act,
        "hessian_mse": args.hessian_mse,
        "calibration_samples": args.calibration_samples,
        "calibration_min_chars": args.calibration_min_chars,
        "calibration_concat_size": args.calibration_concat_size,
        "quant_batch_size": args.quant_batch_size,
        "quant_seconds": quant_seconds,
        "ppl": ppl_value,
        "ppl_seconds": ppl_seconds,
    }


def main():
    args = parse_args()
    output_path = Path(args.output) if args.output else None

    log(f"Writing results to {output_path}" if output_path is not None else "Printing results to stdout only")
    tokenizer = load_tokenizer(args)
    calibration_dataset = load_calibration_dataset(args)

    results = {
        "model_path": args.model,
        "dataset": {
            "path": args.dataset_path,
            "name": args.dataset_name,
            "ppl_split": args.ppl_split,
            "calibration_split": args.calibration_split,
            "text_column": args.text_column,
            "n_ctx": args.n_ctx,
            "n_batch": args.n_batch,
        },
        "baseline": None,
        "quantized": {version: {} for version in args.versions},
    }
    maybe_write_results(output_path, results)

    log("Loading native baseline model")
    baseline_model = AutoModelForCausalLM.from_pretrained(
        args.model,
        device_map="auto",
        torch_dtype="auto",
        trust_remote_code=args.trust_remote_code,
    )
    baseline_model.eval()

    baseline_start = time.time()
    baseline_ppl = average_ppl(baseline_model, tokenizer, args)
    baseline_seconds = time.time() - baseline_start
    log(f"Baseline perplexity: {baseline_ppl:.6f} in {baseline_seconds:.2f}s")

    results["baseline"] = {
        "ppl": baseline_ppl,
        "seconds": baseline_seconds,
    }
    maybe_write_results(output_path, results)

    del baseline_model
    clear_memory()

    for version_name in args.versions:
        version = AutoRoundVersion(version_name)
        for rotation_label in args.rotations:
            combo_result = evaluate_combo(args, tokenizer, calibration_dataset, version, rotation_label)
            combo_result["delta_ppl"] = combo_result["ppl"] - baseline_ppl
            combo_result["ratio_to_baseline"] = combo_result["ppl"] / baseline_ppl
            results["quantized"][version_name][rotation_label] = combo_result
            maybe_write_results(output_path, results)

    log("Evaluation complete")
    print(json.dumps(results, indent=2, ensure_ascii=True), flush=True)


if __name__ == "__main__":
    main()
