#!/usr/bin/env python3
"""Quantize V4 Flash to symmetric W8/W4-G64 with GPTQ or AutoRound."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict

from gptqmodel.recipes.deepseek_v4_flash import (
    prepare_deepseek_v4_flash_bf16,
    quantize_deepseek_v4_flash,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default="/data01/datasets/DeepSeek-V4-Flash-0731",
        help="Native DeepSeek-V4 Flash checkpoint",
    )
    parser.add_argument("--out", required=True, help="GPTQModel output directory")
    parser.add_argument(
        "--prepared-model",
        required=True,
        help="BF16 staging checkpoint; reused after a successful preparation",
    )
    parser.add_argument(
        "--num-layers",
        type=int,
        default=None,
        help="Physically prepare only the first N decoder layers (for example 6)",
    )
    parser.add_argument(
        "--method",
        default="gptq",
        choices=["gptq", "autoround", "auto_round"],
    )
    parser.add_argument("--bits", type=int, default=8, choices=[8])
    parser.add_argument("--expert-bits", type=int, default=4, choices=[4])
    parser.add_argument("--group-size", type=int, default=64, choices=[64])
    parser.add_argument(
        "--sym",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Deployment profile requires symmetric quantization",
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--nsamples", type=int, default=128)
    parser.add_argument("--seqlen", type=int, default=2048)
    parser.add_argument("--calibration-jsonl", default=None)
    parser.add_argument("--calibration-text-key", default="text")
    parser.add_argument("--calibration-dataset", default="NeelNanda/pile-10k")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--preparation-device", default="cuda:0")
    parser.add_argument("--offload-path", default=None)
    parser.add_argument(
        "--moe-routing-batch-size",
        type=int,
        default=128,
        help=(
            "Number of expert projection modules calibrated at once; 128 "
            "keeps one V4 layer comfortably within a single 80-GiB GPU"
        ),
    )
    parser.add_argument("--max-quant-layers", type=int, default=None)
    parser.add_argument(
        "--auto-round-version",
        default="v2",
        choices=["v1", "v2"],
    )
    parser.add_argument("--auto-round-iters", type=int, default=200)
    parser.add_argument("--auto-round-lr", type=float, default=None)
    parser.add_argument("--auto-round-minmax-lr", type=float, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--offload-to-disk",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Opt-in only; disabled by default",
    )
    parser.add_argument(
        "--auto-forward-data-parallel",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Opt in to cloning the current layer on every visible GPU for calibration "
            "forwards; disabled because layerwise GPTQ/AutoRound fits one GPU"
        ),
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Only produce the BF16 staging checkpoint",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.prepare_only:
        if args.dry_run:
            payload = {
                "source_model_dir": args.model,
                "output_dir": args.prepared_model,
                "num_layers": args.num_layers,
                "target_dtype": "bfloat16",
                "dry_run": True,
            }
        else:
            result = prepare_deepseek_v4_flash_bf16(
                model_dir=args.model,
                output_dir=args.prepared_model,
                num_layers=args.num_layers,
                device=args.preparation_device,
            )
            payload = asdict(result)
    else:
        result = quantize_deepseek_v4_flash(
            model_dir=args.model,
            prepared_model_dir=args.prepared_model,
            output_dir=args.out,
            num_layers=args.num_layers,
            method=args.method,
            bits=args.bits,
            expert_bits=args.expert_bits,
            group_size=args.group_size,
            sym=args.sym,
            batch_size=args.batch_size,
            nsamples=args.nsamples,
            seqlen=args.seqlen,
            calibration_jsonl=args.calibration_jsonl,
            calibration_text_key=args.calibration_text_key,
            calibration_dataset=args.calibration_dataset,
            device=args.device,
            preparation_device=args.preparation_device,
            offload_to_disk=args.offload_to_disk,
            offload_path=args.offload_path,
            moe_routing_batch_size=args.moe_routing_batch_size,
            auto_forward_data_parallel=args.auto_forward_data_parallel,
            max_quant_layers=args.max_quant_layers,
            auto_round_version=args.auto_round_version,
            auto_round_iters=args.auto_round_iters,
            auto_round_lr=args.auto_round_lr,
            auto_round_minmax_lr=args.auto_round_minmax_lr,
            seed=args.seed,
            dry_run=args.dry_run,
        )
        payload = asdict(result)
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
