#!/usr/bin/env python3
"""Quantize Ling-3-Flash to W8/W4 G64 symmetric GPTQModel weights."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict

from gptqmodel.recipes.ling3_flash import quantize_ling3_flash


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Ling-3-Flash HF checkpoint")
    parser.add_argument("--out", required=True, help="GPTQModel HF output directory")
    parser.add_argument("--bits", type=int, default=8)
    parser.add_argument("--expert-bits", type=int, default=4)
    parser.add_argument("--group-size", type=int, default=64)
    parser.add_argument("--sym", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--nsamples", type=int, default=128)
    parser.add_argument("--seqlen", type=int, default=2048)
    parser.add_argument("--calibration-jsonl")
    parser.add_argument("--calibration-dataset", default="NeelNanda/pile-10k")
    parser.add_argument("--calibration-text-key", default="text")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--trust-remote-code",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--offload-to-disk",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Opt-in only: the validated Ling recipe keeps disk offload "
            "disabled because the related Qwen3.5 hybrid/MoE path has "
            "produced invalid quantized checkpoints; revalidate the result "
            "when enabling it"
        ),
    )
    parser.add_argument("--offload-path")
    parser.add_argument(
        "--wait-for-submodule-finalizers",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--hessian-mse",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--moe-routing-batch-size", type=int)
    parser.add_argument(
        "--auto-forward-data-parallel",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--max-quant-layers", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = quantize_ling3_flash(
        model_dir=args.model,
        output_dir=args.out,
        bits=args.bits,
        expert_bits=args.expert_bits,
        group_size=args.group_size,
        sym=args.sym,
        batch_size=args.batch_size,
        nsamples=args.nsamples,
        seqlen=args.seqlen,
        calibration_jsonl=args.calibration_jsonl,
        calibration_dataset=args.calibration_dataset,
        calibration_text_key=args.calibration_text_key,
        device=args.device,
        trust_remote_code=args.trust_remote_code,
        offload_to_disk=args.offload_to_disk,
        offload_path=args.offload_path,
        wait_for_submodule_finalizers=args.wait_for_submodule_finalizers,
        hessian_mse=args.hessian_mse,
        moe_routing_batch_size=args.moe_routing_batch_size,
        auto_forward_data_parallel=args.auto_forward_data_parallel,
        max_quant_layers=args.max_quant_layers,
        seed=args.seed,
        dry_run=args.dry_run,
    )
    print(json.dumps(asdict(result), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
