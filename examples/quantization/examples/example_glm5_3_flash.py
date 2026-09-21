#!/usr/bin/env python3
"""GLM-5.3-Flash: GPTQ W4 G64 routed experts, W8 other GPTQ projections."""

import argparse
import json
from dataclasses import asdict

from gptqmodel.recipes.glm5_3_flash import quantize_glm5_3_flash


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", required=True, help="original multimodal BF16/block-FP8 checkpoint"
    )
    parser.add_argument("--out", required=True, help="new GPTQ checkpoint directory")
    parser.add_argument(
        "--prepared-model-dir",
        help="reusable BF16 input, separate from source and output",
    )
    parser.add_argument("--preparation-devices", nargs="+", default=["cpu"])
    parser.add_argument("--calibration-jsonl", required=True)
    parser.add_argument("--calibration-text-key", default="text")
    parser.add_argument("--nsamples", type=int, default=128)
    parser.add_argument("--seqlen", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--moe-routing-batch-size", type=int, default=32)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    result = quantize_glm5_3_flash(
        model_dir=args.model,
        output_dir=args.out,
        prepared_model_dir=args.prepared_model_dir,
        preparation_devices=tuple(args.preparation_devices),
        calibration_jsonl=args.calibration_jsonl,
        calibration_dataset=None,
        calibration_text_key=args.calibration_text_key,
        nsamples=args.nsamples,
        seqlen=args.seqlen,
        batch_size=args.batch_size,
        moe_routing_batch_size=args.moe_routing_batch_size,
        device=args.device,
        seed=args.seed,
        dry_run=args.dry_run,
    )
    print(json.dumps(asdict(result), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
