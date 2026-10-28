# Copyright 2026 HOUMO AI
# SPDX-License-Identifier: Apache-2.0
"""Quantize a native Qwen3-VL feature prefix through the public recipe."""

import argparse
import json
import torch

from gptqmodel.recipes import quantize_qwen3_vl


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--calibration-file", required=True, help="torch-saved list of native processor tensor dicts")
    parser.add_argument("--method", choices=("gptq", "autoround"), default="autoround")
    parser.add_argument("--feature-layer", type=int, default=50)
    parser.add_argument("--bits", type=int, choices=(4, 8), default=4)
    parser.add_argument("--iters", type=int, default=200, help="AutoRound tuning iterations; unused by GPTQ")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--tokenizer-dir")
    parser.add_argument("--processor-dir")
    args = vars(parser.parse_args())
    args["calibration_data"] = torch.load(args.pop("calibration_file"), map_location="cpu", weights_only=True)
    result = quantize_qwen3_vl(**args)
    print(json.dumps(dict(output_dir=result.output_dir, **result.provenance), indent=2))


if __name__ == "__main__":
    main()
