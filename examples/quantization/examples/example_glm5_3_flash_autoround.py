"""GLM-5.3 AutoRound: native decoder blocks, W4 G64 routed experts / W8 others."""

import argparse
from dataclasses import asdict
import json

from gptqmodel.recipes.glm5_3_flash_autoround import quantize_glm5_3_flash_autoround


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--prepared-model-dir")
    parser.add_argument("--calibration-jsonl", required=True)
    parser.add_argument("--device-map", default="0")
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--nsamples", type=int, default=128)
    parser.add_argument("--seqlen", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulate-steps", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true")
    result = quantize_glm5_3_flash_autoround(**vars(parser.parse_args()))
    print(json.dumps(asdict(result), ensure_ascii=False, indent=2))
