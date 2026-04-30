# SPDX-License-Identifier: Apache-2.0

import argparse
import logging
import os

from transformers import AutoTokenizer

from gptqmodel import GPTQModel, QuantizeConfig

from qwen35_common import (
    ensure_processor_saved,
    ensure_visual_tensors_present,
    is_moe_model,
    resolve_rotation_vision,
    rotate_mtp_after_save,
    set_seed,
    validate_vision_demo_pair,
)


DEFAULT_PROMPT = "请简要描述这张图片中的主要内容。"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Save a rotated Qwen3.5 VL fp checkpoint with LLM, vision, and MTP using one shared Q"
    )
    parser.add_argument("--model", type=str, default="/data01/datasets/Qwen3.5-9B")
    parser.add_argument("--out", type=str, default="/data02/datasets/Qwen3.5-9B-rotated-fp")
    parser.add_argument(
        "--rotation",
        "--llm-rotation",
        dest="rotation",
        type=str,
        default="hadamard",
        choices=["hadamard", "block_hadamard", "random"],
    )
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--device-map", type=str, default=None, choices=["auto", "balanced", "balanced_low_0", "sequential"])
    parser.add_argument("--offload-to-disk", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--offload-path", type=str, default="./gptqmodel_offload_qwen35vl_rotate")
    parser.add_argument("--max-shard-size", type=str, default="5GB")
    parser.add_argument("--trust-remote-code", action="store_true", default=False)
    parser.add_argument("--validate", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--validate-device", type=str, default=None)
    parser.add_argument("--validate-dtype", type=str, default="bfloat16", choices=["auto", "float16", "bfloat16", "float32", "float64"])
    parser.add_argument("--validate-prompt", type=str, default=DEFAULT_PROMPT)
    parser.add_argument("--validate-image-size", type=int, default=224)
    parser.add_argument("--validate-generation-tokens", type=int, default=8)
    parser.add_argument("--atol", type=float, default=5e-2)
    parser.add_argument("--rtol", type=float, default=5e-2)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    rotation_vision = resolve_rotation_vision(args.model, args.rotation, args.trust_remote_code)
    is_moe = is_moe_model(args.model, args.trust_remote_code)
    print(f"[rotation] llm={args.rotation} vision={rotation_vision} mtp=same-as-llm is_moe={is_moe}")

    quantize_config = QuantizeConfig(
        bits=4,
        group_size=64,
        rotation=args.rotation,
        rotation_vision=rotation_vision,
        device=None if args.device_map else args.device,
        offload_to_disk=args.offload_to_disk,
        offload_to_disk_path=args.offload_path if args.offload_to_disk else None,
    )

    print("[1/4] Loading fp model ...")
    load_kwargs = dict(trust_remote_code=args.trust_remote_code)
    if args.device_map:
        load_kwargs["device_map"] = args.device_map
    model = GPTQModel.load(args.model, quantize_config, **load_kwargs)
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True, trust_remote_code=args.trust_remote_code)

    print("[2/4] Applying rotation ...")
    model.apply_rotation()

    print(f"[3/4] Saving rotated fp model to: {args.out}")
    os.makedirs(args.out, exist_ok=True)
    model.model.save_pretrained(args.out, safe_serialization=True, max_shard_size=args.max_shard_size)
    tokenizer.save_pretrained(args.out)
    ensure_processor_saved(args.model, args.out, args.trust_remote_code)
    rotate_mtp_after_save(args.model, args.out, model, is_moe=is_moe, rotation=args.rotation)
    ensure_visual_tensors_present(args.model, args.out)

    if args.validate:
        print("[4/4] Validating vision demo equivalence ...")
        validate_vision_demo_pair(
            args.model,
            args.out,
            device=args.validate_device or args.device,
            dtype=args.validate_dtype,
            prompt=args.validate_prompt,
            image_size=args.validate_image_size,
            seed=args.seed,
            trust_remote_code=args.trust_remote_code,
            atol=args.atol,
            rtol=args.rtol,
            generation_tokens=args.validate_generation_tokens,
        )
    else:
        print("[4/4] Validation skipped.")


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    main()
