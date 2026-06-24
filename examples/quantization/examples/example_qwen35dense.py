# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

import argparse
import gc
import json
import logging
import os

import torch
from transformers import AutoTokenizer

from gptqmodel import GPTQModel, QuantizeConfig

from qwen35_common import (
    build_calibration_dataset,
    calculate_avg_ppl,
    ensure_processor_saved,
    ensure_visual_tensors_present,
    is_gptq_quantized_dir,
    is_moe_model,
    layer_limit_dynamic_skips,
    resolve_rotation_vision,
    rotate_mtp_after_save,
    set_seed,
    temporary_ppl_device,
    validate_vision_demo_pair,
)


DEFAULT_PROMPT = "请介绍一下 GPTQ 量化的核心思想。"
DEFAULT_VISION_PROMPT = "请简要描述这张图片中的主要内容。"


def build_chat_inputs(tokenizer, prompt: str, device: str):
    messages = [{"role": "user", "content": prompt}]
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        text = prompt
    return tokenizer(text, return_tensors="pt").to(device)


def _existing_quant_offload_to_disk(model_dir: str):
    for name in ("quantize_config.json", "config.json"):
        path = os.path.join(model_dir, name)
        if not os.path.exists(path):
            continue
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        quant_cfg = payload.get("quantization_config", payload)
        meta = quant_cfg.get("meta", {})
        if "offload_to_disk" in meta:
            return bool(meta["offload_to_disk"])
    return None


def parse_args():
    parser = argparse.ArgumentParser(
        description="Qwen3.5 dense/VL GPTQ quantization with unified LLM+vision+MTP rotation"
    )
    parser.add_argument("--model", type=str, default="/data01/datasets/Qwen3.5-9B", help="fp model path")
    parser.add_argument("--out", type=str, default="/data02/datasets/Qwen3.5-9B-gptq-4bit")
    parser.add_argument("--bits", type=int, default=4, choices=[2, 3, 4, 5, 8])
    parser.add_argument("--group-size", type=int, default=64)
    parser.add_argument(
        "--rotation",
        type=str,
        default=None,
        choices=["hadamard", "block_hadamard", "random"],
        help="LLM rotation mode. When set, vision and MTP rotate with the same Q.",
    )
    parser.add_argument("--check-rotation-ppl", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--ppl-samples", type=int, default=8)
    parser.add_argument("--ppl-seqlen", type=int, default=512)
    parser.add_argument("--check-vision-demo", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--check-quant-vision-demo", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--rotated-out", type=str, default=None)
    parser.add_argument("--stop-after-rotate", action="store_true", default=False)
    parser.add_argument("--mse", type=float, default=2.4)
    parser.add_argument("--hessian-mse", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--wait-for-submodule-finalizers",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Wait for GPTQ packing/finalize tasks before saving. "
            "Qwen3.5 checkpoints can be corrupted with zero qweight buffers if saving races async finalizers."
        ),
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--nsamples", type=int, default=256)
    parser.add_argument("--seqlen", type=int, default=1024)
    parser.add_argument("--calibration-jsonl", type=str, default=None)
    parser.add_argument("--calibration-text-key", type=str, default="text")
    parser.add_argument("--max-quant-layers", type=int, default=None, help="Smoke-test only: quantize at most this many text layers.")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--infer-device", type=str, default=None)
    parser.add_argument("--device-map", type=str, default=None, choices=["auto", "balanced", "balanced_low_0", "sequential"])
    parser.add_argument("--offload-to-disk", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--offload-path", type=str, default="./gptqmodel_offload_qwen35dense")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--prompt", type=str, default=DEFAULT_PROMPT)
    parser.add_argument("--trust-remote-code", action="store_true", default=False)
    parser.add_argument("--skip-quantize", action="store_true", default=False)
    parser.add_argument("--skip-inference", action="store_true", default=False)
    parser.add_argument("--max-shard-size", type=str, default="5GB")
    parser.add_argument("--vision-demo-device", type=str, default="cuda:0")
    parser.add_argument("--vision-demo-dtype", type=str, default="bfloat16", choices=["auto", "float16", "bfloat16", "float32", "float64"])
    parser.add_argument("--vision-demo-prompt", type=str, default=DEFAULT_VISION_PROMPT)
    parser.add_argument("--vision-demo-image-size", type=int, default=224)
    parser.add_argument("--vision-demo-generation-tokens", type=int, default=512)
    parser.add_argument("--vision-demo-atol", type=float, default=5e-2)
    parser.add_argument("--vision-demo-rtol", type=float, default=5e-2)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()

    if is_moe_model(args.model, args.trust_remote_code):
        raise ValueError("example_qwen35dense.py is for Qwen3.5 dense models; use example_qwen35moe.py for MoE.")
    if (args.check_rotation_ppl or args.check_vision_demo or args.rotated_out or args.stop_after_rotate) and not args.rotation:
        raise ValueError("rotation validation/save options require --rotation.")
    if args.check_rotation_ppl and args.offload_to_disk:
        raise ValueError("--check-rotation-ppl requires --no-offload-to-disk; offloaded meta tensors cannot be evaluated.")
    if args.rotation and args.offload_to_disk:
        raise ValueError(
            "Qwen3.5 rotated quantization must run with --no-offload-to-disk; "
            "offload_to_disk is known to produce corrupted rotated checkpoints."
        )
    if not args.skip_quantize and is_gptq_quantized_dir(args.out):
        existing_offload = _existing_quant_offload_to_disk(args.out)
        if args.rotation and existing_offload:
            print(
                f"[0/5] Existing GPTQ output at {args.out} was created with offload_to_disk=true; "
                "re-quantizing because that path produces corrupted rotated dense checkpoints."
            )
        else:
            print(f"[0/5] Existing GPTQ output detected at {args.out}; skipping quantization and reusing it.")
            args.skip_quantize = True

    set_seed(args.seed)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    rotation_vision = resolve_rotation_vision(args.model, args.rotation, args.trust_remote_code)
    print(f"[rotation] llm={args.rotation} vision={rotation_vision} mtp={'same-as-llm' if args.rotation else None}")

    tokenizer_source = args.model
    tokenizer = None
    if not args.skip_quantize:
        tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True, trust_remote_code=args.trust_remote_code)
        calibration_dataset, calibration_source = build_calibration_dataset(args)
        print(f"[1/5] Calibration samples ({calibration_source}): {len(calibration_dataset)}")

        quantize_config = QuantizeConfig(
            bits=args.bits,
            dynamic=layer_limit_dynamic_skips(args.model, args.trust_remote_code, args.max_quant_layers),
            group_size=args.group_size,
            rotation=args.rotation,
            rotation_vision=rotation_vision,
            hessian_mse=args.hessian_mse,
            mse=args.mse,
            wait_for_submodule_finalizers=args.wait_for_submodule_finalizers,
            device=None if args.device_map else args.device,
            offload_to_disk=args.offload_to_disk,
            offload_to_disk_path=args.offload_path if args.offload_to_disk else None,
        )

        print("[2/5] Loading fp model ...")
        load_kwargs = dict(trust_remote_code=args.trust_remote_code)
        if args.device_map:
            load_kwargs["device_map"] = args.device_map
        model = GPTQModel.load(args.model, quantize_config, **load_kwargs)

        rotation_artifact_requested = any([args.check_rotation_ppl, args.check_vision_demo, args.rotated_out, args.stop_after_rotate])
        if args.rotation and rotation_artifact_requested:
            ppl_device = None
            if args.check_rotation_ppl:
                if args.device_map:
                    print(f"[ppl] Using existing device_map='{args.device_map}' for before/after rotation evaluation ...")
                else:
                    ppl_device = args.infer_device or args.device or "cuda:0"
                    print(f"[ppl] Moving fp model to {ppl_device} for before/after rotation evaluation ...")
                    model.model.to(ppl_device)
                print("[rotation] Evaluating fp model PPL before rotation ...")
                fp_ppl_before = calculate_avg_ppl(
                    model.model, tokenizer, device=ppl_device, nsamples=args.ppl_samples, seqlen=args.ppl_seqlen
                )
                print(f"[rotation] FP PPL before: {fp_ppl_before:.4f}")

            print(f"[rotation] Applying {args.rotation} rotation to fp model ...")
            model.apply_rotation()
            if ppl_device is not None:
                model.model.to(ppl_device)

            if args.check_rotation_ppl:
                print("[rotation] Evaluating fp model PPL after rotation ...")
                fp_ppl_after = calculate_avg_ppl(
                    model.model, tokenizer, device=ppl_device, nsamples=args.ppl_samples, seqlen=args.ppl_seqlen
                )
                print(f"[rotation] FP PPL after:  {fp_ppl_after:.4f} (delta {fp_ppl_after - fp_ppl_before:+.6f})")

            if args.rotated_out:
                print(f"[rotation] Saving rotated fp model to: {args.rotated_out}")
                os.makedirs(args.rotated_out, exist_ok=True)
                # Ensure no tied-weight metadata remains so both lm_head.weight and
                # embed_tokens.weight are saved as independent tensors.
                if hasattr(model.model, "_tied_weights_keys"):
                    model.model._tied_weights_keys = {}
                model.model.save_pretrained(args.rotated_out, safe_serialization=True, max_shard_size=args.max_shard_size)
                tokenizer.save_pretrained(args.rotated_out)
                ensure_processor_saved(args.model, args.rotated_out, args.trust_remote_code)
                rotate_mtp_after_save(args.model, args.rotated_out, model, is_moe=False, rotation=args.rotation)
                ensure_visual_tensors_present(args.model, args.rotated_out)

            if args.check_vision_demo:
                if not args.rotated_out:
                    raise ValueError("--check-vision-demo requires --rotated-out so the saved model path can be compared.")
                validate_vision_demo_pair(
                    args.model,
                    args.rotated_out,
                    device=args.vision_demo_device,
                    dtype=args.vision_demo_dtype,
                    prompt=args.vision_demo_prompt,
                    image_size=args.vision_demo_image_size,
                    seed=args.seed,
                    trust_remote_code=args.trust_remote_code,
                    atol=args.vision_demo_atol,
                    rtol=args.vision_demo_rtol,
                    generation_tokens=args.vision_demo_generation_tokens,
                )

            if args.stop_after_rotate:
                print("[rotation] Stop requested after rotation; skipping quantization.")
                return

        print("[3/5] Running GPTQ quantization ...")
        model.quantize(calibration_dataset, batch_size=args.batch_size)
        model.save(args.out)
        print(f"Quantized model saved to: {args.out}")
        rotate_mtp_after_save(args.model, args.out, model, is_moe=False, rotation=args.rotation)
        ensure_visual_tensors_present(args.model, args.out)
    else:
        print(f"[1/5] Skip quantization, using existing quantized model: {args.out}")
        tokenizer = AutoTokenizer.from_pretrained(args.out, use_fast=True, trust_remote_code=args.trust_remote_code)
        tokenizer_source = args.out

    if args.skip_inference:
        print("[4/5] Inference/PPL skipped.")
        return

    infer_device = args.infer_device or args.device
    infer_kwargs = dict(trust_remote_code=args.trust_remote_code)
    if args.device_map:
        infer_kwargs["device_map"] = args.device_map
        print(f"[4/5] Loading quantized model with device_map='{args.device_map}' and generating ...")
    else:
        infer_kwargs["device"] = infer_device
        print(f"[4/5] Loading quantized model on {infer_device} and generating ...")
    model = GPTQModel.load(args.out, **infer_kwargs)
    runtime_device = str(model.device)
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, use_fast=True, trust_remote_code=args.trust_remote_code)
    inputs = build_chat_inputs(tokenizer, args.prompt, runtime_device)

    with torch.inference_mode():
        output_ids = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)

    generated_ids = output_ids[:, inputs["input_ids"].shape[1]:]
    answer = tokenizer.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    print("=== Prompt ===")
    print(args.prompt)
    print("=== Answer ===")
    print(answer)

    print("[5/5] Evaluating perplexity on wikitext2 ...")
    avg_ppl = calculate_avg_ppl(model, tokenizer, nsamples=args.ppl_samples, seqlen=args.ppl_seqlen)
    print(f"Quantized model avg PPL (wikitext2): {avg_ppl:.4f}")

    if args.check_quant_vision_demo:
        del inputs, output_ids, generated_ids
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        ref_model_dir = args.rotated_out or args.model
        validate_vision_demo_pair(
            ref_model_dir,
            args.out,
            device=args.vision_demo_device,
            dtype=args.vision_demo_dtype,
            prompt=args.vision_demo_prompt,
            image_size=args.vision_demo_image_size,
            seed=args.seed,
            trust_remote_code=args.trust_remote_code,
            atol=args.vision_demo_atol,
            rtol=args.vision_demo_rtol,
            generation_tokens=args.vision_demo_generation_tokens,
            require_equal=False,
        )


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    main()
