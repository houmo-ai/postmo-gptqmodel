# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

import argparse
import os
import random

import torch
from transformers import AutoTokenizer

from gptqmodel import GPTQModel, QuantizeConfig


CALIBRATION_TEXTS = [
    "Qwen3.5 combines linear-attention and full-attention layers for better long-context efficiency.",
    "GPTQ is a post-training quantization method that minimizes weight reconstruction error.",
    "Smaller quantized models can reduce memory usage while keeping useful generation quality.",
    "在部署前需要用端到端推理验证量化模型是否还能稳定回答。",
    "请简要说明为什么要在量化后做回归测试。",
]


def build_calibration_dataset(tokenizer, nsamples: int, max_length: int):
    dataset = []
    for i in range(nsamples):
        text = CALIBRATION_TEXTS[i % len(CALIBRATION_TEXTS)]
        encoded = tokenizer(text, truncation=True, max_length=max_length, return_tensors="pt")
        dataset.append(
            {
                "input_ids": encoded["input_ids"],
                "attention_mask": encoded["attention_mask"],
            }
        )
    return dataset


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


def parse_args():
    parser = argparse.ArgumentParser(description="Qwen3.5 dense GPTQ quantization + generation validation")
    parser.add_argument("--model", type=str, default="/data02/datasets/Qwen3.5-4B", help="fp model path")
    parser.add_argument(
        "--out",
        type=str,
        default="/data02/datasets/Qwen3.5-4B-gptq-4bit",
        help="quantized model output directory",
    )
    parser.add_argument("--bits", type=int, default=4, choices=[2, 3, 4, 5, 8])
    parser.add_argument("--group-size", type=int, default=64)
    parser.add_argument("--rotation", type=str, default=None, choices=["hadamard", "block_hadamard", "random"])
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--nsamples", type=int, default=8)
    parser.add_argument("--max-calib-length", type=int, default=1024)
    parser.add_argument("--device", type=str, default="cuda:1", help="quantization device")
    parser.add_argument("--infer-device", type=str, default=None, help="inference device after quantization")
    parser.add_argument("--offload-to-disk", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--offload-path",
        type=str,
        default="./gptqmodel_offload_qwen35dense",
        help="offload path when --offload-to-disk is enabled",
    )
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--prompt", type=str, default="请介绍一下 GPTQ 量化的核心思想。")
    parser.add_argument("--trust-remote-code", action="store_true", default=False)
    parser.add_argument(
        "--skip-quantize",
        action="store_true",
        default=False,
        help="skip quantization and directly run generation on --out",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    tokenizer_source = args.model
    if not args.skip_quantize:
        tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True, trust_remote_code=args.trust_remote_code)
        calibration_dataset = build_calibration_dataset(tokenizer, nsamples=args.nsamples, max_length=args.max_calib_length)
        print(f"[1/4] Calibration samples: {len(calibration_dataset)}")

        quantize_config = QuantizeConfig(
            bits=args.bits,
            group_size=args.group_size,
            rotation=args.rotation,
            device=args.device,
            offload_to_disk=args.offload_to_disk,
            offload_to_disk_path=args.offload_path if args.offload_to_disk else None,
        )

        print("[2/4] Loading fp model ...")
        model = GPTQModel.load(args.model, quantize_config, trust_remote_code=args.trust_remote_code)

        print("[3/4] Running GPTQ quantization ...")
        model.quantize(calibration_dataset, batch_size=args.batch_size)
        model.save(args.out)
        print(f"Quantized model saved to: {args.out}")
    else:
        print(f"[1/4] Skip quantization, using existing quantized model: {args.out}")
        tokenizer = AutoTokenizer.from_pretrained(args.out, use_fast=True, trust_remote_code=args.trust_remote_code)
        tokenizer_source = args.out

    infer_device = args.infer_device or args.device
    print(f"[4/4] Loading quantized model on {infer_device} and generating ...")
    model = GPTQModel.load(args.out, device=infer_device, trust_remote_code=args.trust_remote_code)
    runtime_device = str(model.device)
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, use_fast=True, trust_remote_code=args.trust_remote_code)
    inputs = build_chat_inputs(tokenizer, args.prompt, runtime_device)

    with torch.inference_mode():
        output_ids = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)

    generated_ids = output_ids[:, inputs["input_ids"].shape[1] :]
    answer = tokenizer.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    print("=== Prompt ===")
    print(args.prompt)
    print("=== Answer ===")
    print(answer)


if __name__ == "__main__":
    main()
