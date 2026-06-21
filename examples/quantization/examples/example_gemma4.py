# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

import argparse
import json
import logging
import os
import random
import sys
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoTokenizer


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gptqmodel import GPTQModel, QuantizeConfig
from gptqmodel.quantization.config import ExpertsRoutingBypass, ExpertsRoutingOverride, MoEConfig
from gptqmodel.recipes.gemma4 import _copy_hf_auxiliary_files, _patch_gemma4_quantization_config


def get_wikitext2(nsamples, seqlen):
    traindata = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    texts = [example["text"].strip() for example in traindata if example["text"].strip()]
    long_texts = [text for text in texts if len(text) >= seqlen]
    if len(long_texts) >= nsamples:
        return long_texts[:nsamples]

    logging.warning(
        "Only found %s wikitext2 rows with len(text) >= %s; "
        "building contiguous calibration chunks to reach nsamples=%s",
        len(long_texts),
        seqlen,
        nsamples,
    )
    corpus = "\n\n".join(texts)
    samples = list(long_texts)
    cursor = 0
    while len(samples) < nsamples and cursor < len(corpus):
        chunk = corpus[cursor: cursor + seqlen]
        if chunk.strip():
            samples.append(chunk)
        cursor += seqlen
    if len(samples) < nsamples:
        logging.warning(
            "Requested %s wikitext2 calibration sample(s), but only built %s sample(s)",
            nsamples,
            len(samples),
        )
    return samples


def get_jsonl_texts(path, nsamples, text_key="text"):
    samples = []
    skipped_blank = 0
    with open(path, "r", encoding="utf-8") as file:
        for line_no, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            text = record.get(text_key)
            if not isinstance(text, str):
                raise ValueError(f"Invalid `{text_key}` at {path}:{line_no}")
            if not text.strip():
                skipped_blank += 1
                logging.warning("Skipping blank `%s` at %s:%s", text_key, path, line_no)
                continue
            samples.append(text)
            if len(samples) >= nsamples:
                break

    if len(samples) == 0:
        raise ValueError(f"No calibration samples found in {path}")
    if skipped_blank:
        logging.warning("Skipped %s blank calibration sample(s) from %s", skipped_blank, path)
    if len(samples) < nsamples:
        logging.warning(
            "Requested %s calibration sample(s) from %s, but only found %s valid sample(s)",
            nsamples,
            path,
            len(samples),
        )
    return samples


def build_calibration_dataset(args):
    if args.calibration_jsonl:
        calibration_dataset = get_jsonl_texts(
            path=args.calibration_jsonl,
            nsamples=args.nsamples,
            text_key=args.calibration_text_key,
        )
        source_name = os.path.basename(args.calibration_jsonl)
        return calibration_dataset, f"jsonl:{source_name}"

    return get_wikitext2(nsamples=args.nsamples, seqlen=args.seqlen), "wikitext2"


@torch.inference_mode()
def calculate_avg_ppl(model, tokenizer, device=None):
    from gptqmodel.utils.perplexity import Perplexity

    ppl = Perplexity(
        model=model,
        tokenizer=tokenizer,
        dataset_path="wikitext",
        dataset_name="wikitext-2-raw-v1",
        split="train",
        text_column="text",
        device=device,
    )
    all_ppl = ppl.calculate(n_ctx=512, n_batch=512)
    return sum(all_ppl) / len(all_ppl)


def build_chat_inputs(tokenizer, prompt: str, device: str):
    messages = [{"role": "user", "content": prompt}]
    if hasattr(tokenizer, "apply_chat_template"):
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    else:
        text = prompt
    return tokenizer(text, return_tensors="pt").to(device)


def build_moe_config(args):
    if args.moe_routing == "none":
        return None
    if args.moe_routing == "bypass":
        return MoEConfig(routing=ExpertsRoutingBypass(batch_size=args.moe_routing_batch_size))
    n_per_tok = args.moe_num_experts_per_tok
    if isinstance(n_per_tok, str) and n_per_tok != "all":
        n_per_tok = int(n_per_tok)
    return MoEConfig(routing=ExpertsRoutingOverride(num_experts_per_tok=n_per_tok))


def parse_args():
    parser = argparse.ArgumentParser(description="Gemma4 GPTQ quantization + generation validation")
    parser.add_argument(
        "--model",
        type=str,
        default="/data01/home/chenzx/model/gemma-4-26B-A4B-it",
        help="fp model path",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="/data01/home/chenzx/model/gemma-4-26B-A4B-it-gptq-4bit",
        help="quantized model output directory",
    )
    parser.add_argument("--bits", type=int, default=4, choices=[2, 3, 4, 5, 8])
    parser.add_argument("--group-size", type=int, default=64)
    parser.add_argument(
        "--mse",
        type=float,
        default=2.4,
        help="mse use 2.4 to default",
    )
    parser.add_argument(
        "--hessian-mse",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="enable Hessian MSE assisted optimization",
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--nsamples", type=int, default=256)
    parser.add_argument("--seqlen", type=int, default=1024, help="minimum sequence length for wikitext2 filtering")
    parser.add_argument(
        "--calibration-jsonl",
        type=str,
        default=None,
        help="optional JSONL calibration dataset path; uses wikitext2 when omitted",
    )
    parser.add_argument(
        "--calibration-text-key",
        type=str,
        default="text",
        help="JSON field name used to read text from --calibration-jsonl",
    )
    parser.add_argument("--device", type=str, default=None, help="quantization compute device")
    parser.add_argument("--infer-device", type=str, default=None, help="inference device after quantization")
    parser.add_argument(
        "--device-map",
        type=str,
        default=None,
        choices=["auto", "balanced", "balanced_low_0", "sequential"],
        help="multi-GPU device map strategy; overrides --device/--infer-device when set",
    )
    parser.add_argument("--offload-to-disk", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--offload-path",
        type=str,
        default="./gptqmodel_offload_gemma4",
        help="offload path when --offload-to-disk is enabled",
    )
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--prompt", type=str, default="Please introduce the core idea of GPTQ quantization.")
    parser.add_argument("--trust-remote-code", action="store_true", default=False)
    parser.add_argument(
        "--skip-quantize",
        action="store_true",
        default=False,
        help="skip quantization and directly run generation on --out",
    )
    parser.add_argument(
        "--skip-eval",
        action="store_true",
        default=False,
        help="skip post-quantization generation and perplexity evaluation",
    )
    parser.add_argument(
        "--moe-routing",
        type=str,
        default="none",
        choices=["none", "bypass", "override"],
        help="MoE routing mode: bypass forwards calibration to all experts.",
    )
    parser.add_argument(
        "--moe-routing-batch-size",
        type=int,
        default=None,
        help="Batch size for MoE bypass routing to reduce VRAM pressure.",
    )
    parser.add_argument(
        "--moe-num-experts-per-tok",
        type=str,
        default="all",
        help="Override top-k experts per token when --moe-routing=override.",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    tokenizer_source = args.model
    tokenizer = None
    if not args.skip_quantize:
        tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True, trust_remote_code=args.trust_remote_code)
        calibration_dataset, calibration_source = build_calibration_dataset(args)
        print(f"[1/5] Calibration samples ({calibration_source}): {len(calibration_dataset)}")

        quantize_config = QuantizeConfig(
            bits=args.bits,
            group_size=args.group_size,
            hessian_mse=args.hessian_mse,
            mse=args.mse,
            device=args.device,
            moe=build_moe_config(args),
            offload_to_disk=args.offload_to_disk,
            offload_to_disk_path=args.offload_path if args.offload_to_disk else None,
        )

        print("[2/5] Loading fp model ...")
        load_kwargs = dict(trust_remote_code=args.trust_remote_code)
        if args.device_map:
            load_kwargs["device_map"] = args.device_map
        model = GPTQModel.load(args.model, quantize_config, **load_kwargs)

        print("[3/5] Running GPTQ quantization ...")
        model.quantize(calibration_dataset, batch_size=args.batch_size)
        model.save(args.out)
        _copy_hf_auxiliary_files(args.model, args.out)
        _patch_gemma4_quantization_config(args.out)
        print(f"Quantized model saved to: {args.out}")
    else:
        print(f"[1/5] Skip quantization, using existing quantized model: {args.out}")
        tokenizer = AutoTokenizer.from_pretrained(args.out, use_fast=True, trust_remote_code=args.trust_remote_code)
        tokenizer_source = args.out

    if args.skip_eval:
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

    # print("[5/5] Evaluating perplexity on wikitext2 ...")
    # avg_ppl = calculate_avg_ppl(model, tokenizer)
    # print(f"Quantized model avg PPL (wikitext2): {avg_ppl:.4f}")


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    main()
