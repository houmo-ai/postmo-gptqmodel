# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

import argparse
import gc
import json
import logging
import math
import os
import random
from contextlib import contextmanager
from datetime import datetime

import torch
from datasets import load_dataset
from transformers import AutoConfig, AutoTokenizer

from gptqmodel import GPTQModel, QuantizeConfig
from gptqmodel.quantization.config import ExpertsRoutingBypass, ExpertsRoutingOverride, MoEConfig
from gptqmodel.utils.missing_tensors import (
    count_checkpoint_tensors_with_prefix,
    copy_missing_tensors_from_source,
    validate_tensor_prefix_parity,
)


def get_wikitext2(nsamples, seqlen):
    traindata = load_dataset("wikitext", "wikitext-2-raw-v1", split="train").filter(
        lambda x: len(x["text"]) >= seqlen)

    return [example["text"] for example in traindata.select(range(nsamples))]


def get_jsonl_texts(path, nsamples, text_key="text"):
    samples = []

    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            record = json.loads(line)
            text = record.get(text_key)
            if not isinstance(text, str) or not text.strip():
                raise ValueError(f"Invalid `{text_key}` at {path}:{line_no}")

            samples.append(text)
            if len(samples) >= nsamples:
                break

    if len(samples) == 0:
        raise ValueError(f"No calibration samples found in {path}")

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

    all = ppl.calculate(n_ctx=512, n_batch=512)

    avg = sum(all) / len(all)

    return avg


def _get_model_device(model) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        device = getattr(model, "device", "cpu")
        return torch.device(device)


@contextmanager
def temporary_ppl_device(model, target_device: str | None):
    original_device = _get_model_device(model)
    active_device = str(original_device)
    moved = False
    dispatched = False

    if target_device is None:
        target_device = str(original_device)

    if original_device.type == "cpu" and target_device != "cpu" and torch.cuda.is_available():
        if target_device == "auto":
            if torch.cuda.device_count() > 1:
                try:
                    from accelerate import infer_auto_device_map
                    from accelerate.hooks import remove_hook_from_module
                    from accelerate.utils import get_balanced_memory

                    from gptqmodel.utils.model import simple_dispatch_model

                    no_split = list(getattr(model, "_no_split_modules", None) or [])
                    memory_kwargs = {}
                    if no_split:
                        memory_kwargs["no_split_module_classes"] = no_split

                    max_memory = get_balanced_memory(model, **memory_kwargs)
                    device_map = infer_auto_device_map(model, max_memory=max_memory, **memory_kwargs)
                    print("[ppl] Dispatching model to GPU(s) with accelerate device_map=auto for evaluation ...")
                    simple_dispatch_model(model, device_map)
                    active_device = next(
                        (dev for dev in device_map.values() if dev not in ("cpu", "disk")),
                        "cuda:0",
                    )
                    dispatched = True
                except Exception as exc:
                    target_device = "cuda:0"
                    print(f"[ppl] Accelerate auto dispatch failed ({exc}); falling back to {target_device}.")
            else:
                target_device = "cuda:0"

        if not dispatched and target_device != "auto":
            print(f"[ppl] Moving model from cpu to {target_device} for evaluation ...")
            model.to(target_device)
            active_device = target_device
            moved = True

    try:
        yield active_device
    finally:
        if dispatched:
            from accelerate.hooks import remove_hook_from_module

            print("[ppl] Moving accelerate-dispatched model back to cpu after evaluation ...")
            remove_hook_from_module(model, recurse=True)
            model.to("cpu")
            model.hf_device_map = {"": "cpu"}
            torch.cuda.empty_cache()
        elif moved:
            print("[ppl] Moving model back to cpu after evaluation ...")
            model.to("cpu")
            torch.cuda.empty_cache()


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


@torch.inference_mode()
def capture_last_token_logits(model, inputs):
    embed_device = model.get_input_embeddings().weight.device
    if embed_device.type == "meta":
        raise RuntimeError("Cannot run the precision gate with meta-device input embeddings.")
    device_inputs = {
        key: value.to(embed_device) if isinstance(value, torch.Tensor) else value
        for key, value in inputs.items()
    }
    return model(**device_inputs, use_cache=False).logits[0, -1].detach().float().cpu()


def compare_last_token_logits(reference, candidate):
    reference = reference.detach().float().reshape(-1).cpu()
    candidate = candidate.detach().float().reshape(-1).cpu()
    if reference.shape != candidate.shape:
        raise ValueError(f"Logit shape mismatch: {tuple(reference.shape)} != {tuple(candidate.shape)}")
    diff = (reference - candidate).abs()
    reference_top1 = int(reference.argmax().item())
    candidate_top1 = int(candidate.argmax().item())
    return {
        "cosine_similarity": float(
            torch.nn.functional.cosine_similarity(reference.unsqueeze(0), candidate.unsqueeze(0)).item()
        ),
        "max_abs_error": float(diff.max().item()),
        "mean_abs_error": float(diff.mean().item()),
        "reference_top1": reference_top1,
        "candidate_top1": candidate_top1,
        "top1_match": reference_top1 == candidate_top1,
    }


def enforce_precision_gate(metrics, min_logit_cosine, require_top1_match):
    cosine = float(metrics["cosine_similarity"])
    if not math.isfinite(cosine) or cosine < min_logit_cosine:
        raise RuntimeError(
            "saved-checkpoint precision gate failed: "
            f"cosine_similarity={cosine:.8f} < {min_logit_cosine:.8f}"
        )
    if require_top1_match and not metrics["top1_match"]:
        raise RuntimeError(
            "saved-checkpoint precision gate failed: "
            f"top-1 changed from {metrics['reference_top1']} to {metrics['candidate_top1']}"
        )


def resolve_experts_implementation(requested):
    if requested != "auto":
        return requested
    if not torch.cuda.is_available():
        return "eager"
    capabilities = [
        torch.cuda.get_device_capability(device_idx)
        for device_idx in range(torch.cuda.device_count())
    ]
    return "grouped_mm" if capabilities and all(major >= 9 for major, _ in capabilities) else "eager"


def configure_experts_implementation(model, requested):
    implementation = resolve_experts_implementation(requested)
    setter = getattr(model, "set_experts_implementation", None)
    if callable(setter):
        setter(implementation)
    else:
        model.config._experts_implementation = implementation
    print(f"[experts] Using Transformers experts implementation: {implementation}")
    return implementation


def preserve_mtp_tensors_if_present(source_dir, target_dir, rotation):
    """Preserve optional Qwen3-Next MTP tensors after the target model is saved."""

    source_mtp_count = count_checkpoint_tensors_with_prefix(source_dir, "mtp.")
    if source_mtp_count == 0:
        print("[mtp] Source checkpoint has no mtp.* tensors; keeping target-only output.")
        return 0
    if rotation:
        print("[mtp] Rotation is enabled; unrotated MTP tensors were not restored.")
        return 0

    copied = copy_missing_tensors_from_source(
        source_dir=source_dir,
        target_dir=target_dir,
        tensor_name_prefixes=("mtp.",),
    )
    mtp_count = validate_tensor_prefix_parity(source_dir, target_dir, "mtp.")
    print(f"[mtp] Validated {mtp_count} tensors; copied {copied} to the BF16 sidecar.")
    return mtp_count


def build_moe_config(args):
    if args.moe_routing == "none":
        return None

    if args.moe_routing == "bypass":
        return MoEConfig(routing=ExpertsRoutingBypass(batch_size=args.moe_routing_batch_size))

    return MoEConfig(routing=ExpertsRoutingOverride(num_experts_per_tok=args.moe_num_experts_per_tok))


def layer_limit_dynamic_skips(num_hidden_layers, max_quant_layers):
    if max_quant_layers is None:
        return {}
    if max_quant_layers < 0 or max_quant_layers > num_hidden_layers:
        raise ValueError(
            f"--max-quant-layers must be between 0 and {num_hidden_layers}, got {max_quant_layers}"
        )
    return {
        rf"-:.*\.layers\.{layer_idx}\..*": {}
        for layer_idx in range(max_quant_layers, num_hidden_layers)
    }


def build_dynamic_quant_config(args, num_hidden_layers):
    dynamic = layer_limit_dynamic_skips(num_hidden_layers, args.max_quant_layers)
    dynamic.update({
        r".*\.self_attn\.(q_proj|k_proj|v_proj|o_proj)$": {"bits": args.self_attn_bits},
        r".*\.linear_attn\.(in_proj_qkvz|out_proj)$": {"bits": args.linear_attn_bits},
        r".*\.mlp\.(gate_proj|up_proj|down_proj)$": {"bits": args.dense_mlp_bits},
        r".*\.mlp\.experts\.\d+\.(gate_proj|up_proj|down_proj)$": {"bits": args.expert_bits},
        r".*\.mlp\.shared_expert\.(gate_proj|up_proj|down_proj)$": {"bits": args.shared_expert_bits},
    })
    return dynamic


def build_default_output_dir(args):
    model_name = os.path.basename(os.path.normpath(args.model))
    bit_profile = (
        f"full{args.self_attn_bits}-linear{args.linear_attn_bits}-dense{args.dense_mlp_bits}-"
        f"expert{args.expert_bits}-shared{args.shared_expert_bits}-base{args.bits}"
    )
    time_suffix = datetime.now().strftime("%m%d%H")
    return os.path.join("work_dirs", f"{model_name}-gptq-{bit_profile}-{args.group_size}g_{time_suffix}")


def parse_args():
    parser = argparse.ArgumentParser(description="Qwen3-Next GPTQ quantization + generation validation")
    parser.add_argument("--model", type=str, default="/data02/datasets/Qwen3-Next-80B-A3B-Instruct", help="fp model path")
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help=(
            "quantized model output directory, defaults to "
            "work_dirs/<model>-gptq-full<self-attn>-linear<linear-attn>-dense<dense>-"
            "expert<expert>-shared<shared>-base<bits>-<group_size>g"
        ),
    )
    parser.add_argument("--bits", type=int, default=4, choices=[2, 3, 4, 5, 8])
    parser.add_argument(
        "--self-attn-bits",
        type=int,
        default=8,
        choices=[2, 3, 4, 5, 8],
        help="bit width for full-attention q/k/v/o projections",
    )
    parser.add_argument(
        "--linear-attn-bits",
        type=int,
        default=8,
        choices=[2, 3, 4, 5, 8],
        help="bit width for linear-attention in_proj_qkvz/out_proj",
    )
    parser.add_argument(
        "--dense-mlp-bits",
        type=int,
        default=4,
        choices=[2, 3, 4, 5, 8],
        help="bit width for dense MLP gate/up/down projections",
    )
    parser.add_argument(
        "--expert-bits",
        "--non-shared-expert-bits",
        type=int,
        default=4,
        choices=[2, 3, 4, 5, 8],
        help="bit width for non-shared MoE experts",
    )
    parser.add_argument(
        "--shared-expert-bits",
        type=int,
        default=4,
        choices=[2, 3, 4, 5, 8],
        help="bit width for shared MoE expert",
    )
    parser.add_argument("--group-size", type=int, default=64)
    parser.add_argument(
        "--max-quant-layers",
        type=int,
        default=None,
        help="quantize decoder layers [0, N); later layers remain BF16",
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--nsamples", type=int, default=256)
    parser.add_argument("--seqlen", type=int, default=1024, help="minimum sequence length for wikitext2 filtering")
    parser.add_argument(
        "--calibration-jsonl",
        type=str,
        default=None,
        help="optional JSONL calibration dataset path; overrides wikitext2 when provided",
    )
    parser.add_argument(
        "--calibration-text-key",
        type=str,
        default="text",
        help="JSON field name used to read text from --calibration-jsonl",
    )
    parser.add_argument("--device", type=str, default=None, help="quantization compute device (single GPU)")
    parser.add_argument("--infer-device", type=str, default=None, help="inference device after quantization (single GPU)")
    parser.add_argument(
        "--device-map",
        type=str,
        default=None,
        choices=["auto", "balanced", "balanced_low_0", "sequential"],
        help="multi-GPU device map strategy; overrides --device/--infer-device when set",
    )
    parser.add_argument(
        "--infer-device-map",
        type=str,
        default=None,
        choices=["auto", "balanced", "balanced_low_0", "sequential"],
        help="optional multi-GPU map used only when reloading the saved checkpoint",
    )
    parser.add_argument(
        "--experts-implementation",
        choices=["auto", "eager", "batched_mm", "grouped_mm"],
        default="auto",
        help="Transformers MoE kernel; auto avoids grouped_mm on pre-Hopper GPUs",
    )
    parser.add_argument(
        "--auto-forward-data-parallel",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "replicate decoder blocks across all visible GPUs during calibration forward; "
            "disabled by default because Transformers Qwen3-Next layers contain parameters "
            "that are not safely rehomed by GPTQModel's generic replica path"
        ),
    )
    parser.add_argument("--offload-to-disk", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--offload-path",
        type=str,
        default="./gptqmodel_offload_qwen3next",
        help="offload path when --offload-to-disk is enabled",
    )
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--prompt", type=str, default="请介绍一下 GPTQ 量化的核心思想。")
    parser.add_argument(
        "--check-saved-precision",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="compare source BF16 and reloaded quantized last-token logits",
    )
    parser.add_argument("--min-logit-cosine", type=float, default=0.99)
    parser.add_argument(
        "--require-top1-match",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="require source and saved checkpoints to select the same next token",
    )
    parser.add_argument(
        "--skip-ppl",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="skip the final wikitext perplexity evaluation",
    )
    parser.add_argument("--trust-remote-code", action="store_true", default=False)
    parser.add_argument(
        "--rotation",
        type=str,
        default=None,
        choices=["random", "hadamard", "block_hadamard"],
        help="可选 QuaRot 旋转模式",
    )
    parser.add_argument(
        "--check-rotation-ppl",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="evaluate wikitext2 perplexity before and after rotation, before quantization starts",
    )
    parser.add_argument(
        "--rotated-out",
        type=str,
        default=None,
        help="optional output dir for the rotated-but-not-yet-quantized fp model",
    )
    parser.add_argument(
        "--stop-after-rotate",
        action="store_true",
        default=False,
        help="apply rotation, optionally save/evaluate it, then exit without quantizing",
    )
    parser.add_argument(
        "--hessian-mse",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="enable Hessian MSE assisted optimization",
    )
    parser.add_argument(
        "--moe-routing",
        type=str,
        default="bypass",
        choices=["none", "bypass", "override"],
        help="MoE routing mode: none keeps model routing, bypass forwards calibration to all experts, override forces top-k routing",
    )
    parser.add_argument(
        "--moe-routing-batch-size",
        type=int,
        default=None,
        help="Batch size for bypass routing to reduce VRAM pressure",
    )
    parser.add_argument(
        "--moe-num-experts-per-tok",
        type=str,
        default="all",
        help="Override top-k experts per token when --moe-routing=override; accepts a positive integer or 'all'",
    )
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
    args.out = args.out or build_default_output_dir(args)

    if not 0.0 <= args.min_logit_cosine <= 1.0:
        raise ValueError("--min-logit-cosine must be between 0 and 1")
    if args.check_saved_precision and args.skip_quantize:
        raise ValueError("--check-saved-precision cannot be combined with --skip-quantize")
    if args.check_saved_precision and args.rotation:
        raise ValueError("--check-saved-precision currently requires rotation to be disabled")

    if (args.check_rotation_ppl or args.rotated_out or args.stop_after_rotate) and not args.rotation:
        raise ValueError(
            "--check-rotation-ppl/--rotated-out/--stop-after-rotate require --rotation to be set."
        )

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    tokenizer_source = args.model
    reference_logits = None
    precision_inputs = None
    if not args.skip_quantize:
        tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True, trust_remote_code=args.trust_remote_code)
        calibration_dataset, calibration_source = build_calibration_dataset(args)
        print(f"[1/5] Calibration samples ({calibration_source}): {len(calibration_dataset)}")

        moe_config = build_moe_config(args)
        model_config = AutoConfig.from_pretrained(args.model, trust_remote_code=args.trust_remote_code)
        dynamic_config = build_dynamic_quant_config(args, int(model_config.num_hidden_layers))

        print(
            "[config] bits: "
            f"base={args.bits}, self_attn={args.self_attn_bits}, linear_attn={args.linear_attn_bits}, "
            f"dense_mlp={args.dense_mlp_bits}, expert={args.expert_bits}, shared_expert={args.shared_expert_bits}"
        )

        quantize_config = QuantizeConfig(
            bits=args.bits,
            dynamic=dynamic_config,
            group_size=args.group_size,
            device=args.device,
            rotation=args.rotation,
            hessian_mse=args.hessian_mse,
            moe=moe_config,
            offload_to_disk=args.offload_to_disk,
            offload_to_disk_path=args.offload_path if args.offload_to_disk else None,
            auto_forward_data_parallel=args.auto_forward_data_parallel,
        )

        print("[2/5] Loading fp model ...")
        load_kwargs = dict(trust_remote_code=args.trust_remote_code)
        if args.device_map:
            load_kwargs["device_map"] = args.device_map
        model = GPTQModel.load(args.model, quantize_config, **load_kwargs)
        configure_experts_implementation(model.model, args.experts_implementation)

        if args.check_saved_precision:
            precision_inputs = build_chat_inputs(tokenizer, args.prompt, "cpu")
            print("[precision] Capturing source BF16 last-token logits ...")
            with temporary_ppl_device(model.model, "auto"):
                reference_logits = capture_last_token_logits(model.model, precision_inputs)

        rotation_artifact_requested = any(
            [args.check_rotation_ppl, args.rotated_out is not None, args.stop_after_rotate]
        )
        if args.rotation and rotation_artifact_requested:
            if args.check_rotation_ppl:
                print("[rotation] Evaluating fp model PPL on wikitext2 before rotation ...")
                with temporary_ppl_device(model.model, "auto") as ppl_device:
                    fp_ppl_before = calculate_avg_ppl(model.model, tokenizer, device=ppl_device)
                print(f"[rotation] FP model avg PPL before rotation: {fp_ppl_before:.4f}")

            print(f"[rotation] Applying {args.rotation} rotation to fp model ...")
            model.apply_rotation()

            if args.check_rotation_ppl:
                print("[rotation] Evaluating fp model PPL on wikitext2 after rotation ...")
                with temporary_ppl_device(model.model, "auto") as ppl_device:
                    fp_ppl_after = calculate_avg_ppl(model.model, tokenizer, device=ppl_device)
                print(f"[rotation] FP model avg PPL after rotation: {fp_ppl_after:.4f}")
                print(f"[rotation] PPL delta: {fp_ppl_after - fp_ppl_before:+.6f}")

            if args.rotated_out:
                print(f"[rotation] Saving rotated fp model to: {args.rotated_out}")
                os.makedirs(args.rotated_out, exist_ok=True)
                model.model.save_pretrained(args.rotated_out)
                tokenizer.save_pretrained(args.rotated_out)

            if args.stop_after_rotate:
                print("[rotation] Stop requested after rotation; skipping quantization.")
                return

        print("[3/5] Running GPTQ quantization ...")
        model.quantize(calibration_dataset, batch_size=args.batch_size)
        model.save(args.out)
        print(f"Quantized model saved to: {args.out}")
        preserve_mtp_tensors_if_present(args.model, args.out, args.rotation)
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    else:
        print(f"[1/5] Skip quantization, using existing quantized model: {args.out}")
        tokenizer = AutoTokenizer.from_pretrained(args.out, use_fast=True, trust_remote_code=args.trust_remote_code)
        tokenizer_source = args.out

    infer_device = args.infer_device or args.device
    infer_kwargs = dict(trust_remote_code=args.trust_remote_code)
    infer_device_map = args.infer_device_map or args.device_map
    if infer_device_map:
        infer_kwargs["device_map"] = infer_device_map
        print(f"[4/5] Loading quantized model with device_map='{infer_device_map}' and generating ...")
    else:
        infer_kwargs["device"] = infer_device
        print(f"[4/5] Loading quantized model on {infer_device} and generating ...")
    model = GPTQModel.load(args.out, **infer_kwargs)
    configure_experts_implementation(model.model, args.experts_implementation)
    runtime_device = str(model.device)
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, use_fast=True, trust_remote_code=args.trust_remote_code)

    if reference_logits is not None:
        print("[precision] Capturing reloaded checkpoint last-token logits ...")
        candidate_logits = capture_last_token_logits(model.model, precision_inputs)
        metrics = compare_last_token_logits(reference_logits, candidate_logits)
        precision_path = os.path.join(args.out, "precision_gate.json")
        with open(precision_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, sort_keys=True)
            f.write("\n")
        print(f"[precision] Metrics saved to {precision_path}: {metrics}")
        enforce_precision_gate(metrics, args.min_logit_cosine, args.require_top1_match)
    inputs = build_chat_inputs(tokenizer, args.prompt, runtime_device)

    with torch.inference_mode():
        output_ids = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)

    generated_ids = output_ids[:, inputs["input_ids"].shape[1]:]
    answer = tokenizer.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    print("=== Prompt ===")
    print(args.prompt)
    print("=== Answer ===")
    print(answer)

    if args.skip_ppl:
        print("[5/5] Skipping perplexity evaluation (--skip-ppl).")
    else:
        print("[5/5] Evaluating perplexity on wikitext2 ...")
        avg_ppl = calculate_avg_ppl(model, tokenizer)
        print(f"Quantized model avg PPL (wikitext2): {avg_ppl:.4f}")


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    main()
