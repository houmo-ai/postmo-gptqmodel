#!/usr/bin/env python3
"""Run the native FP8/FP4 DeepSeek-V4 checkpoint as a numerical baseline."""

from __future__ import annotations

import argparse
from contextlib import ExitStack
import gc
import json
import re
import time
from pathlib import Path

import torch
from accelerate import dispatch_model
from safetensors import safe_open
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.utils import logging as transformers_logging


def build_inputs(tokenizer, prompt: str) -> dict[str, torch.Tensor]:
    messages = [{"role": "user", "content": prompt}]
    if getattr(tokenizer, "chat_template", None):
        tokenized = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        )
        return dict(tokenized)
    return dict(tokenizer(prompt, return_tensors="pt"))


def tensor_summary(value: torch.Tensor) -> dict[str, float | list[int]]:
    value = value.detach().float()
    finite = torch.isfinite(value)
    result: dict[str, float | list[int]] = {
        "shape": list(value.shape),
        "finite_fraction": float(finite.float().mean()),
    }
    if not finite.all():
        return result
    result.update(
        {
            "rms": float(value.square().mean().sqrt()),
            "std": float(value.std()),
            "abs_max": float(value.abs().max()),
        }
    )
    if value.ndim >= 3 and value.shape[1] > 1:
        token_delta = value[:, 1:] - value[:, :-1]
        result["adjacent_token_delta_rms"] = float(
            token_delta.square().mean().sqrt()
        )
        first = value[:, 0].flatten(1)
        last = value[:, -1].flatten(1)
        result["first_last_cosine"] = float(
            torch.nn.functional.cosine_similarity(first, last).mean()
        )
    if value.ndim == 4 and value.shape[2] > 1:
        stream_delta = value[:, :, 1:] - value[:, :, :-1]
        result["adjacent_stream_delta_rms"] = float(
            stream_delta.square().mean().sqrt()
        )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument(
        "--prompt",
        default="请简要解释为什么天空看起来是蓝色的。",
    )
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--dtype", choices=("float16", "bfloat16"), default="bfloat16")
    parser.add_argument(
        "--cpu-layers",
        type=int,
        default=0,
        help="Put this many leading layers on CPU and evenly spread the rest over visible GPUs.",
    )
    parser.add_argument(
        "--all-gpu",
        action="store_true",
        help=(
            "Construct the BF16 model temporarily on CPU, then dispatch every parameter "
            "and buffer across all visible GPUs. The final model has no runtime offload."
        ),
    )
    parser.add_argument(
        "--direct-all-gpu",
        action="store_true",
        help=(
            "Load directly into an explicit GPU-only layer map. Use this for "
            "native FP8/FP4 checkpoints that must stay packed during loading."
        ),
    )
    parser.add_argument(
        "--trust-hub-kernels",
        action="store_true",
        help="Allow execution of explicitly requested Hugging Face hub kernels.",
    )
    parser.add_argument(
        "--restore-native-fp32",
        type=Path,
        help="Native checkpoint supplying exact FP32 mHC/sink/router/position tensors.",
    )
    parser.add_argument(
        "--diagnostic-layer",
        type=int,
        help="Capture one layer's mHC, attention, router, expert, and MLP summaries.",
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def native_fp32_source_key(runtime_key: str) -> str | None:
    match = re.fullmatch(
        r"model\.layers\.(\d+)\.(attn_hc|ffn_hc)\.(fn|base|scale)",
        runtime_key,
    )
    if match:
        layer, site, parameter = match.groups()
        site = "attn" if site == "attn_hc" else "ffn"
        return f"layers.{layer}.hc_{site}_{parameter}"
    patterns = (
        (
            r"model\.layers\.(\d+)\.self_attn\.sinks",
            "layers.{}.attn.attn_sink",
        ),
        (
            r"model\.layers\.(\d+)\.self_attn\.compressor\.position_bias",
            "layers.{}.attn.compressor.ape",
        ),
        (
            r"model\.layers\.(\d+)\.self_attn\.compressor\.indexer\.position_bias",
            "layers.{}.attn.indexer.compressor.ape",
        ),
        (
            r"model\.layers\.(\d+)\.mlp\.gate\.e_score_correction_bias",
            "layers.{}.ffn.gate.bias",
        ),
    )
    for pattern, template in patterns:
        match = re.fullmatch(pattern, runtime_key)
        if match:
            return template.format(match.group(1))
    return {
        "model.hc_head.hc_fn": "hc_head_fn",
        "model.hc_head.hc_base": "hc_head_base",
        "model.hc_head.hc_scale": "hc_head_scale",
    }.get(runtime_key)


def build_cpu_layer_device_map(
    model_dir: Path,
    *,
    cpu_layers: int,
) -> dict[str, int | str]:
    config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    num_layers = int(config["num_hidden_layers"])
    num_gpus = torch.cuda.device_count()
    if num_gpus <= 0:
        raise RuntimeError("At least one visible GPU is required")
    cpu_layers = int(cpu_layers)
    if not 0 < cpu_layers < num_layers:
        raise ValueError(f"cpu_layers must be in [1, {num_layers - 1}], got {cpu_layers}")
    remaining = num_layers - cpu_layers
    if remaining % num_gpus:
        raise ValueError(
            f"{remaining} GPU layers cannot be evenly distributed over {num_gpus} GPUs"
        )
    layers_per_gpu = remaining // num_gpus
    device_map: dict[str, int | str] = {
        "model.embed_tokens": 0,
        "model.rotary_emb": num_gpus - 1,
        "model.hc_head": num_gpus - 1,
        "model.norm": num_gpus - 1,
        "lm_head": num_gpus - 1,
    }
    for layer in range(cpu_layers):
        device_map[f"model.layers.{layer}"] = "cpu"
    for layer in range(cpu_layers, num_layers):
        device_map[f"model.layers.{layer}"] = (layer - cpu_layers) // layers_per_gpu
    return device_map


def build_all_gpu_device_map(model_dir: Path) -> dict[str, int]:
    """Build a contiguous, balanced, GPU-only map for the complete model."""

    config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    num_layers = int(config["num_hidden_layers"])
    num_gpus = torch.cuda.device_count()
    if num_gpus <= 0:
        raise RuntimeError("At least one visible GPU is required")
    if num_gpus > num_layers:
        raise ValueError(f"More GPUs ({num_gpus}) than layers ({num_layers}) is unsupported")

    base, extra = divmod(num_layers, num_gpus)
    layer_counts = [base + (gpu < extra) for gpu in range(num_gpus)]
    # Put the two ~1 GiB vocabulary matrices on GPUs with fewer transformer
    # layers. For 43 layers / 8 GPUs this gives 6/6/6/5/5/5/5/5.
    light_gpus = [gpu for gpu, count in enumerate(layer_counts) if count == base]
    if not light_gpus:
        light_gpus = list(range(num_gpus))
    embed_gpu = light_gpus[0]
    head_gpu = light_gpus[1] if len(light_gpus) > 1 else light_gpus[0]
    tail_gpu = light_gpus[-1]

    device_map = {
        "model.embed_tokens": embed_gpu,
        "model.rotary_emb": tail_gpu,
        "model.hc_head": tail_gpu,
        "model.norm": tail_gpu,
        "lm_head": head_gpu,
    }
    layer = 0
    for gpu, count in enumerate(layer_counts):
        for _ in range(count):
            device_map[f"model.layers.{layer}"] = gpu
            layer += 1
    return device_map


def audit_gpu_residency(model) -> dict[str, int]:
    """Fail unless every registered tensor is resident on a CUDA device."""

    bytes_by_device: dict[str, int] = {}
    invalid = []
    for tensor_kind, tensors in (
        ("parameter", model.named_parameters()),
        ("buffer", model.named_buffers()),
    ):
        for name, tensor in tensors:
            device = str(tensor.device)
            if tensor.device.type != "cuda":
                invalid.append(f"{tensor_kind}:{name}={device}")
                continue
            bytes_by_device[device] = bytes_by_device.get(device, 0) + (
                tensor.numel() * tensor.element_size()
            )
    if invalid:
        preview = ", ".join(invalid[:20])
        raise RuntimeError(
            f"GPU-only residency audit failed for {len(invalid)} tensors: {preview}"
        )
    hf_device_map = getattr(model, "hf_device_map", {})
    forbidden = {
        name: device
        for name, device in hf_device_map.items()
        if str(device) in {"cpu", "disk", "meta"}
    }
    if forbidden:
        raise RuntimeError(f"Runtime offload remains in hf_device_map: {forbidden}")
    return bytes_by_device


def enable_dtype_safe_grouped_mm_fallback() -> bool:
    """Match the dtype behavior of Transformers' native grouped-mm path.

    Torch 2.8 cannot run ``torch._grouped_mm`` on SM80, so Transformers uses
    its loop fallback.  The native path explicitly casts LayerNorm's FP32
    output to the BF16 expert-weight dtype, while the fallback omitted that
    cast and consequently fails in ``torch.mm(..., out=...)``.  Keep the
    official FP32 normalizations and apply the same cast at the grouped-GEMM
    boundary.
    """

    from transformers.integrations import moe

    original = moe._grouped_mm
    if getattr(original, "_deepseek_v4_dtype_safe", False):
        return False

    def dtype_safe_grouped_mm(input, weight, offs):
        if input.dtype != weight.dtype:
            input = input.to(weight.dtype)
        return original(input, weight, offs)

    dtype_safe_grouped_mm._deepseek_v4_dtype_safe = True
    moe._grouped_mm = dtype_safe_grouped_mm
    return True


@torch.inference_mode()
def restore_native_fp32(model, native_model: Path) -> int:
    index = json.loads(
        (native_model / "model.safetensors.index.json").read_text(encoding="utf-8")
    )
    weight_map = index["weight_map"]
    runtime_state = model.state_dict()
    pairs = {
        runtime_key: source_key
        for runtime_key in runtime_state
        if (source_key := native_fp32_source_key(runtime_key)) is not None
    }
    if len(pairs) != 406:
        raise ValueError(f"Expected 406 native FP32 mappings, found {len(pairs)}")
    with ExitStack() as stack:
        readers = {}
        for runtime_key, source_key in pairs.items():
            filename = weight_map[source_key]
            if filename not in readers:
                readers[filename] = stack.enter_context(
                    safe_open(
                        native_model / filename,
                        framework="pt",
                        device="cpu",
                    )
                )
            source = readers[filename].get_tensor(source_key)
            destination = runtime_state[runtime_key]
            if source.dtype != torch.float32 or destination.dtype != torch.float32:
                raise TypeError(
                    f"FP32 restoration dtype mismatch for {runtime_key}: "
                    f"{source.dtype} -> {destination.dtype}"
                )
            destination.copy_(source.to(destination.device))
    return len(pairs)


@torch.inference_mode()
def prepare_cpu_layers_for_bf16_gemm(model, cpu_layers: int) -> int:
    """Match CUDA autocast at RMSNorm→Linear boundaries on CPU layers.

    The CPU grouped-mm fallback uses ``torch.mm(..., out=...)`` and therefore
    does not participate in autocast.  V4's strict FP32 RMSNorm weights would
    otherwise promote BF16 hidden states to FP32 immediately before a BF16
    expert GEMM.  The native RMSNorm weights are stored in BF16, so retaining
    that storage dtype on CPU produces the same BF16 GEMM inputs as CUDA
    autocast while all variance reductions still execute in FP32.
    """

    converted = 0
    for layer in model.model.layers[: int(cpu_layers)]:
        for module in layer.modules():
            if type(module).__name__ != "DeepseekV4RMSNorm":
                continue
            if module.weight.device.type != "cpu":
                raise RuntimeError(
                    f"Expected CPU RMSNorm weight, got {module.weight.device}"
                )
            module.weight.data = module.weight.data.to(torch.bfloat16)
            converted += 1
    return converted


def main() -> None:
    args = parse_args()
    transformers_logging.disable_progress_bar()
    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16}[args.dtype]
    model_dir = args.model.expanduser().resolve()
    placement_modes = int(bool(args.all_gpu)) + int(bool(args.direct_all_gpu)) + int(bool(args.cpu_layers))
    if placement_modes > 1:
        raise ValueError("--all-gpu, --direct-all-gpu, and --cpu-layers are mutually exclusive")
    final_device_map = (
        build_all_gpu_device_map(model_dir)
        if args.all_gpu or args.direct_all_gpu
        else None
    )
    load_device_map = (
        {"": "cpu"}
        if args.all_gpu
        else (
            final_device_map
            if args.direct_all_gpu
            else (
                build_cpu_layer_device_map(model_dir, cpu_layers=args.cpu_layers)
                if args.cpu_layers
                else "balanced"
            )
        )
    )

    if args.trust_hub_kernels:
        from transformers.integrations import hub_kernels

        hub_kernels.ALLOW_ALL_KERNELS = True

    started = time.perf_counter()
    print(
        "[baseline] constructing model on CPU before GPU-only dispatch"
        if args.all_gpu
        else f"[baseline] loading with device_map={load_device_map}",
        flush=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        str(model_dir),
        device_map=load_device_map,
        dtype=dtype,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    model.eval()
    restored_fp32_tensors = 0
    if args.restore_native_fp32 is not None:
        print("[baseline] restoring 406 exact native FP32 control tensors", flush=True)
        restored_fp32_tensors = restore_native_fp32(
            model,
            args.restore_native_fp32.expanduser().resolve(),
        )
    if args.all_gpu:
        assert final_device_map is not None
        print(f"[baseline] dispatching full model to {torch.cuda.device_count()} GPUs", flush=True)
        model = dispatch_model(
            model,
            device_map=final_device_map,
            main_device=final_device_map["model.embed_tokens"],
            force_hooks=True,
        )
        gc.collect()
        torch.cuda.empty_cache()
        device_map = final_device_map
        gpu_parameter_bytes = audit_gpu_residency(model)
        print(
            "[baseline] GPU-only residency verified: "
            + ", ".join(
                f"{device}={size / 2**30:.2f} GiB"
                for device, size in sorted(gpu_parameter_bytes.items())
            ),
            flush=True,
        )
    elif args.direct_all_gpu:
        assert final_device_map is not None
        device_map = final_device_map
        gpu_parameter_bytes = audit_gpu_residency(model)
        print(
            "[baseline] direct GPU-only residency verified: "
            + ", ".join(
                f"{device}={size / 2**30:.2f} GiB"
                for device, size in sorted(gpu_parameter_bytes.items())
            ),
            flush=True,
        )
    else:
        device_map = load_device_map
        gpu_parameter_bytes = {}
    cpu_rmsnorms_converted = (
        prepare_cpu_layers_for_bf16_gemm(model, args.cpu_layers)
        if args.cpu_layers
        else 0
    )
    grouped_mm_fallback_dtype_patch = enable_dtype_safe_grouped_mm_fallback()
    load_seconds = time.perf_counter() - started
    tokenizer = AutoTokenizer.from_pretrained(str(args.model), trust_remote_code=True)

    parameter_dtypes = {}
    for name in (
        "model.embed_tokens.weight",
        "model.layers.0.attn_hc.fn",
        "model.layers.0.input_layernorm.weight",
        "model.layers.0.self_attn.q_a_proj.weight",
        "model.layers.0.self_attn.q_a_proj.weight_scale_inv",
        "model.layers.2.self_attn.compressor.kv_proj.weight",
        "model.hc_head.hc_fn",
        "model.norm.weight",
        "lm_head.weight",
    ):
        value = model
        try:
            for part in name.split("."):
                value = getattr(value, part)
        except AttributeError:
            continue
        if isinstance(value, torch.Tensor):
            parameter_dtypes[name] = str(value.dtype)

    layer_summaries = {}
    diagnostic_summaries: dict[str, object] = {}
    hooks = []
    for name, module in model.named_modules():
        if name.startswith("model.layers.") and name.count(".") == 2:
            def observe_layer(_module, _inputs, output, _name=name):
                if _name in layer_summaries:
                    return
                value = output[0] if isinstance(output, tuple) else output
                if isinstance(value, torch.Tensor):
                    layer_summaries[_name] = tensor_summary(value)

            hooks.append(module.register_forward_hook(observe_layer))

    if args.diagnostic_layer is not None:
        layer_prefix = f"model.layers.{args.diagnostic_layer}"
        diagnostic_paths = {
            layer_prefix,
            f"{layer_prefix}.attn_hc",
            f"{layer_prefix}.input_layernorm",
            f"{layer_prefix}.self_attn",
            f"{layer_prefix}.ffn_hc",
            f"{layer_prefix}.post_attention_layernorm",
            f"{layer_prefix}.mlp",
            f"{layer_prefix}.mlp.gate",
            f"{layer_prefix}.mlp.experts",
            f"{layer_prefix}.mlp.shared_experts",
        }

        def summarize_output(value):
            if isinstance(value, torch.Tensor):
                return tensor_summary(value)
            if isinstance(value, (tuple, list)):
                return [summarize_output(item) for item in value]
            return {"type": type(value).__name__}

        for name, module in model.named_modules():
            if name not in diagnostic_paths:
                continue

            def observe_input(_module, inputs, _name=name):
                key = _name + ".input"
                if key not in diagnostic_summaries:
                    diagnostic_summaries[key] = summarize_output(inputs)

            def observe_output(_module, _inputs, output, _name=name):
                key = _name + ".output"
                if key not in diagnostic_summaries:
                    diagnostic_summaries[key] = summarize_output(output)
                if (
                    _name == f"{layer_prefix}.mlp.gate"
                    and isinstance(output, tuple)
                    and len(output) >= 3
                ):
                    weights, indices = output[1], output[2]
                    diagnostic_summaries[_name + ".selected_experts"] = sorted(
                        set(indices.detach().cpu().reshape(-1).tolist())
                    )
                    diagnostic_summaries[_name + ".selected_indices"] = (
                        indices.detach().cpu().tolist()
                    )
                    diagnostic_summaries[_name + ".selected_weights"] = (
                        weights.detach().float().cpu().tolist()
                    )

            hooks.append(module.register_forward_pre_hook(observe_input))
            hooks.append(module.register_forward_hook(observe_output))

    inputs = build_inputs(tokenizer, args.prompt)
    input_device = model.get_input_embeddings().weight.device
    inputs = {key: value.to(input_device) for key, value in inputs.items()}
    prompt_tokens = int(inputs["input_ids"].shape[-1])

    generation_started = time.perf_counter()
    with torch.inference_mode():
        with torch.autocast(device_type="cuda", dtype=dtype):
            generated = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                return_dict_in_generate=True,
                output_scores=True,
            )
    generation_seconds = time.perf_counter() - generation_started
    for hook in hooks:
        hook.remove()

    continuation = generated.sequences[0, prompt_tokens:]
    continuation_ids = continuation.detach().cpu().tolist()
    text = tokenizer.decode(
        continuation,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    score_summaries = []
    for step, scores in enumerate(generated.scores):
        if step not in {0, 1, 2, len(generated.scores) - 1}:
            continue
        row = scores[0].detach().float()
        finite = torch.isfinite(row)
        top_values, top_ids = torch.topk(row, k=min(5, row.numel()))
        score_summaries.append(
            {
                "step": step,
                "finite_fraction": float(finite.float().mean()),
                "min": float(row[finite].min()) if finite.any() else None,
                "max": float(row[finite].max()) if finite.any() else None,
                "std": float(row[finite].std()) if finite.any() else None,
                "top_ids": top_ids.cpu().tolist(),
                "top_values": top_values.cpu().tolist(),
            }
        )

    report = {
        "model": str(args.model.resolve()),
        "requested_activation_dtype": args.dtype,
        "device_map": {key: str(value) for key, value in device_map.items()}
        if isinstance(device_map, dict)
        else device_map,
        "restored_native_fp32_tensors": restored_fp32_tensors,
        "cpu_rmsnorms_converted_to_bfloat16": cpu_rmsnorms_converted,
        "grouped_mm_fallback_dtype_patch": grouped_mm_fallback_dtype_patch,
        "trusted_hub_kernels": args.trust_hub_kernels,
        "gpu_parameter_bytes": gpu_parameter_bytes,
        "parameter_dtypes": parameter_dtypes,
        "prefill_layer_summaries": layer_summaries,
        "diagnostic_layer": args.diagnostic_layer,
        "diagnostic_summaries": diagnostic_summaries,
        "load_seconds": load_seconds,
        "generation_seconds": generation_seconds,
        "prompt": args.prompt,
        "prompt_tokens": prompt_tokens,
        "requested_max_new_tokens": args.max_new_tokens,
        "generated_tokens": len(continuation_ids),
        "continuation_ids": continuation_ids,
        "unique_continuation_ids": len(set(continuation_ids)),
        "continuation_text": text,
        "score_summaries": score_summaries,
    }
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    print(rendered, flush=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
