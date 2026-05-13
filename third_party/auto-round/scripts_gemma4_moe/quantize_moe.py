"""Gemma4 MoE AutoRound quantization script.

This script targets Gemma4 VLM checkpoints whose text backbone enables MoE
blocks, for example gemma-4-26B-A4B-it. It keeps the export format compatible
with GPTQModel by saving with --format auto_gptq.

Example:
    python scripts_gemma4_moe/quantize_moe.py \
        --model /data01/home/chenzx/model/gemma-4-26B-A4B-it \
        --output_dir output/gemma4-moe-mode1 \
        --device cuda:0 \
        --llm_bits 4 --llm_group_size 64 --sym \
        --iters 200 --nsamples 128 --seqlen 2048 --batch_size 8 \
        --dataset NeelNanda/pile-10k --seed 42 --format auto_gptq
"""

import argparse
from contextlib import contextmanager
import gc
import glob
import json
import os
import sys
import time

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import torch


def parse_args():
    parser = argparse.ArgumentParser(
        description="Gemma4 MoE AutoRound quantization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--model", type=str, required=True,
                        help="HuggingFace model path (Gemma4 MoE VLM)")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--dtype", type=str, default="bfloat16",
                        choices=["bfloat16", "float16", "float32"],
                        help="Model loading dtype")

    parser.add_argument("--llm_bits", type=int, default=4)
    parser.add_argument("--llm_group_size", type=int, default=64)
    parser.add_argument("--sym", action="store_true", default=False,
                        help="Use symmetric quantization")
    parser.add_argument("--iters", type=int, default=200,
                        help="SignSGD iterations (0=RTN)")
    parser.add_argument("--nsamples", type=int, default=128)
    parser.add_argument("--seqlen", type=int, default=2048)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--dataset", type=str, default="NeelNanda/pile-10k")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--format", type=str, default="auto_gptq",
                        help="Export format: auto_gptq (GPTQModel compat), auto_round, etc.")
    parser.add_argument("--quantize_router", action="store_true", default=False,
                        help="Quantize router.proj. Disabled by default for routing stability.")
    parser.add_argument("--ignore_layers", type=str, default="",
                        help="Extra comma-separated layer names kept in original precision")
    parser.add_argument("--verify_only", action="store_true", default=False,
                        help="Skip quantization and only load the saved quantized model for generation")
    parser.add_argument("--prompt", type=str, default="你是谁",
                        help="Prompt used by the post-save generation check")
    parser.add_argument("--max_new_tokens", type=int, default=128,
                        help="Maximum new tokens for the post-save generation check")

    return parser.parse_args()


def _get_torch_dtype(dtype_name):
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[dtype_name]


def _build_ignore_layers(args):
    ignore_layers = []
    if not args.quantize_router:
        ignore_layers.append("router.proj")
    if args.ignore_layers.strip():
        ignore_layers.extend([item for item in args.ignore_layers.split(",") if item.strip()])
    return ",".join(dict.fromkeys(ignore_layers))


def patch_position_embeddings(model, seqlen):
    """Patch Gemma4 decoder layers to use correct position embeddings per layer type."""
    import types

    lm = model.model.language_model
    text_config = getattr(model.config, "text_config", model.config)

    position_ids = torch.arange(seqlen).unsqueeze(0)
    dummy_hidden = torch.zeros(1, seqlen, text_config.hidden_size, dtype=torch.bfloat16)

    pos_emb_cache = {}
    for layer_type in set(text_config.layer_types):
        cos, sin = lm.rotary_emb(dummy_hidden, position_ids, layer_type)
        pos_emb_cache[layer_type] = (cos, sin)
        print(
            f"  Prepared position embeddings for '{layer_type}': "
            f"cos={tuple(cos.shape)}, sin={tuple(sin.shape)}"
        )

    original_forward = type(lm.layers[0]).forward

    def make_patched_forward(layer_type_name, orig_fwd):
        cached_pos = pos_emb_cache[layer_type_name]

        def patched_forward(self, hidden_states, per_layer_input=None,
                            position_embeddings=None, attention_mask=None,
                            position_ids=None, past_key_values=None, **kwargs):
            correct_pos = (
                cached_pos[0].to(hidden_states.device),
                cached_pos[1].to(hidden_states.device),
            )
            return orig_fwd(
                self,
                hidden_states,
                per_layer_input=per_layer_input,
                position_embeddings=correct_pos,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                **kwargs,
            )

        return patched_forward

    for index, layer in enumerate(lm.layers):
        layer_type = text_config.layer_types[index]
        layer.forward = types.MethodType(make_patched_forward(layer_type, original_forward), layer)

    print(f"  Patched {len(lm.layers)} decoder layers for mixed RoPE inputs")


def fix_checkpoint_after_save(output_dir):
    """Prefer model shards over extra tensors in the weight map when both exist."""
    try:
        from safetensors import safe_open
        from safetensors.torch import save_file
    except ImportError:
        print("[fix_checkpoint] safetensors not available, skipping")
        return

    idx_path = os.path.join(output_dir, "model.safetensors.index.json")
    extra_path = os.path.join(output_dir, "model_extra_tensors.safetensors")
    if not os.path.exists(idx_path) or not os.path.exists(extra_path):
        return

    shard_keys = {}
    for shard_path in sorted(glob.glob(os.path.join(output_dir, "model-*.safetensors"))):
        shard_name = os.path.basename(shard_path)
        with safe_open(shard_path, framework="pt", device="cpu") as tensors:
            for key in tensors.keys():
                shard_keys[key] = shard_name

    if not shard_keys:
        return

    with open(idx_path) as handle:
        index_data = json.load(handle)

    conflicts = 0
    for key in list(index_data["weight_map"].keys()):
        if key in shard_keys and index_data["weight_map"][key] == "model_extra_tensors.safetensors":
            index_data["weight_map"][key] = shard_keys[key]
            conflicts += 1

    for key, shard_name in shard_keys.items():
        if key not in index_data["weight_map"]:
            index_data["weight_map"][key] = shard_name
            conflicts += 1

    if conflicts == 0:
        print("[fix_checkpoint] No index conflicts found")
        return

    with open(idx_path, "w") as handle:
        json.dump(index_data, handle, indent=2)
    print(f"[fix_checkpoint] Fixed {conflicts} index entries")

    new_tensors = {}
    removed = 0
    with safe_open(extra_path, framework="pt", device="cpu") as tensors:
        for key in tensors.keys():
            if key in shard_keys:
                removed += 1
            else:
                new_tensors[key] = tensors.get_tensor(key)

    if removed > 0:
        save_file(new_tensors, extra_path + ".tmp")
        os.replace(extra_path + ".tmp", extra_path)
        print(f"[fix_checkpoint] Removed {removed} duplicates from model_extra_tensors")


def _patch_incompatible_backends():
    """Remove backends that commonly fail on older GLIBC environments."""
    try:
        from auto_round.inference.backend import BackendInfos

        to_remove = [name for name in list(BackendInfos.keys()) if "marlin" in name or "exllama" in name]
        for name in to_remove:
            del BackendInfos[name]
        if to_remove:
            print(f"[verify] Removed {len(to_remove)} incompatible backends")
    except Exception as exc:
        print(f"[verify] Backend patch skipped: {exc}")


def _normalize_device_map(device):
    if device == "cpu":
        return "cpu"
    return {"": device}


def _checkpoint_uses_split_moe_experts(output_dir):
    config_path = os.path.join(output_dir, "config.json")
    if not os.path.exists(config_path):
        return False

    try:
        with open(config_path) as handle:
            config = json.load(handle)
    except Exception as exc:
        print(f"[verify] Failed to inspect checkpoint config: {exc}")
        return False

    if config.get("model_type") != "gemma4":
        return False

    text_config = config.get("text_config", {})
    if not text_config.get("enable_moe_block", False):
        return False

    modules = config.get("quantization_config", {}).get("modules_in_block_to_quantize", [])
    flat_modules = []
    for module_group in modules:
        if isinstance(module_group, list):
            flat_modules.extend(module_group)
        elif isinstance(module_group, str):
            flat_modules.append(module_group)

    return any(name.startswith("experts.0.") for name in flat_modules)


@contextmanager
def _patched_gemma4_moe_gptq_loader(output_dir):
    if not _checkpoint_uses_split_moe_experts(output_dir):
        yield
        return

    from auto_round.modeling.fused_moe.moe_experts_interface import prepare_model_for_moe_quantization
    from transformers.quantizers.quantizer_gptq import GptqHfQuantizer

    original_process = GptqHfQuantizer._process_model_before_weight_loading

    def patched_process(self, model, **kwargs):
        text_config = getattr(model.config, "text_config", model.config)
        experts = None
        try:
            experts = model.model.language_model.layers[0].experts
        except Exception:
            experts = None

        already_split = experts is not None and hasattr(experts, "0")
        if (
            getattr(self, "pre_quantized", False)
            and getattr(model.config, "model_type", None) == "gemma4"
            and getattr(text_config, "enable_moe_block", False)
            and not already_split
        ):
            prepare_model_for_moe_quantization(model)
            print("[verify] Prepared Gemma4 MoE split experts before GPTQ weight loading")

        return original_process(self, model, **kwargs)

    print("[verify] Detected Gemma4 MoE split-experts checkpoint")
    GptqHfQuantizer._process_model_before_weight_loading = patched_process
    try:
        yield
    finally:
        GptqHfQuantizer._process_model_before_weight_loading = original_process


def _load_tokenizer_for_verify(output_dir, original_model):
    from transformers import AutoTokenizer

    last_error = None
    for path in [output_dir, original_model]:
        try:
            return AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        except Exception as exc:
            last_error = exc
    raise RuntimeError(f"Failed to load tokenizer for verification: {last_error}")


def _build_verify_inputs(tokenizer, prompt, device):
    messages = [{"role": "user", "content": prompt}]
    try:
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    except Exception:
        text = prompt

    inputs = tokenizer(text, return_tensors="pt")
    return {key: value.to(device) for key, value in inputs.items()}


def _load_quantized_model_for_verify(output_dir, device, dtype_name):
    _patch_incompatible_backends()

    from transformers import AutoModelForCausalLM, AutoModelForImageTextToText

    model_kwargs = dict(
        torch_dtype=_get_torch_dtype(dtype_name),
        device_map=_normalize_device_map(device),
        trust_remote_code=True,
    )

    last_error = None
    with _patched_gemma4_moe_gptq_loader(output_dir):
        for loader in (AutoModelForImageTextToText, AutoModelForCausalLM):
            try:
                with torch.no_grad():
                    model = loader.from_pretrained(output_dir, **model_kwargs)
                if not hasattr(model, "generate"):
                    raise TypeError(f"{type(model).__name__} does not implement generate()")
                model.eval()
                return model, loader.__name__
            except Exception as exc:
                last_error = exc

    raise RuntimeError(f"Failed to load quantized model for verification: {last_error}")


def verify_saved_model_generation(output_dir, original_model, device, dtype_name, prompt, max_new_tokens):
    """Reload the saved quantized model and run one generation check."""
    print("\nRunning post-save generation check...")
    tokenizer = _load_tokenizer_for_verify(output_dir, original_model)
    model, loader_name = _load_quantized_model_for_verify(output_dir, device, dtype_name)
    print(f"[verify] Loaded with {loader_name}")

    inputs = _build_verify_inputs(tokenizer, prompt, device)
    input_length = inputs["input_ids"].shape[-1]

    verify_start = time.time()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    verify_time = time.time() - verify_start

    response = tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True).strip()
    if not response:
        raise RuntimeError("Post-save generation returned an empty response")

    print(f"[verify] Prompt: {prompt}")
    print(f"[verify] Response: {response}")
    print(f"[verify] Time: {verify_time:.1f}s")

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "prompt": prompt,
        "response": response,
        "max_new_tokens": max_new_tokens,
        "time_s": verify_time,
    }


def _assert_gemma4_moe(model):
    text_config = getattr(model.config, "text_config", model.config)
    if not getattr(text_config, "enable_moe_block", False):
        raise ValueError("The loaded model does not enable Gemma4 MoE blocks")
    first_layer = model.model.language_model.layers[0]
    if not hasattr(first_layer, "experts") or not hasattr(first_layer, "router"):
        raise ValueError("Gemma4 MoE modules were not found on decoder layer 0")


def main():
    args = parse_args()
    total_start = time.time()
    ignore_layers = _build_ignore_layers(args)

    if args.verify_only:
        verify_result = verify_saved_model_generation(
            output_dir=args.output_dir,
            original_model=args.model,
            device=args.device,
            dtype_name=args.dtype,
            prompt=args.prompt,
            max_new_tokens=args.max_new_tokens,
        )
        total_time = time.time() - total_start
        print(f"\n{'=' * 60}")
        print("Verify Only Done")
        print(f"  Output: {args.output_dir}")
        print(f"  Verify prompt: {verify_result['prompt']}")
        print(f"  Verify response: {verify_result['response']}")
        print(f"  Total time: {total_time:.1f}s ({total_time / 3600:.2f}h)")
        print(f"{'=' * 60}")
        return

    print(f"\n{'=' * 60}")
    print("Gemma4 MoE AutoRound Quantization")
    print(f"Model: {args.model}")
    print(f"LLM: W{args.llm_bits}G{args.llm_group_size}")
    print(f"Sym: {args.sym}")
    print(f"Format: {args.format}")
    print(f"Device: {args.device}")
    if ignore_layers:
        print(f"Ignore layers: {ignore_layers}")
    print(f"{'=' * 60}")

    from transformers import AutoProcessor, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    processor = None
    try:
        processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    except Exception:
        pass

    print("\nLoading model...")
    model_kwargs = dict(
        torch_dtype=_get_torch_dtype(args.dtype),
        device_map="cpu",
        trust_remote_code=True,
    )

    try:
        from transformers import AutoModelForImageTextToText

        model = AutoModelForImageTextToText.from_pretrained(args.model, **model_kwargs)
    except Exception:
        from transformers import AutoModel

        model = AutoModel.from_pretrained(args.model, **model_kwargs)

    _assert_gemma4_moe(model)

    text_config = getattr(model.config, "text_config", model.config)
    print(f"Loaded: {type(model).__name__}")
    print(f"  Layers: {text_config.num_hidden_layers}")
    print(f"  Hidden size: {text_config.hidden_size}")
    print(f"  Experts: {text_config.num_experts}")
    print(f"  Active experts/token: {text_config.top_k_experts}")
    print(f"  Dense MLP intermediate: {text_config.intermediate_size}")
    print(f"  MoE intermediate: {text_config.moe_intermediate_size}")

    print("\nPatching position embeddings for block-wise calibration...")
    patch_position_embeddings(model, args.seqlen)

    print("\nStarting AutoRound quantization...")
    from auto_round import AutoRound

    ar = AutoRound(
        model=model,
        tokenizer=tokenizer,
        processor=processor,
        bits=args.llm_bits,
        group_size=args.llm_group_size,
        sym=args.sym,
        iters=args.iters,
        seqlen=args.seqlen,
        nsamples=args.nsamples,
        batch_size=args.batch_size,
        dataset=args.dataset,
        seed=args.seed,
        quant_nontext_module=False,
        ignore_layers=ignore_layers,
    )

    quant_start = time.time()
    ar.quantize()
    quant_time = time.time() - quant_start
    print(f"Quantization completed in {quant_time:.1f}s ({quant_time / 3600:.2f}h)")

    print(f"\nSaving to: {args.output_dir}")
    os.makedirs(args.output_dir, exist_ok=True)
    ar.save_quantized(args.output_dir, format=args.format)
    fix_checkpoint_after_save(args.output_dir)

    del ar
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    verify_result = verify_saved_model_generation(
        output_dir=args.output_dir,
        original_model=args.model,
        device=args.device,
        dtype_name=args.dtype,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
    )

    total_time = time.time() - total_start
    print(f"\n{'=' * 60}")
    print("Done")
    print(f"  Output: {args.output_dir}")
    print(f"  Format: {args.format}")
    print(f"  Quant time: {quant_time:.1f}s ({quant_time / 3600:.2f}h)")
    print(f"  Verify prompt: {verify_result['prompt']}")
    print(f"  Verify response: {verify_result['response']}")
    print(f"  Total time: {total_time:.1f}s ({total_time / 3600:.2f}h)")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()