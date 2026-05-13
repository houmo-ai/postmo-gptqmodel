"""Comprehensive test & evaluation for quantized Qwen3.5 models.

Tests: model loading, text inference, PPL, chat quality, vision inference.
Supports both transformers and GPTQModel backends.

Usage:
    python scripts/test_eval.py --model_path ./output/mode1-llm-only --device cuda:0
    python scripts/test_eval.py --model_path ./output/mode2-rotate-merger --device cuda:1
"""

import argparse
import json
import os
import sys
import time
import traceback

import torch

# ─── GLIBC compatibility patch (must be before transformers import) ───────────
try:
    from auto_round.inference.backend import BackendInfos
    to_remove = [k for k in list(BackendInfos.keys()) if "marlin" in k or "exllama" in k]
    for k in to_remove:
        del BackendInfos[k]
except Exception:
    pass


def parse_args():
    parser = argparse.ArgumentParser(description="Test quantized Qwen3.5 model")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--original_model", type=str, default="/data01/datasets/Qwen3.5-27B",
                        help="Original model path for processor/tokenizer")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--output_report", type=str, default=None,
                        help="Path to save JSON report (default: <model_path>/eval_report.json)")
    parser.add_argument("--skip_ppl", action="store_true")
    parser.add_argument("--skip_vision", action="store_true")
    return parser.parse_args()


# ─── Model Loading ────────────────────────────────────────────────────────────

def load_model_transformers(model_path, device):
    """Load quantized model with transformers."""
    from transformers import AutoModelForImageTextToText, AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    # Try VLM first (ImageTextToText), then CausalLM
    try:
        model = AutoModelForImageTextToText.from_pretrained(
            model_path, torch_dtype=torch.bfloat16,
            device_map=device, trust_remote_code=True)
        model_type = "VLM"
    except Exception:
        model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.bfloat16,
            device_map=device, trust_remote_code=True)
        model_type = "CausalLM"

    return model, tokenizer, model_type


def _is_vlm_model(model_path):
    """Check if model is a VLM (not text-only) by reading config.json."""
    config_path = os.path.join(model_path, "config.json")
    if not os.path.exists(config_path):
        return False
    with open(config_path) as f:
        config = json.load(f)
    archs = config.get("architectures", [])
    # Qwen3_5ForConditionalGeneration = VLM, Qwen3_5ForCausalLM = text-only
    return any("ConditionalGeneration" in a for a in archs)


def _auto_convert_vlm_to_text(model_path, base_model_dir=None):
    """Auto-convert VLM model to text-only for GPTQModel compatibility."""
    text_path = model_path.rstrip("/") + "-text-auto"
    if os.path.exists(text_path) and os.path.exists(os.path.join(text_path, "config.json")):
        print(f"  Using existing text conversion: {text_path}")
        return text_path
    print(f"  Auto-converting VLM → text-only: {text_path}")
    from convert_vlm_to_text import convert_model
    convert_model(model_path, text_path, keep_visual=False, base_model_dir=base_model_dir)
    return text_path


def load_model_gptqmodel(model_path, device, base_model_dir=None):
    """Load quantized model with GPTQModel.

    If model is VLM format, auto-converts to text-only first (GPTQModel
    doesn't support VLM architectures like Qwen3_5ForConditionalGeneration).
    """
    from gptqmodel import GPTQModel

    actual_path = model_path
    # if _is_vlm_model(model_path):
    #     print(f"  ⚠️ VLM model detected. GPTQModel requires text-only format.")
    #     actual_path = _auto_convert_vlm_to_text(model_path, base_model_dir)

    model = GPTQModel.load(actual_path, device=device)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(actual_path, trust_remote_code=True)
    return model, tokenizer, "GPTQModel"


def load_model(model_path, device, backend="gptqmodel", base_model_dir=None):
    """Load model with specified backend."""
    t0 = time.time()
    if backend == "gptqmodel":
        model, tokenizer, model_type = load_model_gptqmodel(
            model_path, device, base_model_dir)
    else:
        model, tokenizer, model_type = load_model_transformers(model_path, device)
    load_time = time.time() - t0
    return model, tokenizer, model_type, load_time


# ─── Quantization Stats ──────────────────────────────────────────────────────

def count_quant_layers(model):
    """Count quantized layers and total parameters."""
    quant_layers = 0
    total_params = 0
    quant_params = 0
    for name, module in model.named_modules():
        if hasattr(module, 'qweight'):
            quant_layers += 1
            quant_params += module.qweight.numel()
        for p in module.parameters(recurse=False):
            total_params += p.numel()
    return quant_layers, total_params, quant_params


# ─── Text Inference ───────────────────────────────────────────────────────────

def test_text_inference(model, tokenizer, device):
    """Test basic text generation."""
    prompt = "What is the capital of France?"
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    inputs = tokenizer(text, return_tensors="pt").to(device)

    t0 = time.time()
    with torch.no_grad():
        outputs = model.generate(
            **inputs, max_new_tokens=100,
            do_sample=False, temperature=1.0)
    gen_time = time.time() - t0

    response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
    n_tokens = outputs.shape[-1] - inputs["input_ids"].shape[-1]
    tokens_per_sec = n_tokens / gen_time if gen_time > 0 else 0

    return {
        "prompt": prompt,
        "response": response[:200],
        "n_tokens": n_tokens,
        "time_s": round(gen_time, 2),
        "tokens_per_sec": round(tokens_per_sec, 1),
    }


# ─── Chat Quality ────────────────────────────────────────────────────────────

CHAT_PROMPTS = [
    "请用一句话解释量子计算的核心原理。",
    "Write a Python function to calculate Fibonacci numbers recursively.",
    "What are the key differences between TCP and UDP?",
    "用中文写一首关于人工智能的七言绝句。",
    "Explain the concept of attention mechanism in transformers in simple terms.",
]

def test_chat_quality(model, tokenizer, device):
    """Test chat quality with multiple prompts."""
    results = []
    for prompt in CHAT_PROMPTS:
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        inputs = tokenizer(text, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs, max_new_tokens=200,
                do_sample=False, temperature=1.0)

        response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
        results.append({
            "prompt": prompt[:50],
            "response": response[:300],
            "n_tokens": outputs.shape[-1] - inputs["input_ids"].shape[-1],
        })
    return results


# ─── PPL Evaluation ───────────────────────────────────────────────────────────

def test_ppl(model, tokenizer, device, dataset_name="wikitext", max_samples=None):
    """Evaluate perplexity on WikiText-2."""
    from datasets import load_dataset

    print("  Loading wikitext-2 dataset...")
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join([t for t in dataset["text"] if t.strip()])

    # Tokenize
    encodings = tokenizer(text, return_tensors="pt")
    input_ids = encodings.input_ids.to(device)

    seq_len = 2048
    stride = 1024
    nlls = []
    n_tokens = 0

    total_len = input_ids.shape[1]
    if max_samples:
        total_len = min(total_len, max_samples * seq_len)

    print(f"  Evaluating PPL (seqlen={seq_len}, stride={stride}, total_tokens={total_len})...")
    for begin_loc in range(0, total_len - seq_len, stride):
        end_loc = begin_loc + seq_len
        input_chunk = input_ids[:, begin_loc:end_loc]
        target_chunk = input_chunk.clone()
        target_chunk[:, :-stride] = -100

        with torch.no_grad():
            outputs = model(input_ids=input_chunk, labels=target_chunk)
            neg_log_likelihood = outputs.loss

        nlls.append(neg_log_likelihood.item())
        n_tokens += stride

        if len(nlls) % 20 == 0:
            current_ppl = torch.exp(torch.tensor(nlls).mean()).item()
            print(f"    {len(nlls)} chunks, current PPL: {current_ppl:.4f}", end="\r")

    ppl = torch.exp(torch.tensor(nlls).mean()).item()
    print(f"  PPL = {ppl:.4f} ({len(nlls)} chunks, {n_tokens} tokens)")
    return {"ppl": round(ppl, 4), "n_chunks": len(nlls), "n_tokens": n_tokens}


# ─── Vision Inference ─────────────────────────────────────────────────────────

def test_vision_inference(model, tokenizer, original_model_path, device):
    """Test vision inference with a synthetic image."""
    from PIL import Image
    import numpy as np

    # Create a test image (colored rectangles)
    img = Image.new('RGB', (448, 448), color=(255, 255, 255))
    pixels = np.array(img)
    pixels[50:200, 50:200] = [255, 0, 0]   # Red square
    pixels[50:200, 250:400] = [0, 0, 255]  # Blue square
    pixels[250:400, 150:300] = [0, 255, 0]  # Green square
    img = Image.fromarray(pixels)

    # Try to load processor from original model or quantized model
    try:
        from transformers import AutoProcessor
        processor = AutoProcessor.from_pretrained(original_model_path, trust_remote_code=True)
    except Exception:
        return {"error": "Cannot load processor", "success": False}

    # Check if model supports vision — walk the hierarchy to find 'visual'.
    # HF VLM (Qwen3.5):  model.model.visual
    # GPTQModel wrapper:  wrapper.model(=HF).model.visual
    def _find_visual_module(m, depth=4):
        if hasattr(m, 'visual'):
            return m.visual
        if depth > 0 and hasattr(m, 'model'):
            return _find_visual_module(m.model, depth - 1)
        return None

    visual_module = _find_visual_module(model)
    if visual_module is None:
        return {"error": "Model has no visual module", "success": False}

    prompt = "Describe this image. What colored shapes do you see?"
    messages = [
        {"role": "user", "content": [
            {"type": "image", "image": img},
            {"type": "text", "text": prompt},
        ]}
    ]

    try:
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=text, images=[img], return_tensors="pt", padding=True)

        # Move text inputs to main device; vision inputs to visual module device
        visual_device = next(visual_module.parameters()).device
        vision_keys = {"pixel_values", "image_grid_thw", "pixel_values_videos", "video_grid_thw"}
        inputs = {
            k: v.to(visual_device if k in vision_keys else device) if isinstance(v, torch.Tensor) else v
            for k, v in inputs.items()
        }

        t0 = time.time()
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=200, do_sample=False)
        gen_time = time.time() - t0

        response = processor.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
        return {
            "success": True,
            "prompt": prompt,
            "response": response[:300],
            "time_s": round(gen_time, 2),
            "n_tokens": outputs.shape[-1] - inputs["input_ids"].shape[-1],
        }
    except Exception as e:
        return {"error": str(e)[:200], "success": False, "traceback": traceback.format_exc()[-500:]}


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    report = {"model_path": args.model_path, "device": args.device, "tests": {}}

    print(f"\n{'='*70}")
    print(f"Evaluating: {args.model_path}")
    print(f"Device: {args.device}")
    print(f"{'='*70}")

    # ── 1. Load model ──
    print("\n[1/5] Loading model...")
    try:
        # Try GPTQModel first
        model, tokenizer, model_type, load_time = load_model(
            args.model_path, args.device, backend="gptqmodel",
            base_model_dir=args.original_model)
        report["backend"] = "gptqmodel"
    except Exception as e:
        print(f"  GPTQModel failed: {e}")
        print("  Falling back to transformers...")
        model, tokenizer, model_type, load_time = load_model(
            args.model_path, args.device, backend="transformers")
        report["backend"] = "transformers"

    print(f"  ✅ Loaded as {model_type} in {load_time:.1f}s (backend: {report['backend']})")
    report["model_type"] = model_type
    report["load_time_s"] = round(load_time, 1)

    # ── 2. Quant stats ──
    print("\n[2/5] Counting quantized layers...")
    quant_layers, total_params, quant_params = count_quant_layers(model)
    print(f"  Quantized layers: {quant_layers}")
    report["quant_layers"] = quant_layers

    # ── 3. Text inference ──
    print("\n[3/5] Text inference test...")
    text_result = test_text_inference(model, tokenizer, args.device)
    print(f"  ✅ {text_result['n_tokens']} tokens in {text_result['time_s']}s "
          f"({text_result['tokens_per_sec']} tok/s)")
    print(f"  Response: {text_result['response'][:100]}...")
    report["tests"]["text_inference"] = text_result

    # ── 4. Chat quality ──
    print("\n[4/5] Chat quality test (5 prompts)...")
    chat_results = test_chat_quality(model, tokenizer, args.device)
    for i, r in enumerate(chat_results):
        print(f"  [{i+1}] {r['prompt'][:40]}... → {r['n_tokens']} tokens")
    report["tests"]["chat_quality"] = chat_results

    # ── 5. PPL ──
    if not args.skip_ppl:
        print("\n[5a/5] PPL evaluation (wikitext-2)...")
        try:
            ppl_result = test_ppl(model, tokenizer, args.device)
            print(f"  ✅ PPL = {ppl_result['ppl']}")
            report["tests"]["ppl"] = ppl_result
        except Exception as e:
            print(f"  ❌ PPL failed: {e}")
            report["tests"]["ppl"] = {"error": str(e)[:200]}
    else:
        print("\n[5a/5] PPL skipped")

    # ── 6. Vision ──
    if not args.skip_vision:
        print("\n[5b/5] Vision inference test...")
        try:
            vis_result = test_vision_inference(model, tokenizer, args.original_model, args.device)
            if vis_result.get("success"):
                print(f"  ✅ Vision response: {vis_result['response'][:100]}...")
            else:
                print(f"  ⚠️ Vision: {vis_result.get('error', 'unknown error')}")
            report["tests"]["vision"] = vis_result
        except Exception as e:
            print(f"  ❌ Vision failed: {e}")
            report["tests"]["vision"] = {"error": str(e)[:200], "success": False}
    else:
        print("\n[5b/5] Vision skipped")

    # ── Save report ──
    report_path = args.output_report or os.path.join(args.model_path, "eval_report.json")
    os.makedirs(os.path.dirname(report_path) if os.path.dirname(report_path) else ".", exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\n📄 Report saved: {report_path}")

    # ── Summary ──
    print(f"\n{'='*70}")
    print(f"SUMMARY: {args.model_path}")
    print(f"  Backend: {report['backend']}")
    print(f"  Model type: {report['model_type']}")
    print(f"  Load time: {report['load_time_s']}s")
    print(f"  Quant layers: {report['quant_layers']}")
    print(f"  Text: {text_result['tokens_per_sec']} tok/s")
    if 'ppl' in report['tests'] and 'ppl' in report['tests']['ppl']:
        print(f"  PPL: {report['tests']['ppl']['ppl']}")
    vis = report['tests'].get('vision', {})
    print(f"  Vision: {'✅' if vis.get('success') else '❌'} {vis.get('error', 'OK')[:50]}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
