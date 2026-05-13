"""Full evaluation pipeline for all Qwen3.5 quantized models.

Evaluates: PPL (WikiText-2), text inference, chat quality, vision inference.
Handles mixed-attention quantized models (linear_attn + self_attn) correctly.

Usage:
    CUDA_VISIBLE_DEVICES=5,7 python scripts/run_full_eval.py
"""

import argparse
import gc
import json
import os
import time
import traceback

import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# ─── Prevent GLIBC/backend issues ──────────────────────────────────────
try:
    from auto_round.inference.backend import BackendInfos
    to_remove = [k for k in list(BackendInfos.keys()) if "marlin" in k or "exllama" in k]
    for k in to_remove:
        del BackendInfos[k]
except Exception:
    pass


# ═══════════════════════════════════════════════════════════════════════
# Model definitions
# ═══════════════════════════════════════════════════════════════════════

MODELS = [
    ("fp16-original",       "weights/Qwen3.5-27B"),
    ("gptq-baseline",       "weights/Qwen3.5-27B-GPTQModel-self-generated-4bit-hessian-mse"),
    ("mode1-llm-only",      "./output/mode1-llm-only"),
    ("mode2-rotate-merger", "./output/mode2-rotate-merger"),
    ("mode3-rotate-full",   "./output/mode3-rotate-full"),
    ("mode4-rotate-quant-all", "./output/mode4-rotate-quant-all"),
]

CHAT_PROMPTS = [
    "请用一句话解释量子计算的核心原理。",
    "Write a Python function to calculate Fibonacci numbers recursively.",
    "What are the key differences between TCP and UDP?",
    "用中文写一首关于人工智能的七言绝句。",
    "Explain the concept of attention mechanism in transformers in simple terms.",
]


# ═══════════════════════════════════════════════════════════════════════
# Model loading
# ═══════════════════════════════════════════════════════════════════════

def load_model(model_path, device="cuda:0"):
    """Load model and processor. Returns (model, tokenizer, processor, load_time, is_quantized, backend)."""
    from transformers import AutoTokenizer, AutoProcessor

    config_path = os.path.join(model_path, "config.json")
    is_quantized = False
    is_vlm = True
    if os.path.exists(config_path):
        with open(config_path) as f:
            cfg = json.load(f)
        is_quantized = "quantization_config" in cfg
        archs = cfg.get("architectures", [])
        is_vlm = any("Conditional" in a or "ImageText" in a for a in archs)

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    # Load processor for vision
    processor = None
    for path in [model_path, "weights/Qwen3.5-27B"]:
        try:
            processor = AutoProcessor.from_pretrained(path, trust_remote_code=True)
            break
        except Exception:
            continue

    t0 = time.time()
    backend = "transformers"

    if is_vlm:
        from transformers import AutoModelForImageTextToText
        model = AutoModelForImageTextToText.from_pretrained(
            model_path, torch_dtype=torch.bfloat16,
            device_map=device, trust_remote_code=True)
    else:
        # Text-only model (e.g., GPTQ baseline saved as text-only)
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.bfloat16,
            device_map=device, trust_remote_code=True)

    load_time = time.time() - t0
    model.eval()

    # Resolve actual device for inputs (needed when device_map="auto")
    actual_device = device
    if device == "auto":
        try:
            actual_device = next(model.parameters()).device
        except StopIteration:
            actual_device = "cuda:0"

    return model, tokenizer, processor, load_time, is_quantized, backend, is_vlm, actual_device


# ═══════════════════════════════════════════════════════════════════════
# Text inference
# ═══════════════════════════════════════════════════════════════════════

def test_text_inference(model, tokenizer, device):
    prompt = "What is the capital of France?"
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(device)

    t0 = time.time()
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=100, do_sample=False)
    gen_time = time.time() - t0

    response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
    n_tokens = outputs.shape[-1] - inputs["input_ids"].shape[-1]
    return {
        "prompt": prompt,
        "response": response[:300],
        "n_tokens": n_tokens,
        "time_s": round(gen_time, 2),
        "tokens_per_sec": round(n_tokens / gen_time, 1) if gen_time > 0 else 0,
    }


# ═══════════════════════════════════════════════════════════════════════
# Chat quality
# ═══════════════════════════════════════════════════════════════════════

def test_chat_quality(model, tokenizer, device):
    results = []
    for prompt in CHAT_PROMPTS:
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to(device)

        t0 = time.time()
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=200, do_sample=False)
        gen_time = time.time() - t0

        response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
        results.append({
            "prompt": prompt[:60],
            "response": response[:400],
            "n_tokens": outputs.shape[-1] - inputs["input_ids"].shape[-1],
            "time_s": round(gen_time, 2),
        })
    return results


# ═══════════════════════════════════════════════════════════════════════
# PPL evaluation
# ═══════════════════════════════════════════════════════════════════════

def test_ppl(model, tokenizer, device, seq_len=2048, stride=1024):
    from datasets import load_dataset
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join([t for t in dataset["text"] if t.strip()])

    encodings = tokenizer(text, return_tensors="pt")
    input_ids = encodings.input_ids.to(device)

    nlls = []
    total_len = input_ids.shape[1]
    for begin_loc in range(0, total_len - seq_len, stride):
        end_loc = begin_loc + seq_len
        input_chunk = input_ids[:, begin_loc:end_loc]
        target_chunk = input_chunk.clone()
        target_chunk[:, :-stride] = -100

        with torch.no_grad():
            outputs = model(input_ids=input_chunk, labels=target_chunk)
            neg_log_likelihood = outputs.loss

        nlls.append(neg_log_likelihood.item())
        if len(nlls) % 50 == 0:
            current_ppl = torch.exp(torch.tensor(nlls).mean()).item()
            print(f"    Chunks: {len(nlls)}, current PPL: {current_ppl:.4f}", flush=True)

    ppl = torch.exp(torch.tensor(nlls).mean()).item()
    return {"ppl": round(ppl, 4), "n_chunks": len(nlls)}


# ═══════════════════════════════════════════════════════════════════════
# Vision inference
# ═══════════════════════════════════════════════════════════════════════

def make_test_images():
    """Create test images for vision evaluation."""
    tests = []

    # 1. Shape recognition
    img = Image.new('RGB', (512, 512), 'white')
    draw = ImageDraw.Draw(img)
    draw.ellipse([40, 40, 200, 200], fill='red', outline='darkred', width=3)
    draw.rectangle([280, 50, 470, 190], fill='blue', outline='darkblue', width=3)
    draw.polygon([(256, 280), (140, 460), (372, 460)], fill='green', outline='darkgreen')
    tests.append(("shapes", img, "Describe all the shapes and their colors in this image."))

    # 2. Text/OCR
    img = Image.new('RGB', (600, 200), 'white')
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 32)
    except Exception:
        font = ImageFont.load_default()
    draw.text((30, 30), "Hello, World!", fill='black', font=font)
    draw.text((30, 90), "AutoRound + Qwen3.5", fill='blue', font=font)
    draw.text((30, 140), "2024 Quantization Test", fill='darkred', font=font)
    tests.append(("text_ocr", img, "Read all the text shown in this image. What does each line say?"))

    # 3. Counting
    img = Image.new('RGB', (512, 512), 'white')
    draw = ImageDraw.Draw(img)
    positions = [(80, 80), (200, 60), (350, 100), (100, 250),
                 (270, 230), (420, 280), (200, 400)]
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'cyan', 'magenta']
    for (x, y), c in zip(positions, colors):
        draw.ellipse([x-30, y-30, x+30, y+30], fill=c, outline='black', width=2)
    tests.append(("counting", img, "How many circles are in this image? What colors are they?"))

    # 4. Chart
    img = Image.new('RGB', (500, 400), 'white')
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
        title_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18)
    except Exception:
        font = ImageFont.load_default()
        title_font = font
    draw.text((150, 10), "Sales by Quarter", fill='black', font=title_font)
    bars = [("Q1", 120, 'steelblue'), ("Q2", 200, 'coral'),
            ("Q3", 160, 'mediumseagreen'), ("Q4", 250, 'mediumpurple')]
    base_y = 350
    for i, (label, height, color) in enumerate(bars):
        x = 60 + i * 100
        draw.rectangle([x, base_y - height, x + 70, base_y], fill=color, outline='black')
        draw.text((x + 15, base_y + 5), label, fill='black', font=font)
        draw.text((x + 15, base_y - height - 20), str(height), fill='black', font=font)
    draw.line([(50, 50), (50, base_y)], fill='black', width=2)
    draw.line([(50, base_y), (470, base_y)], fill='black', width=2)
    tests.append(("chart", img, "What does this chart show? Which quarter had the highest value?"))

    # 5. Spatial
    img = Image.new('RGB', (512, 512), 'lightyellow')
    draw = ImageDraw.Draw(img)
    draw.rectangle([180, 200, 330, 380], fill='orange', outline='black', width=2)
    draw.polygon([(190, 200), (220, 130), (250, 200)], fill='orange', outline='black')
    draw.polygon([(260, 200), (290, 130), (320, 200)], fill='orange', outline='black')
    draw.ellipse([210, 240, 240, 270], fill='white', outline='black')
    draw.ellipse([270, 240, 300, 270], fill='white', outline='black')
    draw.ellipse([240, 290, 270, 310], fill='pink', outline='black')
    draw.ellipse([380, 340, 440, 400], fill='red', outline='darkred', width=2)
    tests.append(("spatial", img, "What do you see in this image? Describe the objects and their relative positions."))

    return tests


def test_vision(model, processor, device, max_new_tokens=200):
    """Run vision tests. Returns list of results."""
    results = []
    if processor is None:
        return [{"test_id": "all", "error": "No processor available", "success": False}]

    test_cases = make_test_images()
    for test_id, image, prompt in test_cases:
        try:
            messages = [{"role": "user", "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ]}]
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = processor(text=text, images=[image], return_tensors="pt", padding=True)
            inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

            t0 = time.time()
            with torch.no_grad():
                outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
            gen_time = time.time() - t0

            n_new = outputs.shape[-1] - inputs["input_ids"].shape[-1]
            response = processor.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)

            results.append({
                "test_id": test_id,
                "prompt": prompt,
                "response": response[:400],
                "time_s": round(gen_time, 2),
                "n_tokens": n_new,
                "tokens_per_sec": round(n_new / gen_time, 1) if gen_time > 0 else 0,
                "success": True,
            })
        except Exception as e:
            results.append({
                "test_id": test_id,
                "success": False,
                "error": str(e)[:200],
            })
    return results


# ═══════════════════════════════════════════════════════════════════════
# Main pipeline
# ═══════════════════════════════════════════════════════════════════════

def evaluate_single_model(name, model_path, device, output_dir, skip_ppl=False):
    """Evaluate a single model and save results."""
    report = {"name": name, "model_path": model_path, "device": device, "tests": {}}

    print(f"\n{'━'*70}")
    print(f"  Evaluating: {name}")
    print(f"  Path: {model_path}")
    print(f"{'━'*70}")

    if not os.path.isdir(model_path):
        print(f"  ⚠️  Path not found, skipping")
        report["error"] = "path not found"
        return report

    # 1. Load model
    print(f"\n  [1/5] Loading model...", flush=True)
    try:
        model, tokenizer, processor, load_time, is_quantized, backend, is_vlm, actual_device = load_model(model_path, device)
        print(f"  ✅ Loaded in {load_time:.1f}s (quantized={is_quantized}, vlm={is_vlm}, backend={backend}, device={actual_device})")
        report["load_time_s"] = round(load_time, 1)
        report["is_quantized"] = is_quantized
        report["is_vlm"] = is_vlm
        report["backend"] = backend
        # Use actual_device for all subsequent operations
        device = str(actual_device)
    except Exception as e:
        print(f"  ❌ Load failed: {e}")
        report["error"] = str(e)[:300]
        return report

    # 2. Text inference
    print(f"\n  [2/5] Text inference...", flush=True)
    try:
        text_result = test_text_inference(model, tokenizer, device)
        print(f"  ✅ {text_result['n_tokens']} tokens in {text_result['time_s']}s ({text_result['tokens_per_sec']} tok/s)")
        print(f"  Response: {text_result['response'][:100]}...")
        report["tests"]["text_inference"] = text_result
    except Exception as e:
        print(f"  ❌ Text inference failed: {e}")
        report["tests"]["text_inference"] = {"error": str(e)[:200]}

    # 3. Chat quality
    print(f"\n  [3/5] Chat quality (5 prompts)...", flush=True)
    try:
        chat_results = test_chat_quality(model, tokenizer, device)
        for i, r in enumerate(chat_results):
            print(f"    [{i+1}] {r['prompt'][:40]}... → {r['n_tokens']} tokens")
        report["tests"]["chat_quality"] = chat_results
    except Exception as e:
        print(f"  ❌ Chat failed: {e}")
        report["tests"]["chat_quality"] = {"error": str(e)[:200]}

    # 4. PPL
    if not skip_ppl:
        print(f"\n  [4/5] PPL evaluation (wikitext-2)...", flush=True)
        try:
            ppl_result = test_ppl(model, tokenizer, device)
            print(f"  ✅ PPL = {ppl_result['ppl']}")
            report["tests"]["ppl"] = ppl_result
        except Exception as e:
            print(f"  ❌ PPL failed: {e}")
            report["tests"]["ppl"] = {"error": str(e)[:200]}
    else:
        print(f"\n  [4/5] PPL skipped")

    # 5. Vision
    if is_vlm and processor is not None:
        print(f"\n  [5/5] Vision inference (5 tests)...", flush=True)
        try:
            vision_results = test_vision(model, processor, device)
            passed = sum(1 for r in vision_results if r.get("success"))
            print(f"  ✅ Vision: {passed}/{len(vision_results)} passed")
            for r in vision_results:
                if r.get("success"):
                    print(f"    [{r['test_id']}] {r['response'][:80]}...")
                else:
                    print(f"    [{r['test_id']}] ❌ {r.get('error', '')[:80]}")
            report["tests"]["vision"] = vision_results
        except Exception as e:
            print(f"  ❌ Vision failed: {e}")
            report["tests"]["vision"] = [{"error": str(e)[:200], "success": False}]
    else:
        print(f"\n  [5/5] Vision skipped (not VLM or no processor)")

    # Save report
    report_path = os.path.join(output_dir, f"{name}_eval.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\n  📄 Report saved: {report_path}")

    # Cleanup
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output_dir", default="./output/eval_logs")
    parser.add_argument("--skip_ppl", action="store_true")
    parser.add_argument("--models", nargs="*", default=None,
                        help="Specific model names to evaluate (default: all)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    models_to_eval = MODELS
    if args.models:
        models_to_eval = [(n, p) for n, p in MODELS if n in args.models]

    print(f"{'='*70}")
    print(f"  Full Evaluation Pipeline - Qwen3.5 27B")
    print(f"  Device: {args.device}")
    print(f"  Models: {len(models_to_eval)}")
    print(f"  Output: {args.output_dir}")
    print(f"{'='*70}")

    all_reports = {}
    for name, path in models_to_eval:
        report = evaluate_single_model(name, path, args.device, args.output_dir, args.skip_ppl)
        all_reports[name] = report

    # Save combined report
    combined_path = os.path.join(args.output_dir, "all_results.json")
    with open(combined_path, "w") as f:
        json.dump(all_reports, f, indent=2, ensure_ascii=False)

    # Print summary table
    print(f"\n{'='*70}")
    print(f"  SUMMARY")
    print(f"{'='*70}")
    print(f"{'Model':<25} {'Load(s)':<10} {'PPL':<10} {'Text tok/s':<12} {'Vision':<10}")
    print(f"{'-'*70}")
    for name, report in all_reports.items():
        load_t = report.get("load_time_s", "N/A")
        ppl = report.get("tests", {}).get("ppl", {}).get("ppl", "N/A")
        text_tps = report.get("tests", {}).get("text_inference", {}).get("tokens_per_sec", "N/A")
        vision_tests = report.get("tests", {}).get("vision", [])
        if isinstance(vision_tests, list):
            v_pass = sum(1 for v in vision_tests if v.get("success"))
            v_total = len(vision_tests)
            vision = f"{v_pass}/{v_total}"
        else:
            vision = "N/A"
        print(f"{name:<25} {str(load_t):<10} {str(ppl):<10} {str(text_tps):<12} {vision:<10}")

    print(f"\n📄 Combined report: {combined_path}")


if __name__ == "__main__":
    main()
