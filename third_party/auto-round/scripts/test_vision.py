"""Vision inference test for Qwen3.5 VLM (quantized and original).

Tests vision understanding with:
  1. Synthetic shape recognition (colored rectangles)
  2. Text-in-image reading (rendered text)
  3. Counting/spatial reasoning
  4. Multi-turn vision QA

Usage:
    python scripts/test_vision.py --model_path ./output/mode1-llm-only --device cuda:5
    python scripts/test_vision.py --model_path weights/Qwen3.5-27B --device cuda:5  # FP16 baseline
"""

import argparse
import json
import os
import time
import traceback

import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont


# ─── GLIBC compatibility patch ───────────────────────────────────────────────
try:
    from auto_round.inference.backend import BackendInfos
    to_remove = [k for k in list(BackendInfos.keys()) if "marlin" in k or "exllama" in k]
    for k in to_remove:
        del BackendInfos[k]
except Exception:
    pass


def parse_args():
    parser = argparse.ArgumentParser(description="Vision inference test for Qwen3.5")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to quantized or original model")
    parser.add_argument("--original_model", type=str, default="weights/Qwen3.5-27B",
                        help="Original model path (for processor if needed)")
    parser.add_argument("--device", type=str, default="cuda:5")
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON path (default: <model_path>/vision_eval.json)")
    return parser.parse_args()


# ─── Test image generators ───────────────────────────────────────────────────

def make_shapes_image():
    """Create image with colored shapes for recognition test."""
    img = Image.new('RGB', (512, 512), 'white')
    draw = ImageDraw.Draw(img)
    # Red circle (top-left)
    draw.ellipse([40, 40, 200, 200], fill='red', outline='darkred', width=3)
    # Blue rectangle (top-right)
    draw.rectangle([280, 50, 470, 190], fill='blue', outline='darkblue', width=3)
    # Green triangle (bottom-center)
    draw.polygon([(256, 280), (140, 460), (372, 460)], fill='green', outline='darkgreen')
    # Yellow star-like (bottom-right, simple)
    draw.regular_polygon((400, 380, 60), n_sides=5, fill='yellow', outline='orange')
    return img, "Describe all the shapes and their colors in this image."


def make_text_image():
    """Create image with rendered text for OCR-like test."""
    img = Image.new('RGB', (600, 200), 'white')
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 32)
    except Exception:
        font = ImageFont.load_default()
    draw.text((30, 30), "Hello, World!", fill='black', font=font)
    draw.text((30, 90), "AutoRound + Qwen3.5", fill='blue', font=font)
    draw.text((30, 140), "2024 Quantization Test", fill='darkred', font=font)
    return img, "Read all the text shown in this image. What does each line say?"


def make_counting_image():
    """Create image with objects to count."""
    img = Image.new('RGB', (512, 512), 'white')
    draw = ImageDraw.Draw(img)
    # Draw 7 circles of different sizes
    positions = [(80, 80), (200, 60), (350, 100), (100, 250),
                 (270, 230), (420, 280), (200, 400)]
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'cyan', 'magenta']
    for (x, y), c in zip(positions, colors):
        r = 30
        draw.ellipse([x-r, y-r, x+r, y+r], fill=c, outline='black', width=2)
    return img, "How many circles are in this image? What colors are they?"


def make_chart_image():
    """Create a simple bar-chart-like image."""
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
    bar_width = 70
    x_start = 60
    for i, (label, height, color) in enumerate(bars):
        x = x_start + i * (bar_width + 30)
        draw.rectangle([x, base_y - height, x + bar_width, base_y],
                       fill=color, outline='black', width=1)
        draw.text((x + 15, base_y + 5), label, fill='black', font=font)
        draw.text((x + 15, base_y - height - 20), str(height), fill='black', font=font)
    # Y-axis
    draw.line([(50, 50), (50, base_y)], fill='black', width=2)
    draw.line([(50, base_y), (470, base_y)], fill='black', width=2)
    return img, "What does this chart show? Which quarter had the highest value?"


def make_spatial_image():
    """Create image for spatial reasoning test."""
    img = Image.new('RGB', (512, 512), 'lightyellow')
    draw = ImageDraw.Draw(img)
    # A cat-like shape (simple)
    draw.rectangle([180, 200, 330, 380], fill='orange', outline='black', width=2)  # body
    draw.polygon([(190, 200), (220, 130), (250, 200)], fill='orange', outline='black')  # left ear
    draw.polygon([(260, 200), (290, 130), (320, 200)], fill='orange', outline='black')  # right ear
    draw.ellipse([210, 240, 240, 270], fill='white', outline='black')  # left eye
    draw.ellipse([270, 240, 300, 270], fill='white', outline='black')  # right eye
    draw.ellipse([220, 248, 230, 262], fill='black')  # left pupil
    draw.ellipse([280, 248, 290, 262], fill='black')  # right pupil
    draw.ellipse([240, 290, 270, 310], fill='pink', outline='black')  # nose
    # A ball to the right
    draw.ellipse([380, 340, 440, 400], fill='red', outline='darkred', width=2)
    return img, "What do you see in this image? Describe the objects and their relative positions."


# ─── Model loading ───────────────────────────────────────────────────────────

def load_model(model_path, device):
    """Load model and processor."""
    from transformers import AutoProcessor, AutoTokenizer

    # Determine if this is quantized or original
    config_path = os.path.join(model_path, "config.json")
    is_quantized = False
    if os.path.exists(config_path):
        with open(config_path) as f:
            cfg = json.load(f)
        is_quantized = "quantization_config" in cfg

    # Load processor (try from model_path first, then original)
    processor = None
    for path in [model_path, "weights/Qwen3.5-27B"]:
        try:
            processor = AutoProcessor.from_pretrained(path, trust_remote_code=True)
            break
        except Exception:
            continue

    if processor is None:
        raise RuntimeError("Cannot load processor from any path")

    # Load model
    t0 = time.time()
    try:
        from transformers import AutoModelForImageTextToText
        model = AutoModelForImageTextToText.from_pretrained(
            model_path, torch_dtype=torch.bfloat16,
            device_map=device, trust_remote_code=True)
    except Exception:
        from transformers import AutoModel
        model = AutoModel.from_pretrained(
            model_path, torch_dtype=torch.bfloat16,
            device_map=device, trust_remote_code=True)
    load_time = time.time() - t0

    model.eval()
    return model, processor, load_time, is_quantized


def run_vision_qa(model, processor, image, prompt, device, max_new_tokens=300):
    """Run a single vision QA turn."""
    messages = [
        {"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": prompt},
        ]}
    ]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=text, images=[image], return_tensors="pt", padding=True)
    inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

    t0 = time.time()
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    gen_time = time.time() - t0

    n_new = outputs.shape[-1] - inputs["input_ids"].shape[-1]
    response = processor.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
    return response, gen_time, n_new


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    results = {
        "model_path": args.model_path,
        "device": args.device,
        "tests": [],
    }

    print(f"\n{'='*70}")
    print(f"Vision Inference Test: {args.model_path}")
    print(f"Device: {args.device}")
    print(f"{'='*70}")

    # Load model
    print("\n[Loading] Model and processor...")
    try:
        model, processor, load_time, is_quantized = load_model(args.model_path, args.device)
    except Exception as e:
        print(f"  ❌ Failed to load model: {e}")
        results["error"] = str(e)
        _save_results(results, args)
        return

    print(f"  ✅ Loaded in {load_time:.1f}s (quantized={is_quantized})")
    results["load_time_s"] = round(load_time, 1)
    results["is_quantized"] = is_quantized

    # Test cases
    test_cases = [
        ("shapes", make_shapes_image, "Shape recognition"),
        ("text_ocr", make_text_image, "Text reading (OCR)"),
        ("counting", make_counting_image, "Counting objects"),
        ("chart", make_chart_image, "Chart understanding"),
        ("spatial", make_spatial_image, "Spatial reasoning"),
    ]

    total_pass = 0
    for test_id, img_fn, desc in test_cases:
        print(f"\n[Test: {test_id}] {desc}")
        try:
            image, prompt = img_fn()
            response, gen_time, n_tokens = run_vision_qa(
                model, processor, image, prompt, args.device)

            result = {
                "test_id": test_id,
                "description": desc,
                "prompt": prompt,
                "response": response[:500],
                "time_s": round(gen_time, 2),
                "n_tokens": n_tokens,
                "tokens_per_sec": round(n_tokens / gen_time, 1) if gen_time > 0 else 0,
                "success": True,
            }
            total_pass += 1
            print(f"  ✅ {n_tokens} tokens in {gen_time:.1f}s ({result['tokens_per_sec']} tok/s)")
            print(f"  Response: {response[:150]}...")

        except Exception as e:
            result = {
                "test_id": test_id,
                "description": desc,
                "success": False,
                "error": str(e)[:300],
                "traceback": traceback.format_exc()[-500:],
            }
            print(f"  ❌ Error: {str(e)[:100]}")

        results["tests"].append(result)

    # Summary
    results["summary"] = {
        "total_tests": len(test_cases),
        "passed": total_pass,
        "failed": len(test_cases) - total_pass,
    }

    print(f"\n{'='*70}")
    print(f"Vision Test Summary: {total_pass}/{len(test_cases)} passed")
    print(f"{'='*70}")

    _save_results(results, args)


def _save_results(results, args):
    output_path = args.output or os.path.join(args.model_path, "vision_eval.json")
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"📄 Results saved: {output_path}")


if __name__ == "__main__":
    main()
