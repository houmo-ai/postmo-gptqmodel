"""Rotation precision verification for Qwen3.5-MoE (35B-A3B).

Verifies that Hadamard rotation preserves model outputs by comparing:
  1. Numerical precision: logits before vs after rotation (cosine sim, max diff)
  2. Chat consistency: generated text before vs after rotation

Usage:
    python scripts_qwen35moe/verify_rotation.py \
        --model /data01/nfs_shared/Qwen3.5-35B-A3B \
        --device cuda:0

    # Also test vision (requires VLM model)
    python scripts_qwen35moe/verify_rotation.py \
        --model /data01/nfs_shared/Qwen3.5-35B-A3B \
        --device cuda:0 --test_vision
"""

import argparse
import os
import sys
import time

import torch

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Force MoE fallback on GPUs without SM90 (A100 has SM80, grouped_mm needs SM90)
def _patch_grouped_mm_for_sm80():
    """Monkey-patch transformers MoE to use fallback on non-SM90 GPUs."""
    try:
        if torch.cuda.is_available():
            cc = torch.cuda.get_device_capability(0)
            if cc[0] < 9:
                import transformers.integrations.moe as _moe
                _moe._can_use_grouped_mm = lambda *a, **kw: False
                print(f"[Patch] Forced MoE fallback for SM{cc[0]}{cc[1]} (grouped_mm needs SM90)")
    except Exception:
        pass

_patch_grouped_mm_for_sm80()

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def parse_args():
    parser = argparse.ArgumentParser(description="Verify Qwen3.5-MoE rotation precision")
    parser.add_argument("--model", type=str, required=True, help="Model path")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test_vision", action="store_true", help="Also test vision rotation")
    parser.add_argument("--max_new_tokens", type=int, default=50, help="Max tokens for chat test")
    return parser.parse_args()


def get_logits(model, input_ids, device):
    """Get logits from model for given input_ids."""
    model.eval()
    with torch.no_grad():
        out = model(input_ids=input_ids.to(device))
    return out.logits.float().cpu()


def compare_logits(logits_before, logits_after, label=""):
    """Compare two logit tensors and print metrics."""
    cos_sim = torch.nn.functional.cosine_similarity(
        logits_before.reshape(1, -1), logits_after.reshape(1, -1)
    ).item()
    max_diff = (logits_before - logits_after).abs().max().item()
    mean_diff = (logits_before - logits_after).abs().mean().item()
    rel_diff = ((logits_before - logits_after).abs() / (logits_before.abs() + 1e-8)).mean().item()

    pred_before = logits_before[:, -1, :].argmax(dim=-1)
    pred_after = logits_after[:, -1, :].argmax(dim=-1)
    pred_match = (pred_before == pred_after).all().item()

    print(f"  [{label}] Cosine similarity: {cos_sim:.10f}")
    print(f"  [{label}] Max absolute diff:  {max_diff:.6e}")
    print(f"  [{label}] Mean absolute diff: {mean_diff:.6e}")
    print(f"  [{label}] Mean relative diff: {rel_diff:.6e}")
    print(f"  [{label}] Last-token pred match: {pred_match}")

    return {
        "cosine_sim": cos_sim,
        "max_diff": max_diff,
        "mean_diff": mean_diff,
        "rel_diff": rel_diff,
        "pred_match": pred_match,
    }


def generate_text(model, tokenizer, prompt, device, max_new_tokens=50):
    """Generate text with the model."""
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    resp = tokenizer.decode(out[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
    return resp


def _load_model_for_llm_test(model_path, device="cpu"):
    """Load model for LLM-only rotation test.

    Tries AutoModelForCausalLM first, falls back to AutoModelForImageTextToText
    (needed for VLM models like Qwen3_5MoeForConditionalGeneration).
    """
    load_kwargs = dict(torch_dtype=torch.bfloat16, device_map=device, trust_remote_code=True)
    try:
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(model_path, **load_kwargs)
        return model
    except (ValueError, KeyError):
        pass
    from transformers import AutoModelForImageTextToText
    model = AutoModelForImageTextToText.from_pretrained(model_path, **load_kwargs)
    return model


def test_llm_rotation(args):
    """Test LLM rotation precision."""
    from transformers import AutoTokenizer
    from scripts_qwen35moe.rotation_utils_moe import (
        _get_hadamard_matrix,
        fuse_llm_norms_moe,
        rotate_llm_moe,
    )

    print(f"\n{'='*70}")
    print(f"LLM Rotation Precision Test")
    print(f"Model: {args.model}")
    print(f"{'='*70}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    # Prepare test inputs
    test_prompts = [
        "What is quantum computing?",
        "Write a Python function to sort a list.",
        "Explain the theory of relativity in simple terms.",
    ]
    chat_prompt = test_prompts[0]

    # ── Phase 1: Get pre-rotation outputs ──
    print("\n[Phase 1] Loading model and getting pre-rotation outputs...")
    model = _load_model_for_llm_test(args.model, device="cpu")
    model = model.to(args.device)
    model.eval()
    print(f"  Model class: {type(model).__name__}")

    # Get logits for all prompts
    pre_logits_list = []
    for prompt in test_prompts:
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        input_ids = tokenizer(text, return_tensors="pt")["input_ids"]
        logits = get_logits(model, input_ids, args.device)
        pre_logits_list.append(logits)

    # Generate chat response
    print("  Generating pre-rotation chat response...")
    pre_chat = generate_text(model, tokenizer, chat_prompt, args.device, args.max_new_tokens)
    print(f"  Pre-rotation response: {pre_chat[:200]}")

    # Move to CPU for rotation
    model = model.to("cpu")
    torch.cuda.empty_cache()

    # ── Phase 2: Apply rotation ──
    print("\n[Phase 2] Applying Hadamard rotation...")
    t0 = time.time()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    text_config = getattr(model.config, 'text_config', model.config)
    hidden_size = text_config.hidden_size
    Q_llm = _get_hadamard_matrix(hidden_size, args.device)

    fuse_llm_norms_moe(model, args.device)
    rotate_llm_moe(model, Q_llm, args.device)

    rot_time = time.time() - t0
    print(f"  Rotation completed in {rot_time:.1f}s")

    # ── Phase 3: Get post-rotation outputs ──
    print("\n[Phase 3] Getting post-rotation outputs...")
    model = model.to(args.device)
    model.eval()

    post_logits_list = []
    for prompt in test_prompts:
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        input_ids = tokenizer(text, return_tensors="pt")["input_ids"]
        logits = get_logits(model, input_ids, args.device)
        post_logits_list.append(logits)

    print("  Generating post-rotation chat response...")
    post_chat = generate_text(model, tokenizer, chat_prompt, args.device, args.max_new_tokens)
    print(f"  Post-rotation response: {post_chat[:200]}")

    # ── Phase 4: Compare results ──
    print(f"\n{'='*70}")
    print("Numerical Precision Comparison:")
    print(f"{'='*70}")

    results = []
    for i, prompt in enumerate(test_prompts):
        print(f"\nPrompt {i+1}: '{prompt[:50]}...'")
        r = compare_logits(pre_logits_list[i], post_logits_list[i], f"Prompt {i+1}")
        results.append(r)

    print(f"\n{'='*70}")
    print("Chat Consistency:")
    print(f"{'='*70}")
    print(f"  Pre:  {pre_chat[:300]}")
    print(f"  Post: {post_chat[:300]}")
    chat_match = pre_chat.strip() == post_chat.strip()
    print(f"  Exact match: {chat_match}")

    # Summary
    avg_cos = sum(r["cosine_sim"] for r in results) / len(results)
    all_pred_match = all(r["pred_match"] for r in results)
    max_max_diff = max(r["max_diff"] for r in results)

    print(f"\n{'='*70}")
    print("Summary:")
    print(f"  Average cosine similarity: {avg_cos:.10f}")
    print(f"  Max absolute diff (worst): {max_max_diff:.6e}")
    print(f"  All predictions match:     {all_pred_match}")
    print(f"  Chat response match:       {chat_match}")

    passed = avg_cos > 0.9999 and all_pred_match
    status = "✅ PASSED" if passed else "⚠️  DEGRADED (but may be acceptable for bf16)"
    print(f"  Status: {status}")
    print(f"{'='*70}")

    del model
    torch.cuda.empty_cache()
    return passed


def test_vision_rotation(args):
    """Test vision + LLM rotation precision."""
    from PIL import Image, ImageDraw
    from scripts_qwen35moe.rotation_utils_moe import apply_hadamard_rotation_moe

    print(f"\n{'='*70}")
    print(f"Vision + LLM Rotation Precision Test")
    print(f"{'='*70}")

    from transformers import AutoProcessor, AutoModelForImageTextToText, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)

    # Create test image
    img = Image.new('RGB', (256, 256), 'white')
    draw = ImageDraw.Draw(img)
    draw.rectangle([50, 50, 200, 200], fill='red', outline='black', width=2)
    draw.ellipse([80, 80, 170, 170], fill='blue', outline='black', width=2)
    prompt = "Describe the shapes and colors in this image."

    # ── Phase 1: Pre-rotation ──
    print("\n[Phase 1] Loading VLM model...")
    model = AutoModelForImageTextToText.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="cpu", trust_remote_code=True
    )
    model = model.to(args.device)
    model.eval()

    messages = [{"role": "user", "content": [
        {"type": "image", "image": img},
        {"type": "text", "text": prompt},
    ]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=text, images=[img], return_tensors="pt", padding=True)
    inputs = {k: v.to(args.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

    print("  Getting pre-rotation vision output...")
    with torch.no_grad():
        pre_out = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
    pre_resp = processor.decode(pre_out[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
    print(f"  Pre-rotation: {pre_resp[:200]}")

    # ── Phase 2: Apply rotation ──
    model = model.to("cpu")
    torch.cuda.empty_cache()

    print("\n[Phase 2] Applying full rotation (LLM + vision)...")
    t0 = time.time()
    model, Q_llm, Q_vision = apply_hadamard_rotation_moe(
        model, rotate_vision="full", device=args.device, seed=args.seed
    )
    print(f"  Rotation completed in {time.time() - t0:.1f}s")

    # ── Phase 3: Post-rotation ──
    model = model.to(args.device)
    model.eval()

    inputs = processor(text=text, images=[img], return_tensors="pt", padding=True)
    inputs = {k: v.to(args.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

    print("\n[Phase 3] Getting post-rotation vision output...")
    with torch.no_grad():
        post_out = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
    post_resp = processor.decode(post_out[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
    print(f"  Post-rotation: {post_resp[:200]}")

    # ── Compare ──
    print(f"\n{'='*70}")
    print("Vision Chat Consistency:")
    print(f"  Pre:  {pre_resp[:300]}")
    print(f"  Post: {post_resp[:300]}")
    match = pre_resp.strip() == post_resp.strip()
    print(f"  Exact match: {match}")
    status = "✅ PASSED" if match else "⚠️  Responses differ (may be acceptable for bf16 precision)"
    print(f"  Status: {status}")
    print(f"{'='*70}")

    del model
    torch.cuda.empty_cache()
    return match


def main():
    args = parse_args()

    llm_passed = test_llm_rotation(args)

    vision_passed = None
    if args.test_vision:
        vision_passed = test_vision_rotation(args)

    print(f"\n{'='*70}")
    print("FINAL RESULTS:")
    print(f"  LLM rotation:    {'✅ PASSED' if llm_passed else '⚠️  CHECK'}")
    if vision_passed is not None:
        print(f"  Vision rotation: {'✅ PASSED' if vision_passed else '⚠️  CHECK'}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
