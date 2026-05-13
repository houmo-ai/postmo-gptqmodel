"""Gemma4 量化模型 PPL 评测脚本 (per-token 滑动窗口法)。

与 Qwen3.5 报告一致，采用滑动窗口 (seq_len=2048, stride=1024) 计算 per-token PPL，
而非 lm_eval 的 byte-level word_perplexity。

Gemma4 是 VLM 架构 (Gemma4ForConditionalGeneration)，需要用原始架构类加载。

用法:
  # 量化模型
  python scripts_gemma4/eval/eval_ppl.py \
    --model output/gemma4-mode1 --device cuda:0

  # BF16 原始模型 baseline
  python scripts_gemma4/eval/eval_ppl.py \
    --model weights/gemma-4-31B-it --dtype bfloat16 --device cuda:0
"""
import argparse
import json
import math
import os
import time

import torch
from torch.nn import CrossEntropyLoss
from tqdm import tqdm

from scripts_gemma4.eval.common import ensure_local_gptqmodel_path, patch_gptqmodel_for_gemma4

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def load_model(model_path, device, dtype_str=None):
    """Load model with correct architecture class."""
    # Block gptqmodel C++ backends that may have compatibility issues
    try:
        import auto_round.inference.backend as _bk
        for k in [k for k in _bk.BackendInfos if k.startswith("gptqmodel:")]:
            del _bk.BackendInfos[k]
    except Exception:
        pass

    config_path = os.path.join(model_path, "config.json")
    with open(config_path) as f:
        cfg = json.load(f)
    archs = cfg.get("architectures", [])
    is_quantized = "quantization_config" in cfg

    load_kwargs = dict(device_map=device, trust_remote_code=True)
    if dtype_str:
        load_kwargs["torch_dtype"] = getattr(torch, dtype_str)

    model = None

    # For quantized models, try GPTQModel first
    if is_quantized:
        gptq_device_map = device
        if isinstance(gptq_device_map, str) and gptq_device_map not in {
            "auto",
            "balanced",
            "balanced_low_0",
            "sequential",
        }:
            gptq_device_map = "auto"
        try:
            ensure_local_gptqmodel_path()
            patch_gptqmodel_for_gemma4()
            from gptqmodel import GPTQModel, BACKEND
            gptq_model = GPTQModel.load(
                model_path, device_map=gptq_device_map,
                backend=BACKEND.TORCH, trust_remote_code=True,
            )
            model = gptq_model.model
            print(f"Loaded via GPTQModel (TORCH backend): {type(model).__name__}")
        except Exception as e:
            print(f"GPTQModel.load failed: {e}, falling back...")

    if model is None:
        if "Gemma4ForConditionalGeneration" in archs:
            from transformers import AutoModelForImageTextToText
            model = AutoModelForImageTextToText.from_pretrained(model_path, **load_kwargs)
        else:
            from transformers import AutoModelForCausalLM
            model = AutoModelForCausalLM.from_pretrained(model_path, **load_kwargs)
        print(f"Loaded: {type(model).__name__}")

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    return model, tokenizer


def compute_ppl(model, tokenizer, seq_len=2048, stride=1024, device="cuda:0"):
    """滑动窗口 per-token PPL (与 Qwen3.5 报告一致)。

    方法: 将 wikitext-2-raw-v1 test set 拼接为长序列，以 stride 滑动，
    每个窗口计算交叉熵损失，只对新增 token 计算 loss，最终 PPL = exp(mean_NLL)。
    """
    from datasets import load_dataset
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(dataset["text"])

    encodings = tokenizer(text, return_tensors="pt")
    input_ids = encodings.input_ids  # [1, total_len]
    total_len = input_ids.size(1)
    print(f"Total tokens: {total_len}")

    loss_fn = CrossEntropyLoss(reduction="none")
    nlls = []
    n_tokens = 0
    prev_end = 0

    model.eval()
    pbar = tqdm(range(0, total_len, stride), desc="PPL eval")

    for begin in pbar:
        end = min(begin + seq_len, total_len)
        window_len = end - begin
        # Only count loss for tokens not yet scored by previous windows
        target_begin = max(prev_end - begin, 1)

        ids = input_ids[:, begin:end].to(device)

        with torch.no_grad():
            outputs = model(input_ids=ids)
            logits = outputs.logits  # [1, window_len, vocab]

        # Predict token[i] from logits[i-1]
        shift_logits = logits[0, target_begin - 1: window_len - 1, :]
        shift_labels = ids[0, target_begin: window_len]

        loss = loss_fn(shift_logits, shift_labels)
        nlls.append(loss.sum().item())
        n_tokens += loss.numel()
        prev_end = end

        if end >= total_len:
            break

    mean_nll = sum(nlls) / n_tokens
    ppl = math.exp(mean_nll)
    print(f"Chunks: {len(nlls)}, Tokens evaluated: {n_tokens}")
    print(f"Mean NLL: {mean_nll:.6f}")
    print(f"PPL: {ppl:.4f}")
    return ppl, mean_nll, n_tokens


def build_result_payload(model, seq_len, stride, device, dtype, ppl, mean_nll, n_tokens, elapsed):
    return {
        "model": model,
        "seq_len": seq_len,
        "stride": stride,
        "device": device,
        "dtype": dtype,
        "ppl": ppl,
        "mean_nll": mean_nll,
        "n_tokens": n_tokens,
        "elapsed_seconds": elapsed,
    }


def write_result_payload(path, payload):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def main():
    parser = argparse.ArgumentParser(description="Gemma4 PPL 评测 (滑动窗口 per-token)")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--seq_len", type=int, default=2048)
    parser.add_argument("--stride", type=int, default=1024)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--dtype", type=str, default=None,
                        help="显式指定 dtype (bfloat16/float16)，BF16 原始模型评测时使用")
    parser.add_argument("--output", type=str, default=None, help="结果保存路径 (JSON)")
    args = parser.parse_args()

    print(f"Model: {args.model}")
    print(f"seq_len={args.seq_len}, stride={args.stride}")
    print(f"Device: {args.device}")

    model, tokenizer = load_model(args.model, args.device, args.dtype)

    st = time.time()
    ppl, mean_nll, n_tokens = compute_ppl(
        model, tokenizer,
        seq_len=args.seq_len,
        stride=args.stride,
        device=args.device,
    )
    elapsed = int(time.time() - st)
    print(f"\n{'='*50}")
    print(f"Result: PPL = {ppl:.4f}  (Mean NLL = {mean_nll:.6f})")
    print(f"Evaluation time: {elapsed}s")
    print(f"{'='*50}")
    if args.output:
        payload = build_result_payload(
            model=args.model,
            seq_len=args.seq_len,
            stride=args.stride,
            device=args.device,
            dtype=args.dtype,
            ppl=ppl,
            mean_nll=mean_nll,
            n_tokens=n_tokens,
            elapsed=elapsed,
        )
        write_result_payload(args.output, payload)
        print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
