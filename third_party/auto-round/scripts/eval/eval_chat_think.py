"""评测量化前后模型 think 能力变化：比较思考是否变啰嗦。

对每个模型运行 5 个问题（enable_thinking=True），记录 think 长度和 answer 长度，
保存 JSON 结果，用于后续对比。

用法:
  CUDA_VISIBLE_DEVICES=0 python eval_chat_think.py \
    --model weights/Qwen3.5-27B --label HF --device cuda:0

  CUDA_VISIBLE_DEVICES=1 python eval_chat_think.py \
    --model weights/Qwen3.5-27B-GPTQModel-self-generated-4bit-hessian-mse \
    --label GPTQ --device cuda:0

  CUDA_VISIBLE_DEVICES=2 python eval_chat_think.py \
    --model weights/Qwen3.5-27B-W4G64-autoround/Qwen3.5-27B-w4g64 \
    --label AutoRound --quantized --device cuda:0

  CUDA_VISIBLE_DEVICES=3 python eval_chat_think.py \
    --model weights/Qwen3.5-27B-W4G64-rotate-autoround/Qwen3.5-27B-rotated-w4g64 \
    --label AutoRound+Rotate --quantized --device cuda:0
"""
import argparse
import json
import os
import time

import torch

os.environ["TOKENIZERS_PARALLELISM"] = "false"

QUESTIONS = [
    "你是谁？",
    "中国最长的河流是什么？请简要介绍。",
    "请解方程 2x + 5 = 15",
    "用100字描述春天的景色",
    "写一个Python函数计算斐波那契数列的第n项",
]


def _fix_dual_key_index(model_path):
    """修复 weight_map 中同时包含 CausalLM 和 VLM 两套 key 的问题。

    某些 rotate-autoround 模型同时保存了 model.layers.* (CausalLM) 和
    model.language_model.* (VLM) 两套权重，但数值不同。CausalLM 的是错误的中间
    产物，VLM 的才是正确的最终权重。此函数创建修正后的临时索引文件，仅保留 VLM 键。
    """
    import shutil
    import tempfile

    index_path = os.path.join(model_path, "model.safetensors.index.json")
    if not os.path.exists(index_path):
        return model_path  # 没有分片索引，无需修复

    with open(index_path) as f:
        idx = json.load(f)
    wm = idx["weight_map"]

    causal_keys = [k for k in wm if k.startswith("model.layers.") or
                   k.startswith("model.embed_tokens") or k.startswith("model.norm.")]
    vlm_keys = [k for k in wm if k.startswith("model.language_model.")]
    if not causal_keys or not vlm_keys:
        return model_path  # 不存在双套 key，无需修复

    print(f"  ⚠ 检测到双套 weight_map (CausalLM: {len(causal_keys)}, VLM: {len(vlm_keys)})，修正为仅保留 VLM 键")

    # 构建修正后的 weight_map：仅保留 CausalLM 键 + lm_head
    # CausalLM 键 (model.layers.*, model.embed_tokens.*, model.norm.*) 直接匹配
    # Qwen3_5ForCausalLM 模型结构，无需 VLM→CausalLM 重映射
    fixed_wm = {k: v for k, v in wm.items()
                if not k.startswith("model.language_model.")}
    idx["weight_map"] = fixed_wm

    # 修正 config.json 中的 block_name_to_quantize（VLM 路径 → CausalLM 路径）
    config_path = os.path.join(model_path, "config.json")
    config_needs_fix = False
    config_data = None
    if os.path.exists(config_path):
        with open(config_path) as f:
            config_data = json.load(f)
        qcfg = config_data.get("quantization_config", {})
        btn = qcfg.get("block_name_to_quantize")
        if btn and any("language_model" in b for b in btn):
            qcfg["block_name_to_quantize"] = [
                b.replace("model.language_model.", "model.") for b in btn
            ]
            config_needs_fix = True
            print(f"  ⚠ 修正 block_name_to_quantize: {btn} → {qcfg['block_name_to_quantize']}")

    # 写入临时目录，其他文件软链接（使用绝对路径）
    abs_model_path = os.path.abspath(model_path)
    tmp_dir = tempfile.mkdtemp(prefix="fixed_index_")
    for fn in os.listdir(abs_model_path):
        src = os.path.join(abs_model_path, fn)
        dst = os.path.join(tmp_dir, fn)
        if fn == "model.safetensors.index.json":
            with open(dst, "w") as f:
                json.dump(idx, f)
        elif fn == "config.json" and config_needs_fix:
            with open(dst, "w") as f:
                json.dump(config_data, f, indent=2)
        else:
            os.symlink(src, dst)

    return tmp_dir


def load_model(model_path, device, quantized, dtype=None, use_gptqmodel=False):
    """加载模型，根据架构自动选择加载方式。"""
    load_kwargs = dict(device_map=device, trust_remote_code=True)
    if dtype:
        load_kwargs["torch_dtype"] = getattr(torch, dtype)

    config_path = os.path.join(model_path, "config.json")
    with open(config_path) as f:
        cfg = json.load(f)
    archs = cfg.get("architectures", [])

    # GPTQModel 原生加载（解决 Triton kernel 对齐问题）
    if use_gptqmodel:
        from gptqmodel import GPTQModel
        return GPTQModel.load(model_path, device=device, trust_remote_code=True)

    if quantized:
        # Block gptqmodel C++ backends (may need specific GLIBC)
        try:
            import auto_round.inference.backend as _bk
            for k in [k for k in _bk.BackendInfos if k.startswith("gptqmodel:")]:
                del _bk.BackendInfos[k]
        except Exception:
            pass

        # 修复双套 weight_map 问题（rotate-autoround 模型可能存在）
        model_path = _fix_dual_key_index(model_path)

        if "Qwen3_5ForConditionalGeneration" in archs:
            from transformers.models.qwen3_5.modeling_qwen3_5 import (
                Qwen3_5ForConditionalGeneration,
            )
            return Qwen3_5ForConditionalGeneration.from_pretrained(
                model_path, **load_kwargs
            )

    from transformers import AutoModelForCausalLM
    if not dtype:
        load_kwargs["torch_dtype"] = torch.bfloat16
    return AutoModelForCausalLM.from_pretrained(model_path, **load_kwargs)


def generate_response(model, tokenizer, question, max_new_tokens=2048):
    """生成回复（enable_thinking=True），分离 think 和 answer。"""
    messages = [{"role": "user", "content": question}]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output = model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False
        )

    input_len = inputs["input_ids"].shape[1]
    output_len = output.shape[1] - input_len
    full_response = tokenizer.decode(
        output[0][input_len:], skip_special_tokens=False
    )

    # 分离 think / answer
    # 注意: apply_chat_template(enable_thinking=True) 在 prompt 末尾添加了 <think>，
    # 所以模型输出不含 <think>，但含 </think> 作为思考结束标记
    think_content = ""
    answer_content = full_response
    if "</think>" in full_response:
        parts = full_response.split("</think>", 1)
        think_content = parts[0].replace("<think>", "").strip()
        answer_content = parts[1].strip()
    elif "<think>" in full_response:
        parts = full_response.split("<think>", 1)
        if "</think>" in parts[-1]:
            sub = parts[-1].split("</think>", 1)
            think_content = sub[0].strip()
            answer_content = sub[1].strip()

    # 清理 special tokens
    for tok in ["<|im_end|>", "<|endoftext|>", "<|im_start|>"]:
        answer_content = answer_content.replace(tok, "").strip()
        think_content = think_content.replace(tok, "").strip()

    # 统计 token 数 (think 部分 / answer 部分)
    think_tokens = len(tokenizer.encode(think_content)) if think_content else 0
    answer_tokens = len(tokenizer.encode(answer_content)) if answer_content else 0

    return {
        "think": think_content,
        "answer": answer_content,
        "think_chars": len(think_content),
        "answer_chars": len(answer_content),
        "think_tokens": think_tokens,
        "answer_tokens": answer_tokens,
        "total_output_tokens": output_len,
    }


def main():
    parser = argparse.ArgumentParser(description="Qwen3.5 Think 能力对比评测")
    parser.add_argument("--model", type=str, required=True, help="模型路径")
    parser.add_argument("--label", type=str, required=True, help="模型标签")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--quantized", action="store_true")
    parser.add_argument("--use_gptqmodel", action="store_true",
                        help="使用 GPTQModel.load() 加载 GPTQ 模型")
    parser.add_argument("--dtype", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="eval_think_results")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"模型: {args.label} ({args.model})")
    print(f"{'='*60}")

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = load_model(args.model, args.device, args.quantized, args.dtype, args.use_gptqmodel)

    results = {"label": args.label, "model_path": args.model, "questions": []}

    for i, q in enumerate(QUESTIONS, 1):
        print(f"\n[{i}/{len(QUESTIONS)}] Q: {q}")
        t0 = time.time()
        try:
            resp = generate_response(model, tokenizer, q)
            resp["time_seconds"] = round(time.time() - t0, 1)
            resp["error"] = None
        except Exception as e:
            import traceback
            traceback.print_exc()
            resp = {
                "think": "", "answer": "", "error": str(e),
                "think_chars": 0, "answer_chars": 0,
                "think_tokens": 0, "answer_tokens": 0,
                "total_output_tokens": 0,
                "time_seconds": round(time.time() - t0, 1),
            }

        results["questions"].append({"question": q, **resp})
        print(f"  Think: {resp['think_tokens']} tokens ({resp['think_chars']} chars)")
        print(f"  Answer: {resp['answer_tokens']} tokens ({resp['answer_chars']} chars)")
        print(f"  Time: {resp['time_seconds']}s")

    # 汇总统计
    total_think_tokens = sum(r["think_tokens"] for r in results["questions"])
    total_answer_tokens = sum(r["answer_tokens"] for r in results["questions"])
    total_think_chars = sum(r["think_chars"] for r in results["questions"])
    results["summary"] = {
        "total_think_tokens": total_think_tokens,
        "total_answer_tokens": total_answer_tokens,
        "total_think_chars": total_think_chars,
        "avg_think_tokens": round(total_think_tokens / len(QUESTIONS), 1),
        "avg_answer_tokens": round(total_answer_tokens / len(QUESTIONS), 1),
        "avg_think_chars": round(total_think_chars / len(QUESTIONS), 1),
    }

    out_file = os.path.join(args.output_dir, f"{args.label.replace(' ', '_').replace('+', '_')}.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n结果已保存: {out_file}")
    print(f"汇总: think {total_think_tokens} tokens, answer {total_answer_tokens} tokens")

    del model
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
