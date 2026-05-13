"""Qwen3.5 对话质量评测脚本 —— 20 个场景 × think/non-think 模式。

为每个模型生成回复，保存 JSON 结果并输出 Markdown 对比报告。

用法:
  # 评测单个模型
  CUDA_VISIBLE_DEVICES=5 python eval_chat_20.py \
    --model weights/Qwen3.5-27B \
    --label "BF16-Original" \
    --device cuda:0 --dtype bfloat16

  # 评测量化模型
  CUDA_VISIBLE_DEVICES=5 python eval_chat_20.py \
    --model weights/Qwen3.5-27B-W4G64-autoround \
    --label "AutoRound-W4G64" \
    --quantized --device cuda:0
"""
import argparse
import json
import os
import time

import torch

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ── 20 个对话场景，覆盖不同主题和难度 ──
SCENARIOS = [
    # --- 基础知识 (4) ---
    {"id": 1, "topic": "常识", "difficulty": "简单",
     "question": "地球到月球的平均距离大约是多少公里？请简要说明。"},
    {"id": 2, "topic": "历史", "difficulty": "简单",
     "question": "简述第二次世界大战的主要起因和结束时间。"},
    {"id": 3, "topic": "地理", "difficulty": "简单",
     "question": "世界上面积最大的五个国家分别是哪些？请按面积从大到小排列。"},
    {"id": 4, "topic": "科学常识", "difficulty": "简单",
     "question": "为什么天空是蓝色的？请用通俗易懂的语言解释。"},

    # --- 数学推理 (4) ---
    {"id": 5, "topic": "算术", "difficulty": "中等",
     "question": "一个商品原价200元，先打八折，再在此基础上打九折，最终售价是多少？"},
    {"id": 6, "topic": "代数", "difficulty": "中等",
     "question": "求方程组的解：2x + 3y = 12, 4x - y = 5"},
    {"id": 7, "topic": "概率", "difficulty": "较难",
     "question": "一个袋子里有3个红球和5个蓝球。不放回地连续抽取2个球，两个都是红球的概率是多少？"},
    {"id": 8, "topic": "数学证明", "difficulty": "较难",
     "question": "请证明√2是无理数。"},

    # --- 编程 (4) ---
    {"id": 9, "topic": "Python基础", "difficulty": "简单",
     "question": "写一个Python函数，输入一个列表，返回其中所有偶数组成的新列表。"},
    {"id": 10, "topic": "算法", "difficulty": "中等",
     "question": "用Python实现二分查找算法，要求处理元素不存在的情况。"},
    {"id": 11, "topic": "数据结构", "difficulty": "较难",
     "question": "用Python实现一个最小堆（Min Heap），支持insert和extract_min操作。"},
    {"id": 12, "topic": "系统设计", "difficulty": "困难",
     "question": "设计一个简单的URL短链接服务，说明核心数据结构、API接口和关键算法。"},

    # --- 逻辑推理 (3) ---
    {"id": 13, "topic": "逻辑", "difficulty": "中等",
     "question": "三个人A、B、C中有一个说真话，两个说假话。A说'我不是说谎者'，B说'A在说谎'，C说'B在说谎'。谁说的是真话？"},
    {"id": 14, "topic": "推理", "difficulty": "较难",
     "question": "有5间房子排成一排，分别漆成红、蓝、绿、黄、白色。已知：绿色房子在白色房子左边紧邻；红色房子在最左边；蓝色房子不在两端。请推断5间房子从左到右的颜色。"},
    {"id": 15, "topic": "批判性思维", "difficulty": "较难",
     "question": "有人说'所有成功的人都很努力，所以只要努力就能成功'。请分析这个论证的逻辑谬误。"},

    # --- 创意写作 (3) ---
    {"id": 16, "topic": "诗歌", "difficulty": "中等",
     "question": "请写一首关于'人工智能与人类未来'的现代诗，要求至少8行。"},
    {"id": 17, "topic": "故事创作", "difficulty": "中等",
     "question": "以'最后一个图书馆'为题，写一个200字左右的微型科幻故事。"},
    {"id": 18, "topic": "文案", "difficulty": "简单",
     "question": "为一款环保水杯写一段100字以内的广告文案，要求有创意且打动人心。"},

    # --- 专业领域 (2) ---
    {"id": 19, "topic": "医学常识", "difficulty": "中等",
     "question": "什么是高血压？长期高血压不治疗会有什么危害？日常生活中如何预防？"},
    {"id": 20, "topic": "经济学", "difficulty": "较难",
     "question": "请解释'流动性陷阱'的概念，并结合实际经济案例说明其影响和应对策略。"},
]


def load_model(model_path, device, quantized, dtype=None):
    """加载模型。"""
    load_kwargs = dict(device_map=device, trust_remote_code=True)
    if dtype:
        load_kwargs["torch_dtype"] = getattr(torch, dtype)

    config_path = os.path.join(model_path, "config.json")
    with open(config_path) as f:
        cfg = json.load(f)
    archs = cfg.get("architectures", [])

    if quantized:
        # Block gptqmodel C++ backends (require GLIBC_2.32, system has 2.31)
        import auto_round.inference.backend as _bk
        for k in [k for k in _bk.BackendInfos if k.startswith("gptqmodel:")]:
            del _bk.BackendInfos[k]

        if "Qwen3_5ForConditionalGeneration" in archs:
            from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForConditionalGeneration
            return Qwen3_5ForConditionalGeneration.from_pretrained(model_path, **load_kwargs)

    from transformers import AutoModelForCausalLM
    if not dtype:
        load_kwargs["torch_dtype"] = torch.bfloat16
    return AutoModelForCausalLM.from_pretrained(model_path, **load_kwargs)


def generate_response(model, tokenizer, question, enable_thinking, max_new_tokens=1024):
    """生成单条回复。"""
    messages = [{"role": "user", "content": question}]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output = model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False,
        )
    full_response = tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=False)

    # 分离 think 和 answer 部分
    think_content = ""
    answer_content = full_response
    if enable_thinking and "<think>" in full_response:
        parts = full_response.split("</think>")
        if len(parts) >= 2:
            think_content = parts[0].replace("<think>", "").strip()
            answer_content = parts[1].strip()
    # 清理 special tokens
    for tok in ["<|im_end|>", "<|endoftext|>", "<|im_start|>"]:
        answer_content = answer_content.replace(tok, "").strip()
        think_content = think_content.replace(tok, "").strip()

    return {
        "think": think_content,
        "answer": answer_content,
        "full_response": full_response,
        "input_tokens": inputs["input_ids"].shape[1],
        "output_tokens": output.shape[1] - inputs["input_ids"].shape[1],
    }


def evaluate_model(model_path, label, device="cuda:0", quantized=False, dtype=None,
                   output_dir="eval_chat_results"):
    """评测单个模型在所有场景上的表现。"""
    from transformers import AutoTokenizer

    os.makedirs(output_dir, exist_ok=True)
    print(f"\n{'='*70}")
    print(f"评测模型: {label} ({model_path})")
    print(f"{'='*70}")

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = load_model(model_path, device, quantized, dtype)

    results = {"label": label, "model_path": model_path, "scenarios": []}

    for scenario in SCENARIOS:
        print(f"\n[{scenario['id']}/20] {scenario['topic']} ({scenario['difficulty']})")
        print(f"Q: {scenario['question'][:60]}...")

        scenario_result = {
            "id": scenario["id"],
            "topic": scenario["topic"],
            "difficulty": scenario["difficulty"],
            "question": scenario["question"],
        }

        for mode_name, enable_think in [("non-think", False), ("think", True)]:
            t0 = time.time()
            try:
                resp = generate_response(model, tokenizer, scenario["question"], enable_think)
                resp["time_seconds"] = round(time.time() - t0, 1)
                resp["error"] = None
            except Exception as e:
                resp = {"answer": "", "think": "", "error": str(e),
                        "time_seconds": round(time.time() - t0, 1)}
            scenario_result[mode_name] = resp
            print(f"  [{mode_name}] {resp.get('output_tokens', 0)} tokens, {resp['time_seconds']}s")

        results["scenarios"].append(scenario_result)

    # 保存 JSON
    out_file = os.path.join(output_dir, f"{label.replace(' ', '_')}.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n结果已保存: {out_file}")

    del model
    torch.cuda.empty_cache()
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Qwen3.5 20场景对话评测")
    parser.add_argument("--model", type=str, required=True, help="模型路径")
    parser.add_argument("--label", type=str, required=True, help="模型标签 (如 BF16-Original)")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--quantized", action="store_true",
                        help="量化模型需要此标志 (Qwen3.5 VLM 架构)")
    parser.add_argument("--dtype", type=str, default=None,
                        help="BF16原始模型评测时使用 bfloat16")
    parser.add_argument("--output_dir", type=str, default="eval_chat_results")
    args = parser.parse_args()

    evaluate_model(args.model, args.label, args.device, args.quantized, args.dtype, args.output_dir)
