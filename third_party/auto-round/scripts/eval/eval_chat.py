"""简单对话评测脚本：评测 BF16 / 量化模型的生成质量。

用法:
  # BF16 原始模型
  python eval_chat.py --model weights/Qwen3.5-9B

  # W4A16 量化模型 (Qwen3.5 是 VLM 架构，需加 --quantized)
  python eval_chat.py --model weights/Qwen3.5-9B-W4A16-autoround/Qwen3.5-9B-w4g128 --quantized
"""
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

QUESTIONS = [
    "你是谁？"
    "中国最长的河流是什么？请简要介绍。",
    "请解方程 2x + 5 = 15",
    "用100字描述春天的景色",
    "写一个Python函数计算斐波那契数列的第n项",
]


def load_model(model_path, device, quantized):
    """加载模型。量化的 Qwen3.5 VLM 模型需要用原始架构类加载。"""
    if quantized:
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForConditionalGeneration
        return Qwen3_5ForConditionalGeneration.from_pretrained(
            model_path, device_map=device, trust_remote_code=True
        )
    return AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, device_map=device, trust_remote_code=True
    )


def evaluate_chat(model_path, device="cuda:0", quantized=False):
    print(f"\n{'='*60}")
    print(f"模型: {model_path}")
    print(f"{'='*60}")

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = load_model(model_path, device, quantized)

    for q in QUESTIONS:
        messages = [{"role": "user", "content": q}]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        output = model.generate(**inputs, max_new_tokens=512, do_sample=False)
        response = tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        print(f"\nQ: {q}")
        print(f"A: {response}")
        print("---")

    del model
    torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Qwen3.5 对话质量评测")
    parser.add_argument("--model", type=str, required=True, help="模型路径")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--quantized", action="store_true",
                        help="量化模型需要此标志 (Qwen3.5 VLM 架构)")
    args = parser.parse_args()
    evaluate_chat(args.model, args.device, args.quantized)
