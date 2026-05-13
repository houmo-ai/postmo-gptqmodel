"""Qwen3.5-MoE 聊天验证脚本。

用于快速验证量化后的 Qwen3.5-35B-A3B MoE 模型是否能正常加载、路由并生成。
相比 PPL 评测，这个脚本只做小 batch 的 generate，更适合先排查“模型是否可用”。

用法:
  python scripts_qwen35moe/eval/eval_chat.py \
    --model output/Qwen3.5-35B-A3B-mode1-llm-only

  CUDA_VISIBLE_DEVICES=6,7 python scripts_qwen35moe/eval/eval_chat.py \
    --model output/Qwen3.5-35B-A3B-mode1-llm-only \
    --device_map auto \
    --max_new_tokens 128

  python scripts_qwen35moe/eval/eval_chat.py \
    --model output/Qwen3.5-35B-A3B-mode1-llm-only \
    --question "请解释 AutoRound 对 Qwen3.5-MoE experts 是怎么量化的"

  python scripts_qwen35moe/eval/eval_chat.py \
    --model output/Qwen3.5-35B-A3B-mode1-llm-only \
    --interactive
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

_gptq_root = os.path.expanduser("~/gptqmodel")
if os.path.isdir(_gptq_root) and _gptq_root not in sys.path:
    sys.path.insert(0, _gptq_root)

import torch

os.environ["TOKENIZERS_PARALLELISM"] = "false"


DEFAULT_QUESTIONS = [
    "你是谁？请用一句话介绍自己。",
    "中国最长的河流是什么？用三句话概括它的地理意义。",
    "已知 2x + 5 = 15，请给出解题步骤。",
    "用 80 字左右描述春天的景色。",
    "写一个 Python 函数，返回斐波那契数列的第 n 项。",
]


def patch_grouped_mm_for_sm80() -> None:
    try:
        if torch.cuda.is_available():
            cc = torch.cuda.get_device_capability(0)
            if cc[0] < 9:
                import transformers.integrations.moe as _moe

                _moe._can_use_grouped_mm = lambda *a, **kw: False
    except Exception:
        pass


def render_chat_prompt(tokenizer, messages: list[dict]) -> str:
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )


def get_input_device(model: torch.nn.Module) -> torch.device:
    for _, param in model.named_parameters():
        if param.device.type != "meta":
            return param.device
    for _, buffer in model.named_buffers():
        if buffer.device.type != "meta":
            return buffer.device
    return torch.device("cpu")


def load_model(model_path: str, device_map: str, dtype: str | None):
    from transformers import AutoModelForCausalLM, AutoModelForImageTextToText

    config_path = os.path.join(model_path, "config.json")
    with open(config_path, encoding="utf-8") as file:
        cfg = json.load(file)

    archs = cfg.get("architectures", [])
    is_quantized = "quantization_config" in cfg
    load_kwargs = {
        "device_map": device_map,
        "trust_remote_code": True,
    }
    if dtype:
        load_kwargs["torch_dtype"] = getattr(torch, dtype)

    model = None
    source = None

    if is_quantized:
        try:
            from gptqmodel import BACKEND, GPTQModel

            gptq_model = GPTQModel.load(
                model_path,
                device_map=device_map,
                backend=BACKEND.TORCH,
                trust_remote_code=True,
            )
            model = gptq_model.model
            source = "GPTQModel.load(TORCH)"
        except Exception as exc:
            print(f"GPTQModel.load failed: {exc}")

    if model is None and "Qwen3_5MoeForConditionalGeneration" in archs:
        try:
            model = AutoModelForImageTextToText.from_pretrained(model_path, **load_kwargs)
            source = "AutoModelForImageTextToText"
        except Exception:
            from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeForConditionalGeneration

            model = Qwen3_5MoeForConditionalGeneration.from_pretrained(model_path, **load_kwargs)
            source = "Qwen3_5MoeForConditionalGeneration"

    if model is None and "Qwen3_5MoeForCausalLM" in archs:
        from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeForCausalLM

        model = Qwen3_5MoeForCausalLM.from_pretrained(model_path, **load_kwargs)
        source = "Qwen3_5MoeForCausalLM"

    if model is None and "Qwen3_5ForConditionalGeneration" in archs:
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForConditionalGeneration

        model = Qwen3_5ForConditionalGeneration.from_pretrained(model_path, **load_kwargs)
        source = "Qwen3_5ForConditionalGeneration"

    if model is None:
        model = AutoModelForCausalLM.from_pretrained(model_path, **load_kwargs)
        source = "AutoModelForCausalLM"

    model.eval()
    return model, source, is_quantized


@torch.inference_mode()
def generate_one(
    model,
    tokenizer,
    question: str,
    system_prompt: str | None,
    max_input_tokens: int,
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
    top_p: float,
) -> str:
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": question})

    prompt = render_chat_prompt(tokenizer, messages)
    input_device = get_input_device(model)
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=max_input_tokens,
    )
    inputs = {key: value.to(input_device) for key, value in inputs.items()}

    generate_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "pad_token_id": tokenizer.eos_token_id,
    }
    if do_sample:
        generate_kwargs["temperature"] = temperature
        generate_kwargs["top_p"] = top_p

    output = model.generate(**inputs, **generate_kwargs)
    answer_ids = output[0][inputs["input_ids"].shape[1] :]
    return tokenizer.decode(answer_ids, skip_special_tokens=True).strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Qwen3.5-MoE 聊天验证")
    parser.add_argument("--model", type=str, required=True, help="模型路径")
    parser.add_argument("--device_map", type=str, default="auto", help='如 auto、balanced、cuda:0，默认 auto')
    parser.add_argument("--dtype", type=str, default=None, help="可选：bfloat16 / float16，仅非量化模型常用")
    parser.add_argument("--max_input_tokens", type=int, default=1024)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--system", type=str, default=None, help="可选系统提示词")
    parser.add_argument("--question", action="append", default=None, help="可重复指定，未指定则使用内置问题")
    parser.add_argument("--interactive", action="store_true", help="进入交互模式")
    parser.add_argument("--sample", action="store_true", help="是否采样，默认贪心解码")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    patch_grouped_mm_for_sm80()

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model, source, is_quantized = load_model(args.model, args.device_map, args.dtype)
    print(f"Loaded model from: {source}")
    print(f"Quantized: {is_quantized}")
    print(f"Input device: {get_input_device(model)}")

    if args.interactive:
        print("进入交互模式，输入 exit 或 quit 退出。")
        while True:
            question = input("\nUser> ").strip()
            if question.lower() in {"exit", "quit"}:
                break
            if not question:
                continue
            start_time = time.time()
            answer = generate_one(
                model,
                tokenizer,
                question,
                args.system,
                args.max_input_tokens,
                args.max_new_tokens,
                args.sample,
                args.temperature,
                args.top_p,
            )
            elapsed = time.time() - start_time
            print(f"Assistant> {answer}")
            print(f"[time={elapsed:.2f}s]")
        return

    questions = args.question or DEFAULT_QUESTIONS
    for index, question in enumerate(questions, start=1):
        print(f"\n{'=' * 80}")
        print(f"[{index}] Q: {question}")
        start_time = time.time()
        answer = generate_one(
            model,
            tokenizer,
            question,
            args.system,
            args.max_input_tokens,
            args.max_new_tokens,
            args.sample,
            args.temperature,
            args.top_p,
        )
        elapsed = time.time() - start_time
        print(f"A: {answer}")
        print(f"[time={elapsed:.2f}s]")


if __name__ == "__main__":
    main()