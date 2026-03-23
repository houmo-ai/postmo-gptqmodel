"""Quick EvalScope benchmark runner for Qwen3.5 text models.

目标：先用“代表性 + 速度快”的子集进行平台统一与快速回归。

示例：
    CUDA_VISIBLE_DEVICES=0 python examples/qwen3.5/qwen3_5_evalscope_quick_eval.py \
        --model-path weights/Qwen3.5-9B \
        --limit 50

    CUDA_VISIBLE_DEVICES=0 python examples/qwen3.5/qwen3_5_evalscope_quick_eval.py \
        --model-path Qwen/Qwen3.5-9B \
        --datasets arc hellaswag winogrande gsm8k cmmlu \
        --limit 100
"""

import argparse
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch
from evalscope import TaskConfig, run_task
from evalscope.api.messages import ChatMessage
from evalscope.api.model import GenerateConfig, ModelAPI, ModelOutput
from evalscope.api.registry import register_model_api
from evalscope.api.tool import ToolChoice, ToolInfo
from gptqmodel import BACKEND, GPTQModel
from transformers import AutoTokenizer


DEFAULT_DATASETS = ["arc", "hellaswag", "winogrande", "gsm8k", "cmmlu"]
DEFAULT_CMMLU_SUBSETS = [
    "college_actuarial_science",
    "college_medicine",
    "college_law",
    "machine_learning",
]
# DEFAULT_CEVAL_SUBSETS = [
#     "computer_network",
#     "operating_system",
#     "computer_architecture",
#     "college_programming",
# ]

FORCE_DISABLE_THINKING = True


def _resolve_backend(name: str) -> BACKEND:
    try:
        return BACKEND(name.lower())
    except ValueError as exc:
        valid = [member.value for member in BACKEND]
        raise ValueError(f"Unsupported backend: {name}. Available backends: {valid}") from exc


def _resolve_model_device(model: Any) -> Union[str, torch.device]:
    if hasattr(model, "device") and getattr(model, "device") is not None:
        return model.device

    inner_model = getattr(model, "model", None)
    if inner_model is not None:
        if hasattr(inner_model, "device") and getattr(inner_model, "device") is not None:
            return inner_model.device
        if hasattr(inner_model, "hf_device_map"):
            for mapped_device in inner_model.hf_device_map.values():
                if mapped_device not in {"cpu", "disk", None}:
                    return mapped_device
        try:
            return next(inner_model.parameters()).device
        except (AttributeError, StopIteration, TypeError):
            pass

    return "cuda:0" if torch.cuda.is_available() else "cpu"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Qwen3.5-9B quick benchmark with EvalScope",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model-path", type=str, required=True, help="GPTQModel quantized model local path or HF model id")
    parser.add_argument(
        "--datasets",
        type=str,
        nargs="+",
        default=DEFAULT_DATASETS,
        help="Datasets to evaluate",
    )
    parser.add_argument("--limit", type=int, default=50, help="Max samples per dataset")
    parser.add_argument("--max-tokens", type=int, default=1024, help="Default max_new_tokens")
    parser.add_argument("--work-dir", type=str, default="./outputs-evalscope", help="Eval output directory")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-batch-size", type=int, default=1)
    parser.add_argument("--timeout", type=int, default=12000, help="Timeout seconds per dataset")
    parser.add_argument("--enable-thinking", action="store_true", help="(deprecated) Ignored during eval; thinking is forced off")
    parser.add_argument("--device", type=str, default=None, help="Device for GPTQModel.load, e.g. cuda:0 or cpu")
    parser.add_argument(
        "--device-map",
        type=str,
        default=None,
        help="Optional device_map passed to GPTQModel.load, e.g. auto",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="auto",
        help="GPTQModel backend, e.g. auto/torch/triton/cuda/marlin/exllama_v2",
    )
    parser.add_argument(
        "--trust-remote-code",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable trust_remote_code when loading tokenizer and model",
    )
    parser.add_argument(
        "--use-fast-tokenizer",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use fast tokenizer when loading tokenizer from model path",
    )
    parser.add_argument(
        "--ceval-subsets",
        type=str,
        nargs="+",
        default=None,
        help="Optional subset list for ceval; if omitted, evaluate all ceval subsets",
    )
    return parser.parse_args()


def _render_chat_prompt(tokenizer, messages: List[Dict[str, str]], enable_thinking: bool = False) -> str:
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=enable_thinking,
            )
        except TypeError:
            return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return "\n".join([f"{m['role']}: {m['content']}" for m in messages] + ["assistant:"])


def _strip_think_content(text: str) -> str:
    if text is None:
        return ""
    cleaned = text.replace("<|im_end|>", "").strip()
    if "</think>" in cleaned:
        cleaned = cleaned.split("</think>", 1)[1].strip()
    cleaned = re.sub(r"<think>[\s\S]*?</think>", "", cleaned, flags=re.IGNORECASE).strip()
    return cleaned


def _extract_single_choice_letter(text: str) -> Optional[str]:
    if not text:
        return None
    patterns = [
        r"答案\s*[:：]\s*([A-D])",
        r"(?i)ANSWER\s*[:：]\s*([A-D])",
        r"^[\s\*\-\(\[]*([A-D])[\)\]\s\.:：]*$",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.MULTILINE)
        if match:
            return match.group(1).upper()
    return None


def _normalize_ceval_output(output_text: str, prompt_text: str) -> str:
    if not output_text:
        return output_text
    if "单项选择题" not in prompt_text and "中国关于" not in prompt_text:
        return output_text
    if re.search(r"答案：[A-D]", output_text):
        return output_text
    letter = _extract_single_choice_letter(output_text)
    if letter is None:
        return output_text
    return f"答案：{letter}"


@register_model_api(name="Qwen35EvalModel")
class Qwen35EvalModel(ModelAPI):
    def __init__(
        self,
        model_name: str,
        model: Any,
        tokenizer: Any,
        config: GenerateConfig,
        **kwargs: Dict[str, Any],
    ) -> None:
        self.model_name = model_name
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.device = _resolve_model_device(model)

    def generate(
        self,
        input: List[ChatMessage],
        tools: List[ToolInfo],
        tool_choice: ToolChoice,
        config: GenerateConfig,
    ) -> ModelOutput:
        messages = [{"role": message.role, "content": message.content} for message in input]
        prompt = _render_chat_prompt(
            self.tokenizer,
            messages,
            enable_thinking=not FORCE_DISABLE_THINKING,
        )

        model_inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        generation_kwargs = dict(
            **model_inputs,
            max_new_tokens=getattr(config, "max_tokens", getattr(self.config, "max_tokens", 1024)),
            do_sample=getattr(config, "do_sample", getattr(self.config, "do_sample", False)),
            temperature=getattr(config, "temperature", getattr(self.config, "temperature", 0.0)),
            pad_token_id=self.tokenizer.eos_token_id,
        )
        generated = self.model.generate(**generation_kwargs)[0]
        prompt_len = model_inputs["input_ids"].shape[1]
        output_text = self.tokenizer.decode(generated[prompt_len:], skip_special_tokens=True)
        output_text = _strip_think_content(output_text)
        output_text = _normalize_ceval_output(output_text, prompt)
        return ModelOutput.from_content(model=self.model_name, content=output_text)


def dataset_args_for(datasets: List[str], ceval_subsets: Optional[List[str]]) -> Dict[str, Any]:
    args: Dict[str, Any] = {}
    if "arc" in datasets:
        args["arc"] = {"subset_list": ["ARC-Challenge"]}
    if "cmmlu" in datasets:
        args["cmmlu"] = {"subset_list": DEFAULT_CMMLU_SUBSETS}
    if "ceval" in datasets and ceval_subsets:
        args["ceval"] = {"subset_list": ceval_subsets}
    return args


def max_tokens_for(dataset: str, default_max_tokens: int) -> int:
    if dataset in {"gsm8k", "math_500"}:
        return max(default_max_tokens, 1024)
    if dataset in {"arc", "hellaswag", "winogrande", "cmmlu"}:
        return min(default_max_tokens, 512)
    return default_max_tokens


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    if args.enable_thinking:
        print("[WARN] --enable-thinking is ignored during eval. Thinking remains disabled.", flush=True)

    model_name = Path(args.model_path).name if "/" in args.model_path else args.model_path
    load_kwargs: Dict[str, Any] = {
        "backend": _resolve_backend(args.backend),
        "trust_remote_code": args.trust_remote_code,
        "offload_to_disk": False,
    }
    if args.device_map:
        load_kwargs["device_map"] = args.device_map
    elif args.device:
        load_kwargs["device"] = args.device

    model = GPTQModel.load(
        args.model_path,
        **load_kwargs,
    )
    model.eval()

    tokenizer = getattr(model, "tokenizer", None)
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_path,
            use_fast=args.use_fast_tokenizer,
            trust_remote_code=args.trust_remote_code,
        )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    all_dataset_args = dataset_args_for(args.datasets, args.ceval_subsets)

    for dataset in args.datasets:
        print(f"===== EvalScope dataset: {dataset} =====", flush=True)
        generate_cfg = GenerateConfig(
            max_tokens=max_tokens_for(dataset, args.max_tokens),
            temperature=0.0,
            do_sample=False,
            enable_thinking=not FORCE_DISABLE_THINKING,
        )
        evalscope_model = Qwen35EvalModel(
            model_name=model_name,
            model=model,
            tokenizer=tokenizer,
            config=generate_cfg,
        )

        task_cfg_kwargs: Dict[str, Any] = {
            "model": evalscope_model,
            "datasets": [dataset],
            "work_dir": f"{args.work_dir}/{model_name}",
            "timeout": args.timeout,
            "eval_batch_size": args.eval_batch_size,
            "seed": args.seed,
            "limit": args.limit,
        }
        if dataset in all_dataset_args:
            task_cfg_kwargs["dataset_args"] = {dataset: all_dataset_args[dataset]}

        task_cfg = TaskConfig(**task_cfg_kwargs)
        run_task(task_cfg=task_cfg)


if __name__ == "__main__":
    main()
