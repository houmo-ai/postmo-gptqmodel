import argparse
import json
import os
import re
import sys
from pathlib import Path
from types import MethodType
from typing import Any

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import torch


def _patch_grouped_mm_for_sm80() -> None:
    try:
        if torch.cuda.is_available():
            cc = torch.cuda.get_device_capability(0)
            if cc[0] < 9:
                import transformers.integrations.moe as _moe

                _moe._can_use_grouped_mm = lambda *a, **kw: False
    except Exception:
        pass


_patch_grouped_mm_for_sm80()

from evalscope.api.messages import ChatMessage, ChatMessageAssistant, ChatMessageUser
from evalscope.api.model import ChatCompletionChoice, GenerateConfig, ModelAPI, ModelOutput, ModelUsage
from evalscope.config import TaskConfig
from evalscope.run import run_task

from scripts_gemma4.eval.common import ensure_local_gptqmodel_path, patch_gptqmodel_for_gemma4


DATASET_ALIASES = {
    "ceval": "ceval",
    "humaneval": "humaneval",
    "mmlu_pro": "mmlu_pro",
    "mmlu-pro": "mmlu_pro",
}


def _get_multiple_choice_answer_tag(prompt: str) -> str | None:
    if "答案：[LETTER]" in prompt or "答案:[LETTER]" in prompt:
        return "答案"
    if "ANSWER: [LETTER]" in prompt:
        return "ANSWER"
    return None


def _format_multiple_choice_followup(option_letters: tuple[str, ...], answer_tag: str) -> str:
    letters = ", ".join(option_letters)
    if answer_tag == "答案":
        return (
            "请只输出最终单项选择答案，格式严格为 "
            f"`答案：[LETTER]`，其中 [LETTER] 只能是 {letters} 之一。不要添加任何其他文字。"
        )
    return (
        "Please provide only the final multiple-choice answer in the exact format "
        f"`ANSWER: [LETTER]` where [LETTER] is one of {letters}. Do not add any other text."
    )


def _format_multiple_choice_answer(answer_tag: str, answer: str) -> str:
    if answer_tag == "答案":
        return f"答案：{answer}"
    return f"ANSWER: {answer}"


def _resolve_dtype(dtype_name: str | None):
    if dtype_name is None:
        return None
    if isinstance(dtype_name, str) and hasattr(torch, dtype_name):
        return getattr(torch, dtype_name)
    raise ValueError(f"Unsupported dtype: {dtype_name}")


def _detect_quantized(model_path: str) -> bool:
    with open(os.path.join(model_path, "config.json"), encoding="utf-8") as f:
        cfg = json.load(f)
    return "quantization_config" in cfg


def _get_primary_device(model) -> torch.device:
    hf_device_map = getattr(model, "hf_device_map", None)
    if isinstance(hf_device_map, dict):
        for device in hf_device_map.values():
            if isinstance(device, str) and device not in ("cpu", "disk"):
                return torch.device(device)
    model_device = getattr(model, "device", None)
    if model_device is not None:
        return torch.device(model_device)
    return next(model.parameters()).device


def _disable_gptqmodel_torch_compile(model) -> int:
    changed = 0
    for module in model.modules():
        module_cls = type(module)
        if getattr(module_cls, "__module__", "") != "gptqmodel.nn_modules.qlinear.torch":
            continue
        class_method = getattr(module_cls, "dequantize_weight", None)
        if class_method is None:
            continue
        module.dequantize_weight = MethodType(class_method, module)
        changed += 1
    return changed


def _message_content_to_string(messages: list[ChatMessage]) -> list[dict[str, str]]:
    converted = []
    for message in messages:
        content = message.content
        if isinstance(content, list):
            text_parts = []
            for item in content:
                text = getattr(item, "text", None)
                if text:
                    text_parts.append(text)
            content = "".join(text_parts)
        converted.append({"role": message.role, "content": str(content)})
    return converted


def _get_message_text(message: ChatMessage) -> str:
    content = getattr(message, "content", "")
    if isinstance(content, str):
        return content
    text = getattr(message, "text", None)
    if isinstance(text, str):
        return text
    return str(content)


def _build_generation_kwargs(config: GenerateConfig, tokenizer) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "return_dict_in_generate": True,
        "pad_token_id": tokenizer.pad_token_id,
    }
    if tokenizer.eos_token_id is not None:
        kwargs["eos_token_id"] = tokenizer.eos_token_id
    if config.max_tokens is not None:
        kwargs["max_new_tokens"] = config.max_tokens
    do_sample = bool(config.do_sample)
    kwargs["do_sample"] = do_sample
    if do_sample:
        if config.temperature is not None:
            kwargs["temperature"] = config.temperature
        if config.top_p is not None:
            kwargs["top_p"] = config.top_p
        if config.top_k is not None:
            kwargs["top_k"] = config.top_k
    return kwargs


def _get_multiple_choice_option_letters(prompt: str) -> tuple[str, ...]:
    matches = re.findall(r"(?im)^\s*([A-Z])\s*[\)\].:-]", prompt)
    option_letters = []
    for match in matches:
        letter = match.upper()
        if letter not in option_letters:
            option_letters.append(letter)
    return tuple(option_letters)


def _is_multiple_choice_answer_prompt(prompt: str) -> bool:
    return _get_multiple_choice_answer_tag(prompt) is not None and len(_get_multiple_choice_option_letters(prompt)) >= 4


def _has_explicit_multiple_choice_answer(text: str) -> bool:
    return re.search(r"(?im)^\s*(?:ANSWER|答案)\s*[:：]\s*[\[\(]?\s*([A-Z])\s*[\]\)]?\s*$", text) is not None


def _extract_multiple_choice_answer(text: str, allowed_letters: tuple[str, ...] | None = None) -> str | None:
    allowed = set(allowed_letters) if allowed_letters else None
    explicit_match = re.search(r"(?im)^\s*(?:ANSWER|答案)\s*[:：]\s*[\[\(]?\s*([A-Z])\s*[\]\)]?\s*$", text)
    if explicit_match is not None:
        answer = explicit_match.group(1).upper()
        return answer if allowed is None or answer in allowed else None
    choice_match = re.search(r"(?im)^\s*([A-Z])\s*[\)\].:-]", text)
    if choice_match is not None:
        answer = choice_match.group(1).upper()
        return answer if allowed is None or answer in allowed else None
    bare_match = re.search(r"(?im)^\s*([A-Z])\s*$", text)
    if bare_match is not None:
        answer = bare_match.group(1).upper()
        return answer if allowed is None or answer in allowed else None
    return None


class Gemma4EvalScopeAPI(ModelAPI):
    def __init__(
        self,
        model_name: str,
        base_url: str | None = None,
        api_key: str | None = None,
        config: GenerateConfig = GenerateConfig(),
        **model_args: Any,
    ) -> None:
        super().__init__(model_name=model_name, base_url=base_url, api_key=api_key, config=config)
        self.model_path = model_args.pop("model_path", model_name)
        self.device_map = model_args.pop("device_map", "auto")
        self.dtype = _resolve_dtype(model_args.pop("precision", None))
        self.enable_thinking = model_args.pop("enable_thinking", None)
        self.tokenizer_call_args = model_args.pop("tokenizer_call_args", {}) or {}
        self.model, self.tokenizer = self._load_model_and_tokenizer(self.model_path, model_args)
        self.input_device = _get_primary_device(self.model)

    def _load_model_and_tokenizer(self, model_path: str, extra_model_args: dict[str, Any]):
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
        tokenizer.padding_side = "left"

        if _detect_quantized(model_path):
            ensure_local_gptqmodel_path()
            patch_gptqmodel_for_gemma4()
            from gptqmodel import BACKEND, GPTQModel

            gptq_model = GPTQModel.load(
                model_path,
                device_map=self.device_map,
                backend=BACKEND.TORCH,
                trust_remote_code=True,
                **extra_model_args,
            )
            model = gptq_model.model
            _disable_gptqmodel_torch_compile(model)
        else:
            from transformers import AutoModelForCausalLM, AutoModelForImageTextToText

            load_kwargs = dict(device_map=self.device_map, trust_remote_code=True, **extra_model_args)
            if self.dtype is not None:
                load_kwargs["torch_dtype"] = self.dtype
            try:
                model = AutoModelForImageTextToText.from_pretrained(model_path, **load_kwargs)
            except Exception:
                model = AutoModelForCausalLM.from_pretrained(model_path, **load_kwargs)

        model.eval()
        return model, tokenizer

    def _render_chat(self, messages: list[ChatMessage]) -> str:
        rendered_messages = _message_content_to_string(messages)
        if self.tokenizer.chat_template is not None:
            template_kwargs = {
                "add_generation_prompt": True,
                "tokenize": False,
            }
            if self.enable_thinking is not None:
                template_kwargs["enable_thinking"] = self.enable_thinking
            return self.tokenizer.apply_chat_template(rendered_messages, **template_kwargs)
        return "".join(f"{m['role']}: {m['content']}\n" for m in rendered_messages)

    def _generate_prompt_texts(self, prompts: list[str], config: GenerateConfig):
        encoded = self.tokenizer(prompts, return_tensors="pt", padding=True, **self.tokenizer_call_args)
        encoded = {k: v.to(self.input_device) for k, v in encoded.items()}
        kwargs = _build_generation_kwargs(config, self.tokenizer)

        with torch.inference_mode():
            generated = self.model.generate(**encoded, **kwargs)

        sequences = generated.sequences if hasattr(generated, "sequences") else generated
        prompt_width = encoded["input_ids"].shape[1]
        input_lengths = encoded["attention_mask"].sum(dim=1).tolist()
        outputs = []
        for idx, _prompt in enumerate(prompts):
            generated_tokens = sequences[idx, prompt_width:]
            text = self.tokenizer.decode(
                generated_tokens,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            outputs.append((text, int(input_lengths[idx]), int(generated_tokens.shape[-1])))
        return outputs

    def _generate_prefilled_assistant_text(
        self,
        messages: list[ChatMessage],
        assistant_prefix: str,
        config: GenerateConfig,
    ):
        rendered_messages = _message_content_to_string(messages)
        if self.tokenizer.chat_template is not None:
            template_kwargs = {
                "add_generation_prompt": False,
                "tokenize": False,
            }
            if self.enable_thinking is not None:
                template_kwargs["enable_thinking"] = self.enable_thinking
            prompt = self.tokenizer.apply_chat_template(
                [
                    *rendered_messages,
                    {"role": "assistant", "content": assistant_prefix},
                ],
                **template_kwargs,
            )
        else:
            prompt = self._render_chat(messages) + assistant_prefix
        return self._generate_prompt_texts([prompt], config)[0]

    def _finalize_multiple_choice_answer(
        self,
        messages: list[ChatMessage],
        prompt: str,
        text: str,
        config: GenerateConfig,
    ) -> str:
        option_letters = _get_multiple_choice_option_letters(prompt)
        answer_tag = _get_multiple_choice_answer_tag(prompt)
        if len(option_letters) < 4 or answer_tag is None:
            return text
        if _has_explicit_multiple_choice_answer(text) and _extract_multiple_choice_answer(text, option_letters) is not None:
            return text
        followup_messages = [
            *messages,
            ChatMessageAssistant(content=text, model=self.model_name, source="generate"),
            ChatMessageUser(content=_format_multiple_choice_followup(option_letters, answer_tag), source="input"),
        ]
        followup_prompt = self._render_chat(followup_messages)
        followup_config = GenerateConfig(max_tokens=32, do_sample=False, top_p=None, top_k=None)
        followup_text, _, _ = self._generate_prompt_texts([followup_prompt], followup_config)[0]
        followup_answer = _extract_multiple_choice_answer(followup_text, option_letters)
        if followup_answer is not None:
            return _format_multiple_choice_answer(answer_tag, followup_answer)

        user_prompt = ""
        for message in reversed(messages):
            if getattr(message, "role", None) == "user":
                user_prompt = _get_message_text(message).strip()
                break
        if user_prompt:
            prefill_messages = [
                ChatMessageUser(
                    content=(
                        f"{user_prompt}\n\n只输出最终答案，格式为 答案：X。"
                        if answer_tag == "答案"
                        else f"{user_prompt}\n\nRespond with only the final answer in the form ANSWER: X."
                    ),
                    source="input",
                ),
            ]
            prefill_config = GenerateConfig(max_tokens=8, do_sample=False, top_p=None, top_k=None)
            prefill_text, _, _ = self._generate_prefilled_assistant_text(
                prefill_messages,
                "答案：" if answer_tag == "答案" else "ANSWER:",
                prefill_config,
            )
            prefill_answer = _extract_multiple_choice_answer(prefill_text, option_letters)
            if prefill_answer is not None:
                return _format_multiple_choice_answer(answer_tag, prefill_answer)
        return text

    def generate(self, input: list[ChatMessage], tools, tool_choice, config: GenerateConfig) -> ModelOutput:
        return next(self.batch_generate([input], [[]], [tool_choice], [config]))

    def batch_generate(self, inputs: list[list[ChatMessage]], tools, tool_choices, configs: list[GenerateConfig]):
        prompts = [self._render_chat(messages) for messages in inputs]
        cfg = configs[0] if configs else self.config
        generated_outputs = self._generate_prompt_texts(prompts, cfg)

        for idx, prompt in enumerate(prompts):
            text, input_tokens, output_tokens = generated_outputs[idx]
            text = self._finalize_multiple_choice_answer(inputs[idx], prompt, text, cfg)
            yield ModelOutput(
                model=self.model_name,
                choices=[
                    ChatCompletionChoice(
                        message=ChatMessageAssistant(content=text, model=self.model_name, source="generate"),
                        stop_reason="stop",
                    )
                ],
                usage=ModelUsage(
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    total_tokens=input_tokens + output_tokens,
                ),
                time=0.0,
            )

    def supports_batch(self) -> bool:
        return True

    def max_tokens(self):
        return 2048

    def max_connections(self) -> int:
        return self.config.batch_size or 1


def _normalize_results(raw_results: dict, requested_datasets: list[str], blocked: dict[str, str]) -> dict:
    normalized = {}
    for requested_name in requested_datasets:
        dataset_name = DATASET_ALIASES[requested_name]
        if requested_name in blocked:
            normalized[requested_name] = {
                "dataset": dataset_name,
                "status": "blocked",
                "reason": blocked[requested_name],
            }
            continue

        payload = raw_results.get(dataset_name, {})
        metrics = {}
        primary_metric = None
        if isinstance(payload, dict):
            if isinstance(payload.get("score"), (int, float)):
                metrics["score"] = payload["score"]
            metric_list = payload.get("metrics") or []
            if metric_list:
                first_metric = metric_list[0]
                if isinstance(first_metric, dict):
                    metric_name = first_metric.get("name")
                    metric_score = first_metric.get("score")
                    if metric_name is not None and isinstance(metric_score, (int, float)):
                        metrics[str(metric_name)] = metric_score
                        primary_metric = str(metric_name)
        normalized[requested_name] = {
            "dataset": dataset_name,
            "status": "ok",
            "primary_metric": primary_metric,
            "metrics": metrics,
            "raw": payload,
        }
    return normalized


def _load_report_payloads(work_dir: str, model_id: str, requested_datasets: list[str]) -> dict:
    payloads = {}
    for requested_name in requested_datasets:
        dataset_name = DATASET_ALIASES[requested_name]
        report_path = os.path.join(work_dir, "reports", model_id, f"{dataset_name}.json")
        if os.path.exists(report_path):
            with open(report_path, encoding="utf-8") as f:
                payloads[dataset_name] = json.load(f)
    return payloads


def _parse_dataset_subset_overrides(values: list[str] | None) -> dict[str, list[str]]:
    overrides: dict[str, list[str]] = {}
    for value in values or []:
        if "=" in value:
            dataset_key, subset_name = value.split("=", 1)
        elif ":" in value:
            dataset_key, subset_name = value.split(":", 1)
        else:
            raise ValueError(
                f"Invalid dataset subset override '{value}'. Expected format DATASET=SUBSET or DATASET:SUBSET."
            )
        dataset_name = DATASET_ALIASES[dataset_key.strip().lower()]
        overrides.setdefault(dataset_name, []).append(subset_name.strip())
    return overrides


def _build_task_config(args, model_api, model_id: str, resolved_datasets: list[str], dataset_args: dict | None = None):
    generation_config = GenerateConfig(max_tokens=args.max_tokens, do_sample=False, top_p=None, top_k=None)
    cache_resume_dir = args.work_dir if args.use_cache else None
    return TaskConfig(
        model=model_api,
        model_id=model_id,
        datasets=resolved_datasets,
        dataset_args=dataset_args or {},
        work_dir=args.work_dir,
        eval_batch_size=args.eval_batch_size,
        generation_config=generation_config,
        limit=args.limit,
        debug=args.debug,
        use_cache=cache_resume_dir,
        no_timestamp=True,
    )


def _build_model_api_kwargs(args, model_id: str) -> dict[str, Any]:
    return {
        "model_name": model_id,
        "model_path": args.model,
        "device_map": args.device_map,
        "precision": args.dtype,
        "enable_thinking": args.enable_thinking,
        "config": GenerateConfig(max_tokens=args.max_tokens, do_sample=False, top_p=None, top_k=None),
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Run EvalScope on Gemma4 BF16 or quantized checkpoints")
    parser.add_argument("--model", required=True, help="Local BF16 or quantized model path")
    parser.add_argument("--model-id", default=None, help="Display name used in EvalScope outputs")
    parser.add_argument("--datasets", nargs="+", default=["mmlu_pro", "ceval", "humaneval"])
    parser.add_argument("--work-dir", required=True, help="Output directory for EvalScope artifacts")
    parser.add_argument("--eval-batch-size", type=int, default=1)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--dataset-subset",
        action="append",
        default=[],
        help="Limit a dataset to specific subset(s), e.g. --dataset-subset ceval=advanced_mathematics",
    )
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--use-cache", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--enable-thinking", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--allow-unsafe-humaneval", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    requested_datasets = [d.lower() for d in args.datasets]
    subset_overrides = _parse_dataset_subset_overrides(args.dataset_subset)
    resolved_datasets = []
    blocked: dict[str, str] = {}
    for dataset in requested_datasets:
        if dataset not in DATASET_ALIASES:
            raise ValueError(f"Unsupported dataset alias: {dataset}")
        if dataset == "humaneval" and not args.allow_unsafe_humaneval:
            blocked[dataset] = (
                "HumanEval requires executing model-generated code. "
                "This environment has no EvalScope sandbox backend (no docker/ms_enclave), so unsafe execution is disabled."
            )
            continue
        resolved_datasets.append(DATASET_ALIASES[dataset])

    os.makedirs(args.work_dir, exist_ok=True)
    raw_results = {}
    model_id = args.model_id or Path(args.model).name
    if resolved_datasets:
        model_api = Gemma4EvalScopeAPI(**_build_model_api_kwargs(args, model_id))
        dataset_args = {
            dataset_name: {"subset_list": subset_overrides[dataset_name]}
            for dataset_name in resolved_datasets
            if dataset_name in subset_overrides
        }
        task_cfg = _build_task_config(args, model_api, model_id, resolved_datasets, dataset_args=dataset_args)
        run_task(task_cfg)
        raw_results = _load_report_payloads(args.work_dir, model_id, requested_datasets)

    normalized = _normalize_results(raw_results, requested_datasets, blocked)
    with open(os.path.join(args.work_dir, "result_raw.json"), "w", encoding="utf-8") as f:
        json.dump(raw_results, f, indent=2, ensure_ascii=False)
    with open(os.path.join(args.work_dir, "result_normalized.json"), "w", encoding="utf-8") as f:
        json.dump(normalized, f, indent=2, ensure_ascii=False)

    print(json.dumps(normalized, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
