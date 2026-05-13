"""Interactive chat CLI for local Qwen3.5-MoE checkpoints."""

import argparse
import json
import os
import sys

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

try:
    from auto_round.inference.backend import BackendInfos

    for backend_name in [name for name in list(BackendInfos.keys()) if "marlin" in name or "exllama" in name]:
        del BackendInfos[backend_name]
except Exception:
    pass

_gptq_root = os.path.expanduser("~/gptqmodel")
if os.path.isdir(_gptq_root) and _gptq_root not in sys.path:
    sys.path.insert(0, _gptq_root)

import torch


DEFAULT_MODEL = "./output/qwen35moe-no-rotate"


def _patch_grouped_mm_for_sm80() -> None:
    try:
        if torch.cuda.is_available():
            cc = torch.cuda.get_device_capability(0)
            if cc[0] < 9:
                import transformers.integrations.moe as _moe

                _moe._can_use_grouped_mm = lambda *args, **kwargs: False
    except Exception:
        pass


_patch_grouped_mm_for_sm80()


def parse_args():
    parser = argparse.ArgumentParser(description="Interactive chat for local BF16 or quantized Qwen3.5-MoE checkpoints")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Checkpoint path")
    parser.add_argument("--device-map", default="auto", help="transformers/GPTQModel device_map")
    parser.add_argument("--dtype", default="bfloat16", help="dtype for non-quantized models")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--sample", action="store_true", help="Enable sampling")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--enable-thinking", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--system-prompt", default=None, help="Optional system prompt")
    return parser.parse_args()


def is_quantized_checkpoint(model_path: str) -> bool:
    config_path = os.path.join(model_path, "config.json")
    with open(config_path, encoding="utf-8") as f:
        cfg = json.load(f)
    return "quantization_config" in cfg


def resolve_dtype(dtype_name: str | None):
    if dtype_name is None:
        return None
    if hasattr(torch, dtype_name):
        return getattr(torch, dtype_name)
    raise ValueError(f"Unsupported dtype: {dtype_name}")


def load_model_and_tokenizer(model_path: str, device_map="auto", dtype_name="bfloat16"):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token

    if is_quantized_checkpoint(model_path):
        from gptqmodel import BACKEND, GPTQModel

        gptq_model = GPTQModel.load(
            model_path,
            device_map=device_map,
            backend=BACKEND.TORCH,
            trust_remote_code=True,
        )
        return gptq_model.model, tokenizer

    from transformers import AutoModelForCausalLM, AutoModelForImageTextToText

    load_kwargs = dict(
        device_map=device_map,
        trust_remote_code=True,
    )
    dtype = resolve_dtype(dtype_name)
    if dtype is not None:
        load_kwargs["torch_dtype"] = dtype
    try:
        model = AutoModelForImageTextToText.from_pretrained(model_path, **load_kwargs)
    except Exception:
        model = AutoModelForCausalLM.from_pretrained(model_path, **load_kwargs)
    return model, tokenizer


def render_prompt(tokenizer, messages, enable_thinking=False) -> str:
    kwargs = dict(tokenize=False, add_generation_prompt=True)
    try:
        return tokenizer.apply_chat_template(messages, enable_thinking=enable_thinking, **kwargs)
    except TypeError:
        return tokenizer.apply_chat_template(messages, **kwargs)


def get_input_device(model) -> torch.device:
    hf_device_map = getattr(model, "hf_device_map", None)
    if isinstance(hf_device_map, dict):
        for device in hf_device_map.values():
            if isinstance(device, str) and device not in ("cpu", "disk"):
                return torch.device(device)
    model_device = getattr(model, "device", None)
    if model_device is not None:
        return torch.device(model_device)
    return next(model.parameters()).device


def generate_reply(
    model,
    tokenizer,
    messages,
    *,
    max_new_tokens,
    do_sample,
    temperature,
    top_p,
    enable_thinking,
):
    prompt = render_prompt(tokenizer, messages, enable_thinking=enable_thinking)
    encoded = tokenizer(prompt, return_tensors="pt")
    input_device = get_input_device(model)
    encoded = {name: value.to(input_device) for name, value in encoded.items()}

    generation_kwargs = dict(
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        pad_token_id=tokenizer.pad_token_id,
    )
    if tokenizer.eos_token_id is not None:
        generation_kwargs["eos_token_id"] = tokenizer.eos_token_id
    if do_sample:
        generation_kwargs["temperature"] = temperature
        generation_kwargs["top_p"] = top_p

    with torch.inference_mode():
        outputs = model.generate(**encoded, **generation_kwargs)

    generated_tokens = outputs[0][encoded["input_ids"].shape[-1]:]
    return tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )


def run_repl(
    model,
    tokenizer,
    *,
    system_prompt,
    max_new_tokens,
    do_sample,
    temperature,
    top_p,
    enable_thinking,
    input_fn=input,
    output_fn=print,
    reply_fn=generate_reply,
):
    output_fn("Commands: /clear resets history, /exit quits.")
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    while True:
        try:
            user_text = input_fn("You> ").strip()
        except EOFError:
            output_fn("")
            break

        if not user_text:
            continue
        if user_text in {"/exit", "/quit"}:
            break
        if user_text == "/clear":
            messages = [{"role": "system", "content": system_prompt}] if system_prompt else []
            output_fn("History cleared.")
            continue
        if user_text == "/help":
            output_fn("Commands: /clear resets history, /exit quits.")
            continue

        messages.append({"role": "user", "content": user_text})
        reply = reply_fn(
            model,
            tokenizer,
            messages,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            enable_thinking=enable_thinking,
        )
        messages.append({"role": "assistant", "content": reply})
        output_fn(f"Assistant> {reply}")


def main():
    args = parse_args()
    model, tokenizer = load_model_and_tokenizer(args.model, device_map=args.device_map, dtype_name=args.dtype)
    model.eval()
    run_repl(
        model,
        tokenizer,
        system_prompt=args.system_prompt,
        max_new_tokens=args.max_new_tokens,
        do_sample=args.sample,
        temperature=args.temperature,
        top_p=args.top_p,
        enable_thinking=args.enable_thinking,
    )


if __name__ == "__main__":
    main()
