"""Load a QuaRot+AutoRound Laguna checkpoint and run a generation check."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def parse_args():
    parser = argparse.ArgumentParser(description="Chat with a quantized Laguna checkpoint")
    parser.add_argument("--model", required=True, help="Quantized checkpoint directory")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument(
        "--backend",
        default="auto_round:tritonv2_zp",
        help="AutoRound inference backend; the default avoids optional Marlin kernels",
    )
    parser.add_argument("--prompt", default="请用一句话介绍你自己。")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def read_and_validate_config(model_path: str) -> dict:
    config_path = Path(model_path) / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Missing checkpoint config: {config_path}")
    with config_path.open(encoding="utf-8") as handle:
        config = json.load(handle)

    if config.get("model_type") != "laguna":
        raise ValueError(f"Expected model_type='laguna', got {config.get('model_type')!r}")
    if not config.get("quantization_config"):
        raise ValueError("Checkpoint does not contain quantization_config")
    quarot_config = config.get("quarot_config")
    if not isinstance(quarot_config, dict) or not quarot_config.get("applied"):
        raise ValueError("Checkpoint does not advertise an applied QuaRot transform")
    return config


def input_device(model) -> torch.device:
    def normalize(value) -> torch.device:
        if isinstance(value, torch.device):
            return value
        if isinstance(value, str):
            return torch.device(value)
        return torch.device(f"cuda:{value}")

    device_map = getattr(model, "hf_device_map", None)
    if isinstance(device_map, dict):
        preferred_keys = ("model.embed_tokens", "model.model.embed_tokens", "")
        for key in preferred_keys:
            value = device_map.get(key)
            if value is not None and value not in ("cpu", "disk"):
                return normalize(value)
        for value in device_map.values():
            if value not in ("cpu", "disk"):
                return normalize(value)
    return next(model.parameters()).device


def render_prompt(tokenizer, prompt: str) -> str:
    messages = [{"role": "user", "content": prompt}]
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def count_quantized_modules(model) -> tuple[int, dict[str, int]]:
    counts: dict[str, int] = {}
    total = 0
    for module in model.modules():
        if hasattr(module, "qweight"):
            name = type(module).__name__
            counts[name] = counts.get(name, 0) + 1
            total += 1
    return total, counts


def main() -> None:
    args = parse_args()
    config = read_and_validate_config(args.model)

    # Importing the local AutoRound package registers the Transformers backend
    # used by checkpoints saved with format="auto_round".
    import auto_round  # noqa: F401
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        trust_remote_code=True,
        fix_mistral_regex=True,
    )
    model_config = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
    model_config.quantization_config["backend"] = args.backend
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        config=model_config,
        device_map=args.device_map,
        torch_dtype="auto",
        trust_remote_code=True,
    ).eval()

    quantized_count, quantized_types = count_quantized_modules(model)
    if quantized_count == 0:
        raise RuntimeError("Loaded checkpoint exposes no qweight modules")

    prompt_text = render_prompt(tokenizer, args.prompt)
    encoded = tokenizer(prompt_text, return_tensors="pt")
    device = input_device(model)
    encoded = {name: value.to(device) for name, value in encoded.items()}
    input_length = encoded["input_ids"].shape[-1]

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    with torch.inference_mode():
        output = model.generate(
            **encoded,
            max_new_tokens=args.max_new_tokens,
        )
    response = tokenizer.decode(
        output[0][input_length:],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    ).strip()
    if not response:
        raise RuntimeError("Generation completed but returned an empty response")
    if "\ufffd" in response:
        raise RuntimeError(f"Generation returned invalid Unicode: {response}")

    print(f"[verify] model_type={config['model_type']}")
    print(f"[verify] quarot_config={config['quarot_config']}")
    print(f"[verify] quantized_modules={quantized_count}, types={quantized_types}")
    print(f"[verify] generation_config={model.generation_config.to_diff_dict()}")
    print(f"[verify] prompt={args.prompt}")
    print(f"[verify] response={response}")


if __name__ == "__main__":
    main()
