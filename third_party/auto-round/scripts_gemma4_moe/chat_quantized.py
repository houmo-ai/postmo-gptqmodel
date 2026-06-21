"""Interactive chat for quantized Gemma4 MoE checkpoints only.

This script only accepts an already-quantized checkpoint directory and refuses
to run if the checkpoint does not advertise quantization metadata.

Example:
    python scripts_gemma4_moe/chat_quantized.py \
        --quantized_model output/gemma4-26b-a4b-autoround-g64 \
        --device auto
"""

import argparse
import json
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GPTQMODEL_ROOT = os.path.dirname(os.path.dirname(PROJECT_ROOT))
for path in (GPTQMODEL_ROOT, PROJECT_ROOT):
    if path not in sys.path:
        sys.path.insert(0, path)

import torch


def parse_args():
    parser = argparse.ArgumentParser(description="Chat with a quantized Gemma4 MoE checkpoint")
    parser.add_argument("--quantized_model", type=str, required=True,
                        help="Path to the quantized checkpoint directory")
    parser.add_argument("--device", type=str, default="auto",
                        help="Runtime device: auto, cpu, cuda, cuda:0, ...")
    parser.add_argument("--dtype", type=str, default="bfloat16",
                        choices=["bfloat16", "float16", "float32"],
                        help="Preferred model loading dtype")
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--prompt", type=str, default="",
                        help="Optional single-turn prompt. If omitted, enter interactive mode.")
    return parser.parse_args()


def _get_torch_dtype(dtype_name):
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[dtype_name]


def _read_checkpoint_config(quantized_model):
    config_path = os.path.join(quantized_model, "config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"config.json not found under quantized model path: {quantized_model}")

    with open(config_path) as handle:
        return json.load(handle)


def _assert_quantized_gemma4_moe_checkpoint(quantized_model):
    config = _read_checkpoint_config(quantized_model)
    quant_config = config.get("quantization_config")
    if not quant_config:
        raise ValueError("The provided path is not a quantized checkpoint: quantization_config is missing")

    if config.get("model_type") != "gemma4":
        raise ValueError(f"Expected a gemma4 checkpoint, got model_type={config.get('model_type')}")

    text_config = config.get("text_config", {})
    if not text_config.get("enable_moe_block", False):
        raise ValueError("The checkpoint is not a Gemma4 MoE model")

    return config


def _resolve_runtime_device(device):
    if device == "auto":
        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            return "cuda:0"
        return "cpu"

    if device == "cuda":
        device = "cuda:0"

    if device.startswith("cuda"):
        if not torch.cuda.is_available() or torch.cuda.device_count() == 0:
            raise RuntimeError("CUDA device was requested but no visible CUDA device is available")

        if ":" not in device:
            return "cuda:0"

        try:
            index = int(device.split(":", 1)[1])
        except ValueError as exc:
            raise ValueError(f"Invalid CUDA device string: {device}") from exc

        visible_count = torch.cuda.device_count()
        if index < 0 or index >= visible_count:
            print(
                f"[chat] Requested {device}, but only {visible_count} CUDA device(s) are visible. "
                "Falling back to cuda:0"
            )
            return "cuda:0"

    return device


def _count_quantized_layers(model):
    quant_layer_count = 0
    quant_types = {}
    for module in model.modules():
        if hasattr(module, "qweight"):
            quant_layer_count += 1
            name = type(module).__name__
            quant_types[name] = quant_types.get(name, 0) + 1
    return quant_layer_count, quant_types


def _load_quantized_model(quantized_model, device, dtype_name, config):
    """Load quantized Gemma4 MoE with GPTQModel, never AutoRound/vanilla HF."""
    del dtype_name  # GPTQModel resolves dtype/backend from the quantized checkpoint.

    from gptqmodel import GPTQModel
    from transformers import AutoTokenizer

    runtime_device = _resolve_runtime_device(device)
    tokenizer = AutoTokenizer.from_pretrained(quantized_model, trust_remote_code=True)

    with torch.no_grad():
        gptq_model = GPTQModel.load(quantized_model, device=runtime_device, trust_remote_code=True)
    model = gptq_model.model
    if not hasattr(model, "generate"):
        raise TypeError(f"{type(model).__name__} does not implement generate()")

    quant_layer_count, quant_types = _count_quantized_layers(model)
    if quant_layer_count == 0:
        raise RuntimeError("Loaded model does not expose any quantized layers (qweight not found)")

    model.eval()
    print("[chat] Loaded with GPTQModel.load")
    print(f"[chat] Runtime device: {runtime_device}")
    print(
        f"[chat] Quantized checkpoint confirmed: bits={config['quantization_config'].get('bits')}, "
        f"group_size={config['quantization_config'].get('group_size')}, layers={quant_layer_count}"
    )
    print(f"[chat] Quant layer types: {quant_types}")
    return model, tokenizer, runtime_device


def _tokenize_messages(tokenizer, messages, device):
    try:
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    inputs = tokenizer(text, return_tensors="pt")
    return {key: value.to(device) for key, value in inputs.items()}


def _generate_reply(model, tokenizer, messages, device, max_new_tokens):
    inputs = _tokenize_messages(tokenizer, messages, device)
    input_length = inputs["input_ids"].shape[-1]
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True).strip()


def run_single_turn(model, tokenizer, device, prompt, max_new_tokens):
    messages = [{"role": "user", "content": prompt}]
    response = _generate_reply(model, tokenizer, messages, device, max_new_tokens)
    print(f"\nUser: {prompt}")
    print(f"Assistant: {response}")


def run_interactive_chat(model, tokenizer, device, max_new_tokens):
    messages = []
    print("\n[chat] Enter interactive mode. Commands: /exit, /quit, /clear")
    while True:
        try:
            prompt = input("\nUser> ").strip()
        except EOFError:
            print("\n[chat] EOF received, exiting")
            break
        except KeyboardInterrupt:
            print("\n[chat] Interrupted, exiting")
            break

        if not prompt:
            continue
        if prompt in {"/exit", "/quit"}:
            print("[chat] Bye")
            break
        if prompt == "/clear":
            messages.clear()
            print("[chat] Conversation cleared")
            continue

        messages.append({"role": "user", "content": prompt})
        response = _generate_reply(model, tokenizer, messages, device, max_new_tokens)
        messages.append({"role": "assistant", "content": response})
        print(f"Assistant> {response}")


def main():
    args = parse_args()
    config = _assert_quantized_gemma4_moe_checkpoint(args.quantized_model)
    model, tokenizer, runtime_device = _load_quantized_model(
        quantized_model=args.quantized_model,
        device=args.device,
        dtype_name=args.dtype,
        config=config,
    )

    if args.prompt:
        run_single_turn(model, tokenizer, runtime_device, args.prompt, args.max_new_tokens)
        return

    run_interactive_chat(model, tokenizer, runtime_device, args.max_new_tokens)


if __name__ == "__main__":
    main()