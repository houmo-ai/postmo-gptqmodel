import argparse
import os


os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForImageTextToText, AutoTokenizer

from gptqmodel.utils.perplexity import Perplexity


DEFAULT_FP_MODEL_PATH = "/data02/datasets/Qwen3.5-35B-A3B"


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate floating-point model perplexity on wikitext2")
    parser.add_argument("--model", type=str, default=DEFAULT_FP_MODEL_PATH, help="Floating-point model path")
    parser.add_argument("--n-ctx", type=int, default=512, help="Context size for perplexity evaluation")
    parser.add_argument("--n-batch", type=int, default=512, help="Batch size for perplexity evaluation")
    parser.add_argument("--split", type=str, default="test", help="Dataset split to evaluate")
    parser.add_argument("--text-column", type=str, default="text", help="Dataset text column")
    parser.add_argument("--device-map", type=str, default="auto", help="Device map passed to model loader")
    parser.add_argument(
        "--trust-remote-code",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable trust_remote_code when loading tokenizer/model",
    )
    return parser.parse_args()


def load_fp_model(model_path: str, device_map: str, trust_remote_code: bool):
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=trust_remote_code)
    if getattr(config, "model_type", None) == "qwen3_5_moe":
        return AutoModelForImageTextToText.from_pretrained(
            model_path,
            device_map=device_map,
            torch_dtype="auto",
            trust_remote_code=trust_remote_code,
        )

    return AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map=device_map,
        torch_dtype="auto",
        trust_remote_code=trust_remote_code,
    )


def calculate_avg_ppl(model, tokenizer, n_ctx: int, n_batch: int, split: str, text_column: str) -> float:
    ppl = Perplexity(
        model=model,
        tokenizer=tokenizer,
        dataset_path="wikitext",
        dataset_name="wikitext-2-raw-v1",
        split=split,
        text_column=text_column,
    )
    values = ppl.calculate(n_ctx=n_ctx, n_batch=n_batch)
    return sum(values) / len(values)


def main():
    args = parse_args()

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        use_fast=True,
        trust_remote_code=args.trust_remote_code,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = load_fp_model(
        model_path=args.model,
        device_map=args.device_map,
        trust_remote_code=args.trust_remote_code,
    )

    avg_ppl = calculate_avg_ppl(
        model=model,
        tokenizer=tokenizer,
        n_ctx=args.n_ctx,
        n_batch=args.n_batch,
        split=args.split,
        text_column=args.text_column,
    )
    print(f"Model: {args.model}")
    print(f"Dataset: wikitext/wikitext-2-raw-v1 [{args.split}]")
    print(f"Average PPL: {avg_ppl:.4f}")


if __name__ == "__main__":
    main()