"""AutoRound mixed-bit quantization for Laguna S 2.1.

Routed expert projections use W4 while all other quantized linear layers use W8.
Embeddings, normalization weights, and lm_head remain BF16 unless
``--quant-lm-head`` is enabled.
"""

import argparse
import os
import re
from pathlib import Path

import torch
from packaging.version import Version


EXPERT_PATTERN = "model.layers.*.mlp.experts.*.(gate_proj|up_proj|down_proj)"


def parse_args():
    parser = argparse.ArgumentParser(description="Laguna S 2.1 AutoRound expert-W4/rest-W8 quantization")
    parser.add_argument("--model", default="poolside/Laguna-S-2.1")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--expert-bits", type=int, default=4, choices=(2, 3, 4, 5, 8))
    parser.add_argument("--rest-bits", type=int, default=8, choices=(2, 3, 4, 5, 8))
    parser.add_argument("--group-size", type=int, default=64)
    parser.add_argument("--sym", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--nsamples", type=int, default=128)
    parser.add_argument("--seqlen", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulate-steps", type=int, default=1)
    parser.add_argument("--dataset", default="NeelNanda/pile-10k")
    parser.add_argument("--device-map", default="0,1,2,3")
    parser.add_argument("--format", default="auto_round")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--quant-lm-head", action="store_true")
    parser.add_argument("--low-gpu-mem-usage", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--low-cpu-mem-usage", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--inspect-only", action="store_true")
    return parser.parse_args()


def build_layer_config(expert_bits: int, group_size: int, sym: bool) -> dict:
    return {
        EXPERT_PATTERN: {
            "bits": expert_bits,
            "group_size": group_size,
            "sym": sym,
        }
    }


def default_output_dir(model: str, expert_bits: int, rest_bits: int, group_size: int) -> str:
    model_name = Path(model.rstrip("/")).name
    return str(
        Path("work_dirs")
        / f"{model_name}-autoround-expert-w{expert_bits}-rest-w{rest_bits}-g{group_size}"
    )


def validate_environment(model_path: str):
    import transformers
    from transformers import AutoConfig

    if Version(transformers.__version__) < Version("5.12.0"):
        raise RuntimeError(
            f"Laguna remote code requires transformers>=5.12.0; found {transformers.__version__}."
        )

    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    if config.model_type != "laguna":
        raise ValueError(f"Expected model_type='laguna', found {config.model_type!r}.")
    if config.num_experts <= 0 or config.moe_intermediate_size <= 0:
        raise ValueError("Laguna config does not describe routed MoE experts.")
    return config


def inspect_model(model_path: str) -> None:
    from transformers import AutoModelForCausalLM

    from auto_round.modeling.fused_moe.moe_experts_interface import (
        _infer_expert_weights_transposed,
        prepare_model_for_moe_quantization,
    )
    from auto_round.utils.common import to_standard_regex

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        device_map={"": "meta"},
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    sparse_block = next(layer.mlp for layer in model.model.layers if hasattr(layer.mlp, "experts"))
    experts = sparse_block.experts
    projection_config = {"is_input_proj": True, "output_multiplier": 2}
    is_transposed = _infer_expert_weights_transposed(experts, "gate_up_proj", projection_config)
    if is_transposed:
        raise RuntimeError("Laguna expert weights were incorrectly detected as transposed.")

    gate_up_shape = tuple(experts.gate_up_proj.shape)
    down_shape = tuple(experts.down_proj.shape)
    sparse_layer_count = sum(hasattr(layer.mlp, "experts") for layer in model.model.layers)
    unfused_modules = prepare_model_for_moe_quantization(model)
    expert_regex = re.compile(to_standard_regex(EXPERT_PATTERN))
    matched_expert_linears = [
        name
        for name, module in model.named_modules()
        if isinstance(module, torch.nn.Linear) and expert_regex.search(name)
    ]
    expected_linears = sparse_layer_count * experts.num_experts * 3
    if len(unfused_modules) != sparse_layer_count or len(matched_expert_linears) != expected_linears:
        raise RuntimeError(
            f"Expected {sparse_layer_count} unfused modules and {expected_linears} expert linears, "
            f"found {len(unfused_modules)} and {len(matched_expert_linears)}."
        )

    print(f"model={type(model).__name__}, model_type={model.config.model_type}")
    print(f"experts={type(experts).__name__}, count={experts.num_experts}")
    print(f"gate_up_proj={gate_up_shape}, layout=nn.Linear")
    print(f"down_proj={down_shape}, layout=nn.Linear")
    print(f"expert_pattern={EXPERT_PATTERN}")
    print(f"unfused_modules={len(unfused_modules)}, matched_expert_linears={len(matched_expert_linears)}")


def main() -> None:
    args = parse_args()
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    config = validate_environment(args.model)

    print(
        f"Laguna config: layers={config.num_hidden_layers}, experts={config.num_experts}, "
        f"top_k={config.num_experts_per_tok}, hidden={config.hidden_size}, "
        f"moe_intermediate={config.moe_intermediate_size}"
    )
    if args.inspect_only:
        inspect_model(args.model)
        return

    from auto_round import AutoRound

    output_dir = args.output_dir or default_output_dir(
        args.model,
        args.expert_bits,
        args.rest_bits,
        args.group_size,
    )
    layer_config = build_layer_config(args.expert_bits, args.group_size, args.sym)
    print(
        f"Quantization: routed experts=W{args.expert_bits}G{args.group_size}, "
        f"other eligible linears=W{args.rest_bits}G{args.group_size}, lm_head="
        f"{'W' + str(args.rest_bits) if args.quant_lm_head else 'BF16'}"
    )
    print(f"AutoRound device_map={args.device_map}, output={output_dir}, format={args.format}")

    autoround = AutoRound(
        model=args.model,
        bits=args.rest_bits,
        group_size=args.group_size,
        sym=args.sym,
        layer_config=layer_config,
        dataset=args.dataset,
        iters=args.iters,
        nsamples=args.nsamples,
        seqlen=args.seqlen,
        batch_size=args.batch_size,
        gradient_accumulate_steps=args.gradient_accumulate_steps,
        device_map=args.device_map,
        low_gpu_mem_usage=args.low_gpu_mem_usage,
        low_cpu_mem_usage=args.low_cpu_mem_usage,
        quant_lm_head=args.quant_lm_head,
        trust_remote_code=True,
        seed=args.seed,
    )
    autoround.quantize_and_save(output_dir=output_dir, format=args.format)
    print(f"Quantized model saved to: {output_dir}")


if __name__ == "__main__":
    main()