import argparse
import os
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoTokenizer

from gptqmodel import GPTQModel, QuantizeConfig


NON_MOE_LINEAR_SUFFIXES = (
    ".self_attn.q_proj",
    ".self_attn.q_a_proj",
    ".self_attn.q_b_proj",
    ".self_attn.q_absorbed_proj",
    ".self_attn.q_rope_proj",
    ".self_attn.kv_a_proj_with_mqa",
    ".self_attn.kv_a_proj_latent",
    ".self_attn.kv_a_proj_rope",
    ".self_attn.kv_b_proj",
    ".self_attn.o_proj",
    ".mlp.gate_proj",
    ".mlp.up_proj",
    ".mlp.down_proj",
    ".mlp.shared_experts.gate_proj",
    ".mlp.shared_experts.up_proj",
    ".mlp.shared_experts.down_proj",
)


def build_calib_dataset(tokenizer, nsamples: int = 128, seqlen: int = 2048):
    texts = load_dataset(
        'wikitext',
        'wikitext-2-raw-v1',
        split='test',
        verification_mode='no_checks',
    )["text"]
    merged_text = "\n\n".join(t for t in texts if t and t.strip())
    encoded = tokenizer(merged_text, truncation=False, return_tensors="pt", verbose=False)
    input_ids = encoded["input_ids"][0]
    attention_mask = encoded["attention_mask"][0]

    total_chunks = input_ids.numel() // seqlen
    if total_chunks == 0:
        raise ValueError(
            f"Not enough calibration tokens to build a single chunk of length {seqlen}."
        )

    nsamples = min(nsamples, total_chunks)
    dataset = []

    for i in range(nsamples):
        start = i * seqlen
        end = start + seqlen
        dataset.append({
            "input_ids": input_ids[start:end].unsqueeze(0),
            "attention_mask": attention_mask[start:end].unsqueeze(0),
        })

    return dataset


def build_mix_precision_override(model, args):
    override = {}

    if args.max_search_batches is not None:
        override["max_calibration_batches"] = args.max_search_batches

    if args.moe_only_mix_search:
        fixed_bits = {"lm_head": None}

        for module_name, module in model.model.named_modules():
            if not isinstance(module, torch.nn.Linear):
                continue
            if ".experts." in module_name and ".shared_experts." not in module_name:
                continue
            if module_name.endswith(NON_MOE_LINEAR_SUFFIXES):
                fixed_bits[module_name] = args.non_moe_fixed_bits

        override["fixed_bits"] = fixed_bits

    return override


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="/data02/datasets/GLM-4.7-Flash", help="HF 模型 ID 或本地路径，例如 mistralai/GptOss-8x7B-v0.1")
    parser.add_argument("--out", type=str, default="output/GLM-4.7-Flash-4bit-64g", help="量化模型的保存目录")
    parser.add_argument("--bits", type=int, default=4)
    parser.add_argument("--group-size", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--nsamples", type=int, default=128)
    parser.add_argument("--seqlen", type=int, default=2048)
    parser.add_argument("--rotation", type=str, default='hadamard', help="可选旋转优化 (GLM4 MoE Lite暂不支持，留空)")
    parser.add_argument("--sym", default=True, help="对称量化")
    parser.add_argument("--mse", default=True, help="开启 MSE 辅助优化")
    parser.add_argument("--hessian-mse", default=True, help="开启 Hessian MSE 辅助优化")
    parser.add_argument("--mix-precision-search", default=True, help="启用混合精度搜索")
    parser.add_argument("--max-search-batches", type=int, default=None, help="仅限制混合精度搜索使用的校准 batch 数；默认复用全部 --nsamples")
    parser.add_argument("--moe-only-mix-search", action="store_true", help="普通 linear 固定为 --non-moe-fixed-bits，仅对 MoE 算子做混合精度搜索")
    parser.add_argument("--non-moe-fixed-bits", type=int, default=8, help="开启 --moe-only-mix-search 时，普通 linear 固定使用的 bit 数")
    parser.add_argument("--diceq-search", action="store_true", help="启用 DiceQ 搜索")
    args = parser.parse_args()

    model_id = os.path.normpath(args.model)
    model_name = Path(model_id).name
    save_dir = args.out or f"{model_name}-{args.bits}bit-{args.group_size}g-hessian_mse"

    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True, trust_remote_code=True)
    calib_dataset = build_calib_dataset(tokenizer, nsamples=args.nsamples, seqlen=args.seqlen)

    qcfg = QuantizeConfig(
        bits=args.bits,
        group_size=args.group_size,
        sym=bool(args.sym),
        mse=bool(args.mse),
        rotation=args.rotation if not args.mix_precision_search else None,
        hessian_mse=bool(args.hessian_mse),
        mix_precision_search=args.mix_precision_search,
        diceq_search=args.diceq_search,
    )

    model = GPTQModel.load(model_id, qcfg, trust_remote_code=True)

    if args.mix_precision_search:
        mix_precision_override = build_mix_precision_override(model, args)
        if mix_precision_override:
            model.quantize_config.extension_set("mix_precision", mix_precision_override)
    
    # 根据显存调整 batch_size 以加速量化
    model.quantize(calib_dataset, batch_size=args.batch_size)

    model.save(save_dir)

    # 简单推理验证
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    # model = GPTQModel.load(save_dir, device=device, trust_remote_code=True)
    prompt = "GLM-4.7-Flash is"
    print(tokenizer.decode(model.generate(**tokenizer(prompt, return_tensors="pt"))[0]))


if __name__ == "__main__":
    main()