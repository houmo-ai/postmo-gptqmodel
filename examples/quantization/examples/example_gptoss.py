import argparse
import os
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoTokenizer

from gptqmodel import GPTQModel, QuantizeConfig


def build_calib_dataset(tokenizer, nsamples: int = 128, seqlen: int = 2048):
    texts = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train').select(range(nsamples))["text"]
    dataset = []
    for t in texts:
        enc = tokenizer(t, truncation=True, max_length=seqlen, return_tensors="pt")
        dataset.append({
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
        })
    return dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="/data02/datasets/gpt-oss-20b", help="HF 模型 ID 或本地路径，例如 mistralai/GptOss-8x7B-v0.1")
    parser.add_argument("--out", type=str, default="output/gpt-oss-20b-4bit-64g", help="量化模型的保存目录")
    parser.add_argument("--bits", type=int, default=4)
    parser.add_argument("--group-size", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--nsamples", type=int, default=128)
    parser.add_argument("--seqlen", type=int, default=2048)
    parser.add_argument("--rotation", type=str, default="hadamard", choices=[None, "hadamard", "block_hadamard", "random"], help="可选旋转优化")
    parser.add_argument("--sym", default=True, help="对称量化")
    parser.add_argument("--mse", default=True, help="开启 MSE 辅助优化")
    args = parser.parse_args()

    model_id = os.path.normpath(args.model)
    model_name = Path(model_id).name
    save_dir = args.out or f"{model_name}-{args.bits}bit-{args.group_size}g"

    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True, trust_remote_code=True)
    calib_dataset = build_calib_dataset(tokenizer, nsamples=args.nsamples, seqlen=args.seqlen)

    qcfg = QuantizeConfig(
        bits=args.bits,
        group_size=args.group_size,
        sym=bool(args.sym),
        mse=bool(args.mse),
        rotation=args.rotation,
    )

    # gpt-oss 需要 remote code（模型定义），这里务必开启
    model = GPTQModel.load(model_id, qcfg, trust_remote_code=True)

    # 根据显存调整 batch_size 以加速量化
    model.quantize(calib_dataset, batch_size=args.batch_size)

    model.save(save_dir)

    # 简单推理验证
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = GPTQModel.load(save_dir, device=device, trust_remote_code=True)
    prompt = "gpt-oss is"
    print(tokenizer.decode(model.generate(**tokenizer(prompt, return_tensors="pt").to(device))[0]))


if __name__ == "__main__":
    main()