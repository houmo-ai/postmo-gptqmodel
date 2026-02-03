"""
GLM-4.7-Flash (glm4_moe_lite) Quantization Example

This script demonstrates how to quantize GLM-4.7-Flash model using GPTQModel.
The model uses MLA (Multi-head Latent Attention) architecture, which requires
weight fusion before quantization for efficient inference.

Usage:
    python glm4_moe_lite_quant.py
"""

import argparse
import os
import random
from pathlib import Path

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer

from gptqmodel import GPTQModel, QuantizeConfig
from gptqmodel.utils import Perplexity
from gptqmodel.utils.logger import setup_logger
# 添加 gptqmodel 到 Python 路径
# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))


# Model path - 使用实际的模型路径
MODEL_PATH = "/data02/datasets/GLM-4.7-Flash/"
OUTPUT_PATH = "./glm-4.7-flash-gptq-int4"

# Quantization configuration
quantize_config = QuantizeConfig(
    bits=4,                    # 4-bit quantization
    group_size=128,            # Group size for quantization
    desc_act=True,             # Enable activation order optimization
    sym=True,                  # Symmetric quantization
)

def get_wikitext2(tokenizer, nsamples, seqlen):
    # traindata = load_dataset("wikitext", "wikitext-2-raw-v1", split="train").filter(lambda x: len(x["text"]) >= seqlen)

    # return [tokenizer(example["text"]) for example in traindata.select(range(nsamples))]
    traindata = load_dataset(
        "wikitext",
        "wikitext-2-raw-v1",
        split="train",
        verification_mode="no_checks",
    )
    seed = 0
    trainenc = tokenizer("\n\n".join(traindata["text"]))
    random.seed(seed)
    train_samples = []
    input_ids = trainenc.input_ids
    data_seq_len = len(input_ids)
    nsamples = min(nsamples, data_seq_len // seqlen)

    for i in tqdm(range(nsamples)):
        start = i * seqlen
        end = start + seqlen
        inp = trainenc.input_ids[start:end]
        inp_mask = trainenc["attention_mask"][start:end]
        train_samples.append(
            {
                "input_ids": inp,
                "attention_mask": inp_mask,
            }
        )
    # for _ in range(nsamples):
    #     i = random.randint(0, data_seq_len - seqlen - 1)
    #     j = i + seqlen
    #     inp = trainenc.input_ids[i:j]
    #     inp_mask = trainenc["attention_mask"][i:j]

    # train_samples.append(
    #     {
    #         "input_ids": inp,
    #         "attention_mask": inp_mask,
    #     }
    # )
    return train_samples

@torch.no_grad()
def calculate_avg_ppl(model, tokenizer):
    ppl = Perplexity(
        model=model,
        tokenizer=tokenizer,
        dataset_path="wikitext",
        dataset_name="wikitext-2-raw-v1",
        split="test",
        text_column="text",
    )

    all = ppl.calculate(n_ctx=2048, n_batch=2048)

    # average ppl
    avg = sum(all) / len(all)

    return avg


def main():
    print("=" * 60)
    print("GLM-4.7-Flash Quantization with MLA Fusion")
    print("=" * 60)
    
    # Load the model
    print(f"\n[1/4] Loading model from: {MODEL_PATH}")
    model = GPTQModel.from_pretrained(
        MODEL_PATH,
        quantize_config=quantize_config,
        trust_remote_code=True,  # Required for glm4_moe_lite
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=True)
    
    # The MLA fusion happens automatically in pre_quantize_generate_hook_start()
    # This will:
    # 1. Split kv_a_proj_with_mqa into kv_a_proj_latent and kv_a_proj_rope
    # 2. Fuse kv_b_proj into q_absorbed_proj and o_proj
    # 3. Create q_rope_proj from q_b_proj

    traindataset = get_wikitext2(tokenizer, nsamples=256, seqlen=2048)
    
    print("\n[2/4] Starting quantization (MLA fusion will be applied automatically)...")
    model.quantize(
        calibration=traindataset,
        batch_size=1,
    )
    
    print(f"\n[3/4] Saving quantized model to: {OUTPUT_PATH}")
    model.save(OUTPUT_PATH)
    
    print("\n[4/4] Quantization completed!")
    print(f"Quantized model saved to: {OUTPUT_PATH}")
    print("=" * 60)
    
    # Optional: Test the quantized model
    print("\nTesting quantized model...")
    quantized_model = GPTQModel.from_quantized(
        OUTPUT_PATH,
        trust_remote_code=True,
    )
    
    test_input = "你好，请介绍一下你自己。"
    inputs = quantized_model.tokenizer(test_input, return_tensors="pt").to(quantized_model.device)
    
    outputs = quantized_model.generate(
        **inputs,
        max_new_tokens=100,
        do_sample=True,
        temperature=0.7,
    )
    
    response = quantized_model.tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"\nTest input: {test_input}")
    print(f"Model output: {response}")


if __name__ == "__main__":
    main()
