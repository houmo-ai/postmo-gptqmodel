# Copyright 2024-2025 ModelCloud.ai
# Copyright 2024-2025 qubitium@modelcloud.ai
# Contact: qubitium@modelcloud.ai, x.com/qubitium
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
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

logger = setup_logger()
# pretrained_model_id = "/monster/data/model/TinyLlama-1.1B-Chat-v1.0"  # "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
# quantized_model_id = "TinyLlama-1.1B-Chat-v1.0-4bit-128g"

# pretrained_model_id = "/data01/nfs_shared/llm_models/Qwen/Qwen2.5-7B-Instruct"
# quantized_model_id = "Qwen2.5-7B-Instruct-4bit-64g"

# pretrained_model_id = "/data01/nfs_shared/llm_models/Qwen/Qwen3-8B"
# quantized_model_id = "Qwen3-8B-4bit-64g"


# os.makedirs(quantized_model_dir, exist_ok=True)
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


def main(args):
    pretrained_model_id = args.model
    pretrained_model_id = os.path.normpath(pretrained_model_id)
    model_name = Path(pretrained_model_id).name
    bits = args.bits
    group_size = args.group_size
    quantized_model_id = f"{model_name}-{bits}bit-{group_size}g"
    logger.info(f"HF Model: {pretrained_model_id}")
    logger.info(f"Save Model: {quantized_model_id}")
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_id, use_fast=True)

    traindataset = get_wikitext2(tokenizer, nsamples=256, seqlen=2048)

    quantize_config = QuantizeConfig(
        bits=bits,  # quantize model to 4-bit
        group_size=group_size,  # it is recommended to set the value to 128
        sym=True,
        # mse=True,
        rotation="hadamard",
        device="cuda:0",
        desc_act=False,
    )

    # load un-quantized model, the model will always be force loaded into cpu
    model = GPTQModel.load(pretrained_model_id, quantize_config, torch_dtype=torch.float32)

    # quantize model, the calibration_dataset should be list of dict whose keys can only be "input_ids" and "attention_mask"
    # with value under torch.LongTensor type.
    model.quantize(traindataset)

    # save quantized model using safetensors
    model.save(quantized_model_id)

    # load quantized model, currently only support cpu or single gpu
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = GPTQModel.load(quantized_model_id, device=device)

    # inference with model.generate
    print(tokenizer.decode(model.generate(**tokenizer("test is", return_tensors="pt").to(device))[0]))

    print(f"Quantized Model {quantized_model_id} avg PPL is {calculate_avg_ppl(model, tokenizer):.04f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model")
    parser.add_argument("--bits", type=int, default=4)
    parser.add_argument("--group-size", type=int, default=64)
    args = parser.parse_args()

    main(args)
