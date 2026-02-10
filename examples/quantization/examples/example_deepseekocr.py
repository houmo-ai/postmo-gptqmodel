import argparse
import os
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoTokenizer
from gptqmodel.hf_models.hf_deepseekocr.modeling_deepseekv2 import DeepseekV2ForCausalLM
from gptqmodel.hf_models.hf_deepseekocr.configuration_deepseek_v2 import DeepseekV2Config
from gptqmodel.hf_models.hf_deepseekocr.modeling_deepseekocr import format_messages,text_encode

from gptqmodel import GPTQModel, QuantizeConfig
from gptqmodel.models import MODEL_MAP, BaseQModel


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


def quantize_deepseekv2():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="/data02/datasets/DeepSeek-OCR", help="HF 模型 ID 或本地路径，例如 mistralai/GptOss-8x7B-v0.1")
    parser.add_argument("--out", type=str, default="output/DeepSeek-OCR-4bit-64g", help="量化模型的保存目录")
    parser.add_argument("--bits", type=int, default=4)
    parser.add_argument("--group-size", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--nsamples", type=int, default=128)
    parser.add_argument("--seqlen", type=int, default=2048)
    parser.add_argument("--rotation", type=str, default='hadamard', help="可选旋转优化 (GLM4 MoE Lite暂不支持，留空)")
    parser.add_argument("--sym", default=True, help="对称量化")
    parser.add_argument("--mse", default=True, help="开启 MSE 辅助优化")
    parser.add_argument("--hessian-mse", default=True, help="开启 Hessian MSE 辅助优化")
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
        rotation=args.rotation,
        hessian_mse=bool(args.hessian_mse),
        device="cuda",
    )

    # gpt-oss 需要 remote code（模型定义），这里务必开启
    # model = GPTQModel.load(model_id, qcfg, trust_remote_code=True)
    config = DeepseekV2Config(**DeepseekV2Config.from_pretrained(model_id).language_config)
    model = DeepseekV2ForCausalLM.from_pretrained(model_id, config=config, trust_remote_code=True)
    cfg = model.config
    model_type = "deepseek_v2_ocr"
    QModelClass = MODEL_MAP.get(model_type, BaseQModel)
    qmodel = QModelClass(
        model=model,           # 你已经加载好的 transformers 模型
        quantized=False,          # 还没量化
        quantize_config=qcfg,
        tokenizer=tokenizer,
        qlinear_kernel=None,
        load_quantized_model=False,
        trust_remote_code=True,
        model_local_path=model_id,
        turtle_model=None,
    )

    # 根据显存调整 batch_size 以加速量化
    qmodel.quantize(calib_dataset, batch_size=args.batch_size)

    qmodel.save(save_dir)

    # 简单推理验证
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    # model = GPTQModel.load(save_dir, device=device, trust_remote_code=True)
    prompt = "Who are you?"
    conversation = [
                {
                    "role": "<|User|>",
                    # "content": "<image>\n<|grounding|>Given the layout of the image. ",
                    "content": f'{prompt}',
                    # "content": "君不见黄河之水天上来的下一句是什么？",
                    # "content": "<image>\nFree OCR. ",
                    # "content": "<image>\nParse the figure. ",
                    # "content": "<image>\nExtract the text in the image. ",
                    # "images": [f'{image_file}'],
                },
                {"role": "<|Assistant|>", "content": ""},
            ]
    text = format_messages(conversations=conversation, sft_format='plain', system_prompt='')
    model_inputs = [text_encode(tokenizer, text, bos=False, eos=False)]
    input_ids = torch.LongTensor(model_inputs)
    print(tokenizer.decode(model.generate(input_ids)[0]))


if __name__ == "__main__":
    quantize_deepseekv2()
    ## vision部分暂时不用支持gptqmodel优化，直接W8A8/W8A16就行


