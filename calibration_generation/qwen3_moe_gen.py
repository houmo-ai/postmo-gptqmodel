from transformers import AutoTokenizer
import torch
import json
import sys
import os
from EBSS_models.modeling_qwen3_moe import Qwen3MoeForCausalLM
import random
from transformers.generation.utils import GenerationMixin
from custom_ebss import _beam_search_ebss
import builtins
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path",type=str,default="/data02/datasets/Qwen3-30B-A3B")
    parser.add_argument("--num_beams",type=int,default=4)
    parser.add_argument("--seqlen",type=int,default=512)
    parser.add_argument("--nsamples",type=int,default=100)
    parser.add_argument("--gen_dir",type=str,default="gen_data")
    args = parser.parse_args()
    # monkey patch，用于替换生成beam search
    print("替换前:", GenerationMixin._beam_search)
    GenerationMixin._beam_search = _beam_search_ebss
    print("替换后:", GenerationMixin._beam_search)
    # 全局变量，用于模型初始化
    builtins.NUM_BEAMS = args.num_beams
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = Qwen3MoeForCausalLM.from_pretrained(args.model_path, device_map='auto')

    n_vocab = random.sample(range(model.config.vocab_size), args.nsamples) 

    if not os.path.exists(args.gen_dir):
        os.mkdir(args.gen_dir)
    prefix = args.model_path.split('/')[-1]
    for i in n_vocab:
        input_ids = torch.tensor([[i]]).cuda()
        print("generating: ", i)
        random_len = random.randint(3, 6)
        outputs1 = model.generate(input_ids, do_sample=False, max_length=random_len)
        outputs = model.generate(outputs1, do_sample=False, num_beams=args.num_beams, max_length=args.seqlen,repetition_penalty=2.0)
        gen_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        print(gen_text)
        text_dict = {"text" : gen_text[0]}
        with open(os.path.join(args.gen_dir,prefix+".jsonl"), "a") as f:
            f.write(json.dumps(text_dict))
            f.write('\n')
