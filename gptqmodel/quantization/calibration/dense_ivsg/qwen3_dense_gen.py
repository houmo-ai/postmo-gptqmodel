from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import json
import sys
import os
import random
import  torch.nn as nn
from transformers.generation.utils import GenerationMixin
from custom_ivsg import _beam_search_ivsg
import builtins
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path",type=str,default="/data02/datasets/Qwen3-8B")
    parser.add_argument("--num_beams",type=int,default=4)
    parser.add_argument("--seqlen",type=int,default=512)
    parser.add_argument("--nsamples",type=int,default=100)
    parser.add_argument("--gen_dir",type=str,default="gen_data")
    args = parser.parse_args()

    # monkey patch，用于替换生成beam search
    print("替换前:", GenerationMixin._beam_search)
    GenerationMixin._beam_search = _beam_search_ivsg
    print("替换后:", GenerationMixin._beam_search)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForCausalLM.from_pretrained(args.model_path, device_map='auto')
    def save_input_hook(name):
        def hook_fn(module, input, output):
            x = input[0]
            module.input = x
        return hook_fn

    hook_handles = []
    hidden_dim = 0
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            handle = module.register_forward_hook(save_input_hook(name))
            hook_handles.append(handle)
            hidden_dim = hidden_dim+module.weight.shape[1]
    builtins.HIDDEN_DIM = hidden_dim

    n_vocab = random.sample(range(model.config.vocab_size), args.nsamples) 

    if not os.path.exists(args.gen_dir):
        os.mkdir(args.gen_dir)
    prefix = args.model_path.split('/')[-1]
    with open(prefix+".jsonl", "a") as f:
        for i in n_vocab:
            input_ids = torch.tensor([[i]]).cuda()
            print("generating: ", i)
            random_len = random.randint(3, 6)
            outputs1 = model.generate(input_ids, do_sample=False, min_new_tokens=random_len,max_new_tokens=random_len,eos_token_id=None)
            outputs = model.generate(outputs1, do_sample=False, num_beams=4, min_new_tokens=300,max_new_tokens=args.seqlen,repetition_penalty=1.5, eos_token_id=None)
            gen_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            print(gen_text)
            text_dict = {"text" : gen_text[0]}
            f.write(json.dumps(text_dict))
            f.write('\n')
