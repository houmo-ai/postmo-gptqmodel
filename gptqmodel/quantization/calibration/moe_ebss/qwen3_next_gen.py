from transformers import AutoTokenizer
import torch
import json
import sys
import os
import random
from transformers.generation.utils import GenerationMixin
from custom_ebss import _beam_search_ebss
import builtins
import argparse
from qwen3_next_ebss import Qwen3NextForCausalLM, patch_qwen3_next_sparse_moe_block


@torch.inference_mode()
def greedy_prefix_with_cache(model, input_ids, total_prefix_length):
    sequences = input_ids
    attention_mask = torch.ones_like(sequences, device=sequences.device)
    past_key_values = None
    num_new_tokens = max(total_prefix_length - input_ids.shape[1], 0)

    for _ in range(num_new_tokens):
        current_input_ids = sequences if past_key_values is None else sequences[:, -1:]
        outputs = model(
            input_ids=current_input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=True,
            return_dict=True,
        )
        next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)
        sequences = torch.cat([sequences, next_token], dim=-1)
        attention_mask = torch.ones_like(sequences, device=sequences.device)
        past_key_values = outputs.past_key_values

    return sequences, attention_mask, past_key_values


def expand_qwen3_next_cache(past_key_values, expand_size):
    if past_key_values is None or expand_size <= 1:
        return past_key_values

    for cache_name in ("key_cache", "value_cache", "conv_states", "recurrent_states"):
        cache_group = getattr(past_key_values, cache_name, None)
        if cache_group is None:
            continue
        for layer_idx, cache_tensor in enumerate(cache_group):
            if cache_tensor is not None:
                cache_group[layer_idx] = cache_tensor.repeat_interleave(expand_size, dim=0)
    return past_key_values


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path",type=str,default="/data02/datasets/Qwen3-Next-80B-A3B-Instruct")
    parser.add_argument("--num_beams",type=int,default=4)
    parser.add_argument("--seqlen",type=int,default=512)
    parser.add_argument("--nsamples",type=int,default=100)
    parser.add_argument("--gen_dir",type=str,default="gen_data")
    args = parser.parse_args()
    # monkey patch，用于替换生成beam search
    print("替换前:", GenerationMixin._beam_search, flush=True)
    GenerationMixin._beam_search = _beam_search_ebss
    print("替换后:", GenerationMixin._beam_search, flush=True)
    # 全局变量，用于模型初始化
    builtins.NUM_BEAMS = args.num_beams
    patch_qwen3_next_sparse_moe_block()
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = Qwen3NextForCausalLM.from_pretrained(args.model_path, dtype=torch.bfloat16,device_map='auto')

    n_vocab = random.sample(range(model.config.vocab_size), args.nsamples) 

    if not os.path.exists(args.gen_dir):
        os.mkdir(args.gen_dir)
    prefix = args.model_path.split('/')[-1]
    with open(os.path.join(args.gen_dir,prefix+".jsonl"), "a") as f:
        for i in n_vocab:
            input_ids = torch.tensor([[i]]).cuda()
            print("generating: ", i, flush=True)
            random_len = random.randint(3, 6)
            outputs1, attention_mask, past_key_values = greedy_prefix_with_cache(model, input_ids, random_len)
            expand_qwen3_next_cache(past_key_values, args.num_beams)
            remaining_new_tokens = max(args.seqlen - outputs1.shape[1], 1)
            outputs = model.generate(
                outputs1,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=True,
                do_sample=False,
                num_beams=args.num_beams,
                max_new_tokens=remaining_new_tokens,
                repetition_penalty=2.0,
            )
            gen_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            print(gen_text, flush=True)
            text_dict = {"text" : gen_text[0]}
            f.write(json.dumps(text_dict))
            f.write('\n')
            f.flush()
