from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import json
import sys
import os
import random
from transformers.generation.utils import GenerationMixin
from custom_ivsg import (
    _beam_search_ivsg,
    expand_ivsg_linear_stats,
    register_ivsg_linear_hooks,
    reset_ivsg_linear_stats,
)
import argparse


@torch.inference_mode()
def greedy_prefix_with_cache(model, input_ids, num_new_tokens):
    sequences = input_ids
    attention_mask = torch.ones_like(sequences, device=sequences.device)
    past_key_values = None

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
    _hook_handles, _hidden_dim = register_ivsg_linear_hooks(model)

    n_vocab = random.sample(range(model.config.vocab_size), args.nsamples) 

    if not os.path.exists(args.gen_dir):
        os.mkdir(args.gen_dir)
    prefix = args.model_path.split('/')[-1]
    with open(os.path.join(args.gen_dir, prefix + ".jsonl"), "a") as f:
        for i in n_vocab:
            reset_ivsg_linear_stats(model)
            input_ids = torch.tensor([[i]]).cuda()
            print("generating: ", i, flush=True)
            random_len = random.randint(3, 6)
            outputs1, attention_mask, past_key_values = greedy_prefix_with_cache(model, input_ids, random_len)
            if past_key_values is not None:
                past_key_values.batch_repeat_interleave(args.num_beams)
            expand_ivsg_linear_stats(model, args.num_beams)
            outputs = model.generate(
                outputs1,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=True,
                do_sample=False,
                num_beams=args.num_beams,
                min_new_tokens=300,
                max_new_tokens=args.seqlen,
                repetition_penalty=1.5,
                eos_token_id=None,
            )
            gen_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            print(gen_text, flush=True)
            text_dict = {"text" : gen_text[0]}
            f.write(json.dumps(text_dict))
            f.write('\n')
            f.flush()
