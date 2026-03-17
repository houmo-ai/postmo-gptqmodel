from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import json
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


def expand_qwen3_5_cache(past_key_values, expand_size):
    if past_key_values is None or expand_size <= 1:
        return past_key_values

    if hasattr(past_key_values, "batch_repeat_interleave"):
        past_key_values.batch_repeat_interleave(expand_size)
        return past_key_values

    for cache_name in ("key_cache", "value_cache", "conv_states", "recurrent_states"):
        cache_group = getattr(past_key_values, cache_name, None)
        if cache_group is None:
            continue
        for layer_idx, cache_tensor in enumerate(cache_group):
            if cache_tensor is not None:
                cache_group[layer_idx] = cache_tensor.repeat_interleave(expand_size, dim=0)
    return past_key_values


def get_model_input_device(model):
    embed = model.get_input_embeddings()
    if embed is not None and hasattr(embed, "weight"):
        return embed.weight.device
    return next(model.parameters()).device


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="/data02/datasets/Qwen3.5-4B")
    parser.add_argument("--num_beams", type=int, default=4)
    parser.add_argument("--seqlen", type=int, default=512)
    parser.add_argument("--nsamples", type=int, default=100)
    parser.add_argument("--min_new_tokens", type=int, default=300)
    parser.add_argument("--gen_dir", type=str, default="gen_data")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--trust_remote_code", action="store_true", default=False)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    print("替换前:", GenerationMixin._beam_search, flush=True)
    GenerationMixin._beam_search = _beam_search_ivsg
    print("替换后:", GenerationMixin._beam_search, flush=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=args.trust_remote_code)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        device_map="auto",
        torch_dtype="auto",
        trust_remote_code=args.trust_remote_code,
    )
    _hook_handles, _hidden_dim = register_ivsg_linear_hooks(model)

    vocab_sample_size = min(args.nsamples, model.config.vocab_size)
    n_vocab = random.sample(range(model.config.vocab_size), vocab_sample_size)

    os.makedirs(args.gen_dir, exist_ok=True)
    prefix = args.model_path.rstrip("/").split("/")[-1]
    model_input_device = get_model_input_device(model)

    with open(os.path.join(args.gen_dir, prefix + ".jsonl"), "a") as f:
        for i in n_vocab:
            reset_ivsg_linear_stats(model)
            input_ids = torch.tensor([[i]], device=model_input_device)
            print("generating:", i, flush=True)
            random_len = random.randint(3, 6)
            outputs1, attention_mask, past_key_values = greedy_prefix_with_cache(model, input_ids, random_len)
            expand_qwen3_5_cache(past_key_values, args.num_beams)
            expand_ivsg_linear_stats(model, args.num_beams)

            remaining_new_tokens = max(args.seqlen - outputs1.shape[1], 1)
            min_new_tokens = min(args.min_new_tokens, remaining_new_tokens)
            outputs = model.generate(
                outputs1,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=True,
                do_sample=False,
                num_beams=args.num_beams,
                min_new_tokens=min_new_tokens,
                max_new_tokens=remaining_new_tokens,
                repetition_penalty=1.5,
                eos_token_id=None,
            )
            gen_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            print(gen_text, flush=True)
            text_dict = {"text": gen_text[0]}
            f.write(json.dumps(text_dict))
            f.write("\n")
            f.flush()
