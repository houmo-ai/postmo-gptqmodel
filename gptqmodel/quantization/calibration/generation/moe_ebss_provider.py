import json
import os
import random
from pathlib import Path

import torch
from transformers import AutoTokenizer

from .base import CalibrationProviderBase
from .adapters.registry import create_expert_stats_adapter
from .ebss_beam_search import bind_local_ebss_beam_search, restore_local_ebss_beam_search


def _greedy_prefix_with_cache(model, input_ids, total_prefix_length):
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


def _expand_qwen3_next_cache(past_key_values, expand_size):
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


class MoeEbssProvider(CalibrationProviderBase):
    name = "moe_ebss"

    def _resolve_model_type(self, model, cfg) -> str:
        model_patch = getattr(cfg, "model_patch", None)
        if isinstance(model_patch, str) and model_patch.strip():
            return model_patch.strip().lower()

        model_type = ""
        if model is not None and getattr(model, "config", None) is not None:
            model_type = str(getattr(model.config, "model_type", "")).lower()
        if "glm4_moe_lite" in model_type:
            return "glm4_moe_lite"
        return "qwen3_next" if "next" in model_type else "qwen3_moe"

    def _resolve_model_path(self, model) -> str:
        model_path = ""
        if model is not None and getattr(model, "config", None) is not None:
            model_path = str(getattr(model.config, "_name_or_path", "") or "")
        model_path = model_path.strip()
        if not model_path:
            raise ValueError("Calibration generation: unable to resolve model path from model.config._name_or_path")
        return model_path

    def _load_generation_model(self, model_path: str, resolved_model_type: str, num_beams: int = 4):
        if resolved_model_type == "qwen3_next":
            from ..moe_ebss.qwen3_next_ebss import Qwen3NextForCausalLM, patch_qwen3_next_sparse_moe_block

            patch_qwen3_next_sparse_moe_block()
            return Qwen3NextForCausalLM.from_pretrained(model_path, dtype=torch.bfloat16, device_map="auto")

        from transformers import AutoModelForCausalLM

        return AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )

    def _export_rows(self, rows, output_path: str, overwrite: bool):
        if not output_path:
            return None
        if os.path.exists(output_path) and not overwrite:
            raise ValueError(f"Calibration generation: output already exists: {output_path}")
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False))
                f.write("\n")
        return output_path

    def generate(self, model, tokenizer, cfg):
        output_path = str(cfg.output_path)
        model_path = self._resolve_model_path(model)
        resolved_model_type = self._resolve_model_type(model, cfg)
        generation_model = self._load_generation_model(model_path, resolved_model_type, num_beams=cfg.num_beams)
        generation_tokenizer = tokenizer or AutoTokenizer.from_pretrained(model_path)

        rows = []
        adapter = create_expert_stats_adapter(generation_model, num_beams=cfg.num_beams, model_type=resolved_model_type)
        adapter.enable_collection()

        from ..moe_ebss.custom_ebss import _beam_search_ebss

        original_beam_search = bind_local_ebss_beam_search(generation_model, _beam_search_ebss, adapter)

        try:
            sampled_vocab = random.sample(range(generation_model.config.vocab_size), cfg.nsamples)
            for token_id in sampled_vocab:
                input_ids = torch.tensor([[token_id]], device=generation_model.device)
                if resolved_model_type == "qwen3_next":
                    random_len = random.randint(3, 6)
                    outputs1, attention_mask, past_key_values = _greedy_prefix_with_cache(generation_model, input_ids, random_len)
                    _expand_qwen3_next_cache(past_key_values, cfg.num_beams)
                    remaining_new_tokens = max(cfg.seqlen - outputs1.shape[1], 1)
                    outputs = generation_model.generate(
                        outputs1,
                        attention_mask=attention_mask,
                        past_key_values=past_key_values,
                        use_cache=True,
                        do_sample=False,
                        num_beams=cfg.num_beams,
                        max_new_tokens=remaining_new_tokens,
                        repetition_penalty=2.0,
                    )
                else:
                    random_len = random.randint(3, 6)
                    outputs1 = generation_model.generate(input_ids, do_sample=False, max_length=random_len)
                    outputs = generation_model.generate(
                        outputs1,
                        do_sample=False,
                        num_beams=cfg.num_beams,
                        max_length=cfg.seqlen,
                        repetition_penalty=2.0,
                    )

                generated_text = generation_tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
                rows.append({"text": generated_text})
        finally:
            restore_local_ebss_beam_search(generation_model, original_beam_search)
            adapter.disable_collection()

        exported_path = self._export_rows(rows, output_path, bool(getattr(cfg, "overwrite", False)))
        return {"rows": rows, "output_path": exported_path}
