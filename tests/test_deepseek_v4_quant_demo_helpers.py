from __future__ import annotations

import importlib.util
import json
import sys
import time
from pathlib import Path

import pytest
import torch


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "examples"
    / "inference"
    / "deepseek_v4_quant_demo.py"
)
SPEC = importlib.util.spec_from_file_location("deepseek_v4_quant_demo_helpers", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
DEMO = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = DEMO
SPEC.loader.exec_module(DEMO)


class CharacterTokenizer:
    chat_template = None

    def __call__(self, text, *, add_special_tokens=False, return_tensors=None):
        del add_special_tokens
        token_ids = [ord(character) for character in text]
        if return_tensors == "pt":
            input_ids = torch.tensor([token_ids], dtype=torch.long)
            return {
                "input_ids": input_ids,
                "attention_mask": torch.ones_like(input_ids),
            }
        return {"input_ids": token_ids}


def test_exact_prompt_length_keeps_question_and_generation_suffix():
    tokenizer = CharacterTokenizer()

    def renderer(content):
        return "<bos><user>" + content + "<assistant>"

    inputs, metadata = DEMO.build_prompt_inputs(
        tokenizer,
        "最终问题",
        renderer=renderer,
        prompt_format="test_chat",
        target_prompt_tokens=256,
    )

    ids = inputs["input_ids"][0].tolist()
    suffix = [ord(character) for character in "<assistant>"]
    question = [ord(character) for character in "最终问题"]
    assert len(ids) == 256
    assert ids[-len(suffix) :] == suffix
    assert ids[-len(suffix) - len(question) : -len(suffix)] == question
    assert metadata["actual_tokens"] == 256
    assert metadata["exact_target_met"] is True
    assert metadata["generation_suffix_tokens"] == len(suffix)


def test_exact_prompt_rejects_target_shorter_than_framing():
    with pytest.raises(ValueError, match="already require"):
        DEMO.build_prompt_inputs(
            CharacterTokenizer(),
            "question",
            renderer=lambda content: "prefix" + content + "suffix",
            prompt_format="test_chat",
            target_prompt_tokens=4,
        )


def test_resolve_deepseek_builtin_chat_when_encoding_directory_is_absent(tmp_path):
    (tmp_path / "config.json").write_text(
        json.dumps({"model_type": "deepseek_v4"}), encoding="utf-8"
    )
    renderer, prompt_format = DEMO.resolve_prompt_renderer(
        CharacterTokenizer(), tmp_path
    )
    rendered = renderer("hello")
    assert prompt_format == "deepseek_v4_builtin_chat"
    assert rendered.endswith("hello<｜Assistant｜></think>")
    assert rendered.startswith("<｜begin▁of▁sentence｜><｜User｜>")


def test_generation_observer_retains_first_three_and_last_only():
    observer = DEMO.LightweightGenerationObserver()
    observer.start(time.perf_counter())
    for step in range(7):
        scores = torch.full((1, 11), float(step))
        returned = observer(torch.ones((1, step + 1), dtype=torch.long), scores)
        assert returned is scores

    retained = observer.retained_rows()
    assert [step for step, _row in retained] == [0, 1, 2, 6]
    assert [float(row[0]) for _step, row in retained] == [0.0, 1.0, 2.0, 6.0]
    assert observer.first_score_seconds is not None


def test_resolve_prefill_chunking_defaults_to_real_chunks_and_allows_opt_out():
    assert DEMO.resolve_prefill_chunking(8192, 256) == (256, 32)
    assert DEMO.resolve_prefill_chunking(78, 256) == (256, 1)
    assert DEMO.resolve_prefill_chunking(8192, 0) == (None, 1)
    with pytest.raises(ValueError, match="non-negative"):
        DEMO.resolve_prefill_chunking(8192, -1)


def test_fp16_reduced_precision_reduction_policy_restores_global_state():
    backend = torch.backends.cuda.matmul
    original = bool(backend.allow_fp16_reduced_precision_reduction)
    requested = not original

    with DEMO.fp16_reduced_precision_reduction_policy(allow=requested) as state:
        assert state == {
            "original_value": original,
            "requested_value": requested,
            "actual_during_run": requested,
            "restored_value": None,
            "restoration_succeeded": None,
        }
        assert bool(backend.allow_fp16_reduced_precision_reduction) is requested

    assert bool(backend.allow_fp16_reduced_precision_reduction) is original
    assert state["restored_value"] is original
    assert state["restoration_succeeded"] is True


def test_fp16_reduced_precision_reduction_policy_restores_after_exception():
    backend = torch.backends.cuda.matmul
    original = bool(backend.allow_fp16_reduced_precision_reduction)

    with pytest.raises(RuntimeError, match="deliberate"):
        with DEMO.fp16_reduced_precision_reduction_policy(allow=not original):
            raise RuntimeError("deliberate")

    assert bool(backend.allow_fp16_reduced_precision_reduction) is original


def test_full_sequence_logits_reports_next_token_likelihood_and_padding():
    one_shot = torch.tensor(
        [
            [
                [0.0, 1.0, 2.0, 3.0],
                [1.0, 2.0, 3.0, 4.0],
                [2.0, 3.0, 4.0, 5.0],
                [3.0, 4.0, 5.0, 6.0],
            ]
        ]
    )
    chunked = one_shot + torch.tensor([0.0, 0.01, -0.01, 0.0])
    input_ids = torch.tensor([[0, 1, 2, 3]])
    attention_mask = torch.tensor([[0, 1, 1, 1]])

    report = DEMO.compare_full_sequence_logits(
        one_shot,
        chunked,
        input_ids=input_ids,
        attention_mask=attention_mask,
        token_block_size=2,
    )

    expected_one_shot_nll = torch.nn.functional.cross_entropy(
        one_shot[:, 1:3].reshape(-1, 4),
        input_ids[:, 2:4].reshape(-1),
    )
    assert report["valid_prompt_tokens"] == 3
    assert report["finite_full_distribution_tokens"] == 3
    assert report["top1_match_rate"] == 1.0
    assert report["relative_rmse"]["p95"] > 0
    assert report["jensen_shannon_divergence"]["mean"] >= 0
    next_token = report["next_token_lm"]
    assert next_token["predicted_prompt_tokens"] == 2
    assert next_token["one_shot_mean_nll"] == pytest.approx(
        float(expected_one_shot_nll)
    )
    assert next_token["mean_nll_absolute_difference"] > 0


def test_transformers_deepseek_v4_chunked_prefill_reuses_cache_for_decode_cpu():
    from transformers import DeepseekV4Config, DeepseekV4ForCausalLM

    config = DeepseekV4Config(
        vocab_size=32,
        hidden_size=16,
        moe_intermediate_size=8,
        num_hidden_layers=3,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        q_lora_rank=8,
        num_experts_per_tok=2,
        n_routed_experts=4,
        n_shared_experts=1,
        max_position_embeddings=64,
        layer_types=[
            "sliding_attention",
            "compressed_sparse_attention",
            "heavily_compressed_attention",
        ],
        compress_rates={
            "compressed_sparse_attention": 2,
            "heavily_compressed_attention": 4,
        },
        hc_mult=2,
        hc_sinkhorn_iters=2,
        mlp_layer_types=["moe"] * 3,
        sliding_window=4,
        o_groups=2,
        o_lora_rank=4,
        index_n_heads=2,
        index_head_dim=4,
        index_topk=2,
        partial_rotary_factor=0.5,
        bos_token_id=1,
        eos_token_id=31,
        pad_token_id=0,
    )
    torch.manual_seed(3)
    model = DeepseekV4ForCausalLM(config).eval()
    prompt = torch.tensor([[1, 4, 5, 6, 7, 8, 9]])
    attention_mask = torch.ones_like(prompt)
    equivalence = DEMO.compare_prefill_chunk_equivalence(
        model,
        input_ids=prompt,
        attention_mask=attention_mask,
        chunk_size=3,
        capture_diagnostics=True,
    )
    assert equivalence["one_shot_chunk_lengths"] == [7]
    assert equivalence["chunked_chunk_lengths"] == [3, 3, 1]
    assert equivalence["one_shot_cache_length"] == 7
    assert equivalence["chunked_cache_length"] == 7
    assert equivalence["one_shot_finite_fraction"] == 1.0
    assert equivalence["chunked_finite_fraction"] == 1.0
    assert equivalence["top1_match"] is True
    assert equivalence["cosine_similarity"] == pytest.approx(1.0, abs=1e-6)
    assert equivalence["relative_rmse"] < 1e-5
    full_sequence = equivalence["full_sequence"]
    assert full_sequence["sequence_length"] == prompt.shape[-1]
    assert full_sequence["finite_full_distribution_tokens"] == prompt.shape[-1]
    assert full_sequence["top1_match_rate"] == 1.0
    assert full_sequence["relative_rmse"]["p95"] < 1e-5
    assert (
        full_sequence["next_token_lm"]["predicted_prompt_tokens"]
        == prompt.shape[-1] - 1
    )
    assert equivalence["layer_types"] == config.layer_types
    assert "layer.0.attn_hc.collapsed" in equivalence["stage_comparisons"]
    assert "layer.1.self_attn" in equivalence["stage_comparisons"]
    assert "layer.2.output" in equivalence["stage_comparisons"]
    assert (
        equivalence["cache_comparisons"]["layer.0.cumulative_length"]["equal"] is True
    )
    assert (
        equivalence["cache_comparisons"]["layer.1.entry_count.indexer"]["equal"] is True
    )
    assert (
        equivalence["cache_comparisons"]["layer.2.entry_count.compressor"]["equal"]
        is True
    )
    trace = []

    def record_forward(_module, _args, kwargs):
        input_ids = kwargs["input_ids"]
        cache = kwargs["past_key_values"]
        position_ids = kwargs["position_ids"]
        mask = kwargs["attention_mask"]
        trace.append(
            {
                "input_ids": input_ids.detach().clone(),
                "past_length": cache.get_seq_length(),
                "position_ids": position_ids.detach().clone(),
                "mask_length": mask.shape[-1],
                "cache_types": [type(layer).__name__ for layer in cache.layers],
            }
        )

    hook = model.register_forward_pre_hook(record_forward, with_kwargs=True)
    chunked = model.generate(
        input_ids=prompt,
        attention_mask=attention_mask,
        do_sample=False,
        max_new_tokens=3,
        prefill_chunk_size=3,
        cache_implementation="dynamic",
    )
    hook.remove()
    unchunked = model.generate(
        input_ids=prompt,
        attention_mask=attention_mask,
        do_sample=False,
        max_new_tokens=3,
    )

    assert torch.equal(chunked, unchunked)
    assert [row["input_ids"].tolist()[0] for row in trace[:3]] == [
        [1, 4, 5],
        [6, 7, 8],
        [9],
    ]
    assert [row["past_length"] for row in trace] == [0, 3, 6, 7, 8]
    assert [row["position_ids"].tolist()[0] for row in trace] == [
        [0, 1, 2],
        [3, 4, 5],
        [6],
        [7],
        [8],
    ]
    assert [row["mask_length"] for row in trace] == [3, 6, 7, 8, 9]
    assert trace[0]["cache_types"] == [
        "DynamicSlidingWindowLayer",
        "DeepseekV4CSACache",
        "DeepseekV4HCACache",
    ]
    # The final prompt token produces the first sampled token directly. Decode
    # then feeds generated tokens once each; it never replays prompt[-1].
    assert trace[3]["input_ids"].item() == chunked[0, prompt.shape[1]].item()
    assert trace[4]["input_ids"].item() == chunked[0, prompt.shape[1] + 1].item()
