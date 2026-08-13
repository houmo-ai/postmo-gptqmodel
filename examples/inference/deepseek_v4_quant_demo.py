#!/usr/bin/env python3
"""Run a numerical sanity check on a packed DeepSeek-V4 GPTQ checkpoint.

This intentionally loads the saved GPTQ/AutoRound artifact directly.  It does
not use xhquant wrapping or an exported graph, so a failure here belongs to the
quantized checkpoint/runtime rather than the deployment conversion pipeline.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import importlib.util
import json
import math
import time
from pathlib import Path
from typing import Callable

import torch
from transformers import AutoTokenizer, DynamicCache
from transformers.generation.logits_process import LogitsProcessor, LogitsProcessorList
from transformers.utils import logging as transformers_logging

from gptqmodel import BACKEND, GPTQModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument(
        "--prompt",
        default="请简要解释为什么天空看起来是蓝色的。",
    )
    parser.add_argument(
        "--target-prompt-tokens",
        type=int,
        default=None,
        help=(
            "Build an exact-length prompt (for example 8192) by inserting "
            "deterministic context before the user question while retaining "
            "the chat assistant-generation suffix."
        ),
    )
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument(
        "--prefill-chunk-size",
        type=int,
        default=256,
        help=(
            "Run prompt prefill through the Transformers DynamicCache in chunks of "
            "this many tokens. The default 256 avoids materializing an 8K eager-attention "
            "matrix; use 0 only for an explicit unchunked comparison."
        ),
    )
    parser.add_argument(
        "--validate-prefill-chunk-equivalence-tokens",
        type=int,
        default=0,
        help=(
            "Before writing the report, compare fresh-cache one-shot and chunked "
            "prefill on this many prompt tokens using both final logits and the full "
            "next-token distribution sequence. Zero disables the check; 512 validates "
            "one 512-token call against 2x256 by default."
        ),
    )
    parser.add_argument(
        "--diagnose-prefill-chunk-equivalence",
        action="store_true",
        help=(
            "When the prefill equivalence check is enabled, also compare each "
            "decoder layer's final-token stages and the final SWA/CSA/HCA cache state."
        ),
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature; 0 selects deterministic greedy decoding.",
    )
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=33377335)
    parser.add_argument(
        "--backend", choices=("auto", "torch", "triton"), default="auto"
    )
    parser.add_argument("--dtype", choices=("float16", "bfloat16"), default="float16")
    parser.add_argument(
        "--allow-fp16-reduced-precision-reduction",
        action="store_true",
        help=(
            "Opt into CUDA's faster FP16 GEMM reduction. By default the demo disables "
            "it because M-dependent accumulation rounding can make one-shot and chunked "
            "prefill diverge even when cache and position handling are correct."
        ),
    )
    parser.add_argument(
        "--require-all-gpu",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Fail if any materialized model tensor is placed outside CUDA.",
    )
    parser.add_argument(
        "--diagnostic-layer",
        type=int,
        help="Capture first-prefill inputs/outputs inside this decoder layer.",
    )
    parser.add_argument(
        "--capture-layer-summaries",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Capture every decoder-layer output summary. By default this is enabled "
            "for normal prompts and disabled for exact long-context prompts so the "
            "reported prefill timing is not dominated by diagnostic reductions."
        ),
    )
    parser.add_argument(
        "--dump-module-tensors",
        help="Exact named module whose first prefill input/output should be saved on CPU.",
    )
    parser.add_argument(
        "--dump-module-tensors-output",
        type=Path,
        help="Output .pt file for --dump-module-tensors.",
    )
    parser.add_argument(
        "--allow-degenerate",
        action="store_true",
        help="Return success even when the continuation is an obvious repeated-token failure.",
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


_PROMPT_MARKER = "__GPTQMODEL_DEEPSEEK_V4_USER_CONTENT_MARKER_7F31__"


@contextmanager
def fp16_reduced_precision_reduction_policy(*, allow: bool):
    """Apply a process-global CUDA matmul policy and always restore it.

    PyTorch may choose different GEMM kernels for M=512 and M=256.  With
    reduced-precision FP16 accumulation enabled, equal rows can consequently
    differ solely because prefill was chunked.  The validation demo defaults
    to the stricter reduction policy while retaining an explicit performance
    opt-in.
    """

    matmul_backend = torch.backends.cuda.matmul
    original = bool(matmul_backend.allow_fp16_reduced_precision_reduction)
    requested = bool(allow)
    state = {
        "original_value": original,
        "requested_value": requested,
        "actual_during_run": None,
        "restored_value": None,
        "restoration_succeeded": None,
    }
    matmul_backend.allow_fp16_reduced_precision_reduction = requested
    state["actual_during_run"] = bool(
        matmul_backend.allow_fp16_reduced_precision_reduction
    )
    try:
        yield state
    finally:
        matmul_backend.allow_fp16_reduced_precision_reduction = original
        restored = bool(matmul_backend.allow_fp16_reduced_precision_reduction)
        state["restored_value"] = restored
        state["restoration_succeeded"] = restored == original


def _token_ids(tokenizer, text: str) -> list[int]:
    encoded = tokenizer(text, add_special_tokens=False)
    token_ids = encoded["input_ids"]
    if isinstance(token_ids, torch.Tensor):
        token_ids = token_ids.detach().cpu().tolist()
    if token_ids and isinstance(token_ids[0], list):
        if len(token_ids) != 1:
            raise ValueError("Only batch-size one prompt construction is supported")
        token_ids = token_ids[0]
    return [int(token_id) for token_id in token_ids]


def _load_deepseek_v4_renderer(model_path: Path) -> Callable[[str], str] | None:
    encoder_path = model_path / "encoding" / "encoding_dsv4.py"
    if not encoder_path.is_file():
        return None
    spec = importlib.util.spec_from_file_location(
        f"gptqmodel_deepseek_v4_encoding_{abs(hash(encoder_path))}",
        encoder_path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load DeepSeek-V4 encoding module: {encoder_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    def render(content: str) -> str:
        return module.encode_messages(
            [{"role": "user", "content": content}],
            thinking_mode="chat",
            reasoning_effort="low",
        )

    return render


def resolve_prompt_renderer(
    tokenizer,
    model_path: Path,
) -> tuple[Callable[[str], str], str]:
    deepseek_renderer = _load_deepseek_v4_renderer(model_path)
    if deepseek_renderer is not None:
        return deepseek_renderer, "deepseek_v4_encoding_module_chat"

    config_path = model_path / "config.json"
    model_type = None
    if config_path.is_file():
        with config_path.open(encoding="utf-8") as handle:
            model_type = json.load(handle).get("model_type")
    if model_type == "deepseek_v4":
        # Quantized artifacts do not always copy the optional encoding/ folder.
        # This is the official basic chat-mode form from encoding_dsv4.py.
        return (
            lambda content: (
                "<｜begin▁of▁sentence｜><｜User｜>"
                + content
                + "<｜Assistant｜></think>"
            ),
            "deepseek_v4_builtin_chat",
        )

    if getattr(tokenizer, "chat_template", None):

        def render_chat_template(content: str) -> str:
            return tokenizer.apply_chat_template(
                [{"role": "user", "content": content}],
                tokenize=False,
                add_generation_prompt=True,
            )

        return render_chat_template, "tokenizer_chat_template"
    return lambda content: content, "plain_text_no_generation_suffix"


def _long_context_filler_text() -> str:
    return "".join(
        f"\n长上下文校验资料第{index}段：这是一段中性的预填充材料，"
        "用于检查缓存、位置编码和稀疏注意力；请忽略本段，只回答末尾的问题。"
        for index in range(256)
    )


def build_prompt_inputs(
    tokenizer,
    prompt: str,
    *,
    renderer: Callable[[str], str],
    prompt_format: str,
    target_prompt_tokens: int | None,
) -> tuple[dict[str, torch.Tensor], dict[str, object]]:
    marked = renderer(_PROMPT_MARKER)
    if marked.count(_PROMPT_MARKER) != 1:
        raise ValueError("Prompt renderer must preserve the user content exactly once")
    prefix_text, suffix_text = marked.split(_PROMPT_MARKER, 1)
    rendered = prefix_text + prompt + suffix_text
    suffix_ids = _token_ids(tokenizer, suffix_text)

    if target_prompt_tokens is None:
        inputs = dict(
            tokenizer(
                rendered,
                add_special_tokens=False,
                return_tensors="pt",
            )
        )
        actual_tokens = int(inputs["input_ids"].shape[-1])
        metadata = {
            "format": prompt_format,
            "requested_tokens": None,
            "actual_tokens": actual_tokens,
            "exact_target_met": None,
            "filler_tokens": 0,
            "generation_suffix_tokens": len(suffix_ids),
            "generation_suffix_ids": suffix_ids,
        }
        return inputs, metadata

    if target_prompt_tokens <= 0:
        raise ValueError("--target-prompt-tokens must be positive")
    prefix_ids = _token_ids(tokenizer, prefix_text)
    prompt_ids = _token_ids(tokenizer, prompt)
    question_intro_ids = _token_ids(tokenizer, "\n\n现在请回答以下问题：\n")
    fixed_tokens = (
        len(prefix_ids) + len(question_intro_ids) + len(prompt_ids) + len(suffix_ids)
    )
    if target_prompt_tokens < fixed_tokens:
        raise ValueError(
            f"Requested {target_prompt_tokens} prompt tokens, but the chat framing "
            f"and user question already require {fixed_tokens}"
        )
    filler_count = target_prompt_tokens - fixed_tokens
    filler_source = _token_ids(tokenizer, _long_context_filler_text())
    if filler_count and not filler_source:
        raise ValueError("Tokenizer produced no tokens for the long-context filler")
    filler_ids = (
        (
            filler_source
            * ((filler_count + len(filler_source) - 1) // len(filler_source))
        )[:filler_count]
        if filler_count
        else []
    )
    # The filler precedes the real question, leaving both the question and the
    # assistant-generation suffix at the end of the prompt.
    input_ids = prefix_ids + filler_ids + question_intro_ids + prompt_ids + suffix_ids
    if len(input_ids) != target_prompt_tokens:
        raise AssertionError("Exact-length prompt construction failed")
    if suffix_ids and input_ids[-len(suffix_ids) :] != suffix_ids:
        raise AssertionError("Chat assistant-generation suffix was not preserved")
    input_tensor = torch.tensor([input_ids], dtype=torch.long)
    inputs = {
        "input_ids": input_tensor,
        "attention_mask": torch.ones_like(input_tensor),
    }
    metadata = {
        "format": prompt_format,
        "requested_tokens": target_prompt_tokens,
        "actual_tokens": len(input_ids),
        "exact_target_met": True,
        "fixed_chat_and_question_tokens": fixed_tokens,
        "filler_tokens": filler_count,
        "question_intro_tokens": len(question_intro_ids),
        "generation_suffix_tokens": len(suffix_ids),
        "generation_suffix_ids": suffix_ids,
    }
    return inputs, metadata


def resolve_prefill_chunking(
    prompt_tokens: int, configured_size: int
) -> tuple[int | None, int]:
    """Return the Transformers chunk size and number of prompt forward calls.

    A zero CLI value is an explicit opt-out. Positive values are passed directly
    to ``GenerationConfig.prefill_chunk_size`` so Transformers owns cache
    construction, position ids, prefix masks, and transition into one-token
    decode.
    """
    if configured_size < 0:
        raise ValueError("--prefill-chunk-size must be non-negative")
    if configured_size == 0:
        return None, 1
    return configured_size, math.ceil(prompt_tokens / configured_size)


def prefill_last_logits_with_fresh_cache(
    model,
    *,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    chunk_size: int | None,
    capture_diagnostics: bool = False,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    list[int],
    int,
    dict[str, torch.Tensor],
    dict[str, object],
]:
    """Prefill a fresh DeepSeek-V4 DynamicCache and return CPU logits.

    This mirrors Transformers' chunked-prefill contract: every call receives
    only its new tokens, the full attention-mask prefix, and the corresponding
    absolute position-id slice. Keeping this helper independent from
    ``generate`` makes the numerical equivalence check inspect logits directly.
    """
    if input_ids.ndim != 2 or attention_mask.shape != input_ids.shape:
        raise ValueError(
            "input_ids and attention_mask must have matching [batch, sequence] shapes"
        )
    if chunk_size is not None and chunk_size <= 0:
        raise ValueError("chunk_size must be positive or None")

    sequence_length = input_ids.shape[-1]
    effective_chunk_size = sequence_length if chunk_size is None else chunk_size
    position_ids = attention_mask.long().cumsum(dim=-1) - 1
    position_ids.masked_fill_(attention_mask == 0, 0)
    stage_trace: dict[str, torch.Tensor] = {}
    hooks = []
    if capture_diagnostics:
        layers = getattr(getattr(model, "model", None), "layers", None)
        if layers is None:
            raise ValueError("Layer diagnostics require model.model.layers")

        def save_last_token(name: str, value) -> None:
            if isinstance(value, torch.Tensor):
                stage_trace[name] = value[:, -1].detach().float().cpu()

        for layer_index, layer in enumerate(layers):
            prefix = f"layer.{layer_index}"

            def capture_attn_hc(_module, _inputs, output, _prefix=prefix):
                if isinstance(output, tuple) and len(output) == 3:
                    save_last_token(_prefix + ".attn_hc.post", output[0])
                    save_last_token(_prefix + ".attn_hc.comb", output[1])
                    save_last_token(_prefix + ".attn_hc.collapsed", output[2])

            def capture_self_attn(_module, _inputs, output, _prefix=prefix):
                value = output[0] if isinstance(output, tuple) else output
                save_last_token(_prefix + ".self_attn", value)

            def capture_ffn_hc(_module, _inputs, output, _prefix=prefix):
                if isinstance(output, tuple) and len(output) == 3:
                    save_last_token(_prefix + ".ffn_hc.post", output[0])
                    save_last_token(_prefix + ".ffn_hc.comb", output[1])
                    save_last_token(_prefix + ".ffn_hc.collapsed", output[2])

            def capture_mlp(_module, _inputs, output, _prefix=prefix):
                save_last_token(_prefix + ".mlp", output)

            def capture_layer(_module, _inputs, output, _prefix=prefix):
                save_last_token(_prefix + ".output", output)

            hooks.extend(
                [
                    layer.attn_hc.register_forward_hook(capture_attn_hc),
                    layer.self_attn.register_forward_hook(capture_self_attn),
                    layer.ffn_hc.register_forward_hook(capture_ffn_hc),
                    layer.mlp.register_forward_hook(capture_mlp),
                    layer.register_forward_hook(capture_layer),
                ]
            )

    cache = DynamicCache(config=model.config)
    chunk_lengths = []
    outputs = None
    logits_chunks = []
    try:
        for start in range(0, sequence_length, effective_chunk_size):
            end = min(start + effective_chunk_size, sequence_length)
            chunk_lengths.append(end - start)
            outputs = model(
                input_ids=input_ids[:, start:end],
                attention_mask=attention_mask[:, :end],
                position_ids=position_ids[:, start:end],
                past_key_values=cache,
                use_cache=True,
                logits_to_keep=0,
                return_dict=True,
            )
            # Retain FP16/BF16 payloads on CPU and compute distribution metrics
            # later in small FP32 token blocks.  For the standard 512-token
            # audit this avoids keeping two full-vocabulary outputs on GPU.
            logits_chunks.append(outputs.logits.detach().cpu())
    finally:
        for hook in hooks:
            hook.remove()
    if outputs is None:
        raise ValueError("Cannot prefill an empty prompt")
    full_logits = torch.cat(logits_chunks, dim=1)
    logits = full_logits[:, -1, :].float()
    cache_snapshot = snapshot_deepseek_v4_cache(cache) if capture_diagnostics else {}
    return (
        logits,
        full_logits,
        chunk_lengths,
        cache.get_seq_length(),
        stage_trace,
        cache_snapshot,
    )


def snapshot_deepseek_v4_cache(cache) -> dict[str, object]:
    """Copy compact SWA/CSA/HCA cache state to CPU for cross-run comparison."""
    snapshot: dict[str, object] = {}
    tensor_attrs = ("keys", "values")
    dict_attrs = (
        "buffer_kv",
        "buffer_gate",
        "compressed_kv",
        "entry_count",
        "overlap_kv",
        "overlap_gate",
    )
    for layer_index, layer in enumerate(cache.layers):
        prefix = f"layer.{layer_index}"
        snapshot[prefix + ".cache_class"] = type(layer).__name__
        snapshot[prefix + ".cumulative_length"] = int(
            getattr(layer, "cumulative_length", 0)
        )
        seen_tensor_ids = set()
        for attr in tensor_attrs:
            value = getattr(layer, attr, None)
            if not isinstance(value, torch.Tensor) or id(value) in seen_tensor_ids:
                continue
            seen_tensor_ids.add(id(value))
            snapshot[prefix + "." + attr] = value.detach().cpu()
        for attr in dict_attrs:
            values = getattr(layer, attr, None)
            if not isinstance(values, dict):
                continue
            for name, value in values.items():
                key = f"{prefix}.{attr}.{name}"
                if isinstance(value, torch.Tensor):
                    snapshot[key] = value.detach().cpu()
                elif value is None or isinstance(value, (int, float, str, bool)):
                    snapshot[key] = value
    return snapshot


def tensor_comparison(
    actual: torch.Tensor, reference: torch.Tensor
) -> dict[str, object]:
    """Return stable comparison metrics without serializing tensor payloads."""
    result: dict[str, object] = {
        "actual_shape": list(actual.shape),
        "reference_shape": list(reference.shape),
        "shape_match": actual.shape == reference.shape,
    }
    if actual.shape != reference.shape:
        return result
    actual = actual.float()
    reference = reference.float()
    finite = torch.isfinite(actual) & torch.isfinite(reference)
    result["finite_fraction"] = float(finite.float().mean())
    result["exact_fraction"] = float((actual == reference).float().mean())
    if not finite.any():
        return result
    actual_finite = actual[finite]
    reference_finite = reference[finite]
    error = actual_finite - reference_finite
    result["relative_rmse"] = float(
        error.square().mean().sqrt()
        / reference_finite.square()
        .mean()
        .sqrt()
        .clamp_min(torch.finfo(torch.float32).tiny)
    )
    result["max_absolute_error"] = float(error.abs().max())
    if actual_finite.numel() > 1:
        result["cosine_similarity"] = float(
            torch.nn.functional.cosine_similarity(
                actual_finite.flatten(), reference_finite.flatten(), dim=0
            )
        )
    return result


def compare_named_snapshots(
    actual: dict[str, object], reference: dict[str, object]
) -> dict[str, dict[str, object]]:
    comparisons = {}
    for name in sorted(set(actual) | set(reference)):
        lhs = actual.get(name)
        rhs = reference.get(name)
        if isinstance(lhs, torch.Tensor) and isinstance(rhs, torch.Tensor):
            comparisons[name] = tensor_comparison(lhs, rhs)
        else:
            comparisons[name] = {
                "actual": lhs,
                "reference": rhs,
                "equal": lhs == rhs,
            }
    return comparisons


def _distribution_summary(values: list[torch.Tensor]) -> dict[str, float] | None:
    """Summarize a list of small CPU vectors without serializing samples."""

    if not values:
        return None
    joined = torch.cat([value.detach().float().flatten().cpu() for value in values])
    return {
        "mean": float(joined.mean()),
        "p50": float(torch.quantile(joined, 0.50)),
        "p95": float(torch.quantile(joined, 0.95)),
        "min": float(joined.min()),
        "max": float(joined.max()),
    }


def compare_full_sequence_logits(
    one_shot: torch.Tensor,
    chunked: torch.Tensor,
    *,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    token_block_size: int = 8,
) -> dict[str, object]:
    """Compare full-vocabulary logits and next-token likelihoods on CPU.

    Full DeepSeek-V4 logits are large even at 512 tokens.  Inputs remain in
    their compact checkpoint dtype and are promoted to FP32 one token block at
    a time for cosine/RMSE and KL/JS calculations.
    """

    if one_shot.shape != chunked.shape or one_shot.ndim != 3:
        raise ValueError(
            "one_shot and chunked logits must have equal [batch, sequence, vocab] shapes"
        )
    if input_ids.shape != one_shot.shape[:2] or attention_mask.shape != input_ids.shape:
        raise ValueError(
            "input_ids and attention_mask must match the logits batch/sequence dimensions"
        )
    if token_block_size <= 0:
        raise ValueError("token_block_size must be positive")

    one_shot = one_shot.detach().cpu()
    chunked = chunked.detach().cpu()
    input_ids = input_ids.detach().long().cpu()
    attention_mask = attention_mask.detach().bool().cpu()
    batch_size, sequence_length, vocab_size = one_shot.shape
    valid_token_count = int(attention_mask.sum())
    total_elements = valid_token_count * vocab_size
    one_shot_finite_count = 0
    chunked_finite_count = 0
    finite_token_count = 0
    top1_matches = 0
    cosine_values = []
    relative_rmse_values = []
    max_abs_values = []
    forward_kl_values = []
    reverse_kl_values = []
    js_values = []
    tiny = torch.finfo(torch.float32).tiny
    log_two = math.log(2.0)

    for start in range(0, sequence_length, token_block_size):
        end = min(start + token_block_size, sequence_length)
        reference = one_shot[:, start:end].float()
        candidate = chunked[:, start:end].float()
        token_mask = attention_mask[:, start:end]
        one_finite = torch.isfinite(reference)
        chunk_finite = torch.isfinite(candidate)
        one_shot_finite_count += int((one_finite & token_mask.unsqueeze(-1)).sum())
        chunked_finite_count += int((chunk_finite & token_mask.unsqueeze(-1)).sum())
        finite_rows = one_finite.all(dim=-1) & chunk_finite.all(dim=-1) & token_mask
        finite_token_count += int(finite_rows.sum())
        if not finite_rows.any():
            continue
        reference = reference[finite_rows]
        candidate = candidate[finite_rows]
        difference = candidate - reference
        top1_matches += int(
            (reference.argmax(dim=-1) == candidate.argmax(dim=-1)).sum()
        )
        cosine_values.append(
            torch.nn.functional.cosine_similarity(reference, candidate, dim=-1)
        )
        relative_rmse_values.append(
            difference.square().mean(dim=-1).sqrt()
            / reference.square().mean(dim=-1).sqrt().clamp_min(tiny)
        )
        max_abs_values.append(difference.abs().amax(dim=-1))

        reference_log_prob = torch.log_softmax(reference, dim=-1)
        candidate_log_prob = torch.log_softmax(candidate, dim=-1)
        reference_prob = reference_log_prob.exp()
        candidate_prob = candidate_log_prob.exp()
        forward_kl_values.append(
            (reference_prob * (reference_log_prob - candidate_log_prob))
            .sum(dim=-1)
            .clamp_min(0)
        )
        reverse_kl_values.append(
            (candidate_prob * (candidate_log_prob - reference_log_prob))
            .sum(dim=-1)
            .clamp_min(0)
        )
        mixture_log_prob = (
            torch.logaddexp(reference_log_prob, candidate_log_prob) - log_two
        )
        js_values.append(
            (
                0.5
                * (reference_prob * (reference_log_prob - mixture_log_prob)).sum(dim=-1)
                + 0.5
                * (candidate_prob * (candidate_log_prob - mixture_log_prob)).sum(dim=-1)
            ).clamp_min(0)
        )

    # Logit position t predicts prompt token t+1.  Requiring both sides of the
    # transition to be unmasked naturally excludes the first valid prompt
    # token, which has no predecessor prediction inside this prompt.
    transition_mask = attention_mask[:, :-1] & attention_mask[:, 1:]
    one_shot_nll_values = []
    chunked_nll_values = []
    nll_abs_difference_values = []
    for start in range(0, max(0, sequence_length - 1), token_block_size):
        end = min(start + token_block_size, sequence_length - 1)
        reference = one_shot[:, start:end].float()
        candidate = chunked[:, start:end].float()
        target = input_ids[:, start + 1 : end + 1]
        rows = transition_mask[:, start:end]
        rows &= torch.isfinite(reference).all(dim=-1) & torch.isfinite(candidate).all(
            dim=-1
        )
        if not rows.any():
            continue
        reference_nll = (
            -torch.log_softmax(reference[rows], dim=-1)
            .gather(
                dim=-1,
                index=target[rows].unsqueeze(-1),
            )
            .squeeze(-1)
        )
        candidate_nll = (
            -torch.log_softmax(candidate[rows], dim=-1)
            .gather(
                dim=-1,
                index=target[rows].unsqueeze(-1),
            )
            .squeeze(-1)
        )
        one_shot_nll_values.append(reference_nll)
        chunked_nll_values.append(candidate_nll)
        nll_abs_difference_values.append((candidate_nll - reference_nll).abs())

    one_nll = torch.cat(one_shot_nll_values) if one_shot_nll_values else torch.empty(0)
    chunk_nll = torch.cat(chunked_nll_values) if chunked_nll_values else torch.empty(0)
    one_mean_nll = float(one_nll.mean()) if one_nll.numel() else None
    chunk_mean_nll = float(chunk_nll.mean()) if chunk_nll.numel() else None
    nll_absolute_difference = (
        abs(chunk_mean_nll - one_mean_nll)
        if one_mean_nll is not None and chunk_mean_nll is not None
        else None
    )
    nll_relative_difference = (
        nll_absolute_difference
        / max(abs(one_mean_nll), torch.finfo(torch.float32).tiny)
        if nll_absolute_difference is not None
        else None
    )

    return {
        "batch_size": batch_size,
        "sequence_length": sequence_length,
        "vocabulary_size": vocab_size,
        "valid_prompt_tokens": valid_token_count,
        "finite_full_distribution_tokens": finite_token_count,
        "one_shot_finite_fraction": (
            one_shot_finite_count / total_elements if total_elements else None
        ),
        "chunked_finite_fraction": (
            chunked_finite_count / total_elements if total_elements else None
        ),
        "top1_match_rate": top1_matches / finite_token_count
        if finite_token_count
        else None,
        "cosine_similarity": _distribution_summary(cosine_values),
        "relative_rmse": _distribution_summary(relative_rmse_values),
        "max_absolute_error": _distribution_summary(max_abs_values),
        "kl_divergence_one_shot_to_chunked": _distribution_summary(forward_kl_values),
        "kl_divergence_chunked_to_one_shot": _distribution_summary(reverse_kl_values),
        "jensen_shannon_divergence": _distribution_summary(js_values),
        "next_token_lm": {
            "predicted_prompt_tokens": int(one_nll.numel()),
            "one_shot_mean_nll": one_mean_nll,
            "chunked_mean_nll": chunk_mean_nll,
            "one_shot_perplexity": math.exp(one_mean_nll)
            if one_mean_nll is not None
            else None,
            "chunked_perplexity": math.exp(chunk_mean_nll)
            if chunk_mean_nll is not None
            else None,
            "mean_nll_absolute_difference": nll_absolute_difference,
            "mean_nll_relative_difference": nll_relative_difference,
            "per_token_nll_absolute_difference": _distribution_summary(
                nll_abs_difference_values
            ),
        },
    }


def compare_prefill_chunk_equivalence(
    model,
    *,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    chunk_size: int,
    capture_diagnostics: bool = False,
) -> dict[str, object]:
    """Compare one-shot and chunked fresh-cache full-sequence logits."""
    (
        one_shot,
        one_shot_full,
        one_shot_chunks,
        one_shot_cache_length,
        one_shot_stages,
        one_shot_cache,
    ) = prefill_last_logits_with_fresh_cache(
        model,
        input_ids=input_ids,
        attention_mask=attention_mask,
        chunk_size=None,
        capture_diagnostics=capture_diagnostics,
    )
    (
        chunked,
        chunked_full,
        chunk_lengths,
        chunked_cache_length,
        chunked_stages,
        chunked_cache,
    ) = prefill_last_logits_with_fresh_cache(
        model,
        input_ids=input_ids,
        attention_mask=attention_mask,
        chunk_size=chunk_size,
        capture_diagnostics=capture_diagnostics,
    )
    one_shot_finite = torch.isfinite(one_shot)
    chunked_finite = torch.isfinite(chunked)
    both_finite = one_shot_finite & chunked_finite
    if both_finite.any():
        reference = one_shot[both_finite]
        candidate = chunked[both_finite]
        diff = candidate - reference
        rel_rmse = float(
            diff.square().mean().sqrt()
            / reference.square()
            .mean()
            .sqrt()
            .clamp_min(torch.finfo(torch.float32).tiny)
        )
        max_abs_error = float(diff.abs().max())
    else:
        rel_rmse = None
        max_abs_error = None
    cosine = None
    if one_shot_finite.all() and chunked_finite.all():
        cosine = float(
            torch.nn.functional.cosine_similarity(one_shot, chunked, dim=-1).mean()
        )
    one_shot_top1 = one_shot.argmax(dim=-1).tolist()
    chunked_top1 = chunked.argmax(dim=-1).tolist()
    report = {
        "tokens": input_ids.shape[-1],
        "chunk_size": chunk_size,
        "one_shot_chunk_lengths": one_shot_chunks,
        "chunked_chunk_lengths": chunk_lengths,
        "one_shot_cache_length": one_shot_cache_length,
        "chunked_cache_length": chunked_cache_length,
        "one_shot_finite_fraction": float(one_shot_finite.float().mean()),
        "chunked_finite_fraction": float(chunked_finite.float().mean()),
        "one_shot_top1_ids": one_shot_top1,
        "chunked_top1_ids": chunked_top1,
        "top1_match": one_shot_top1 == chunked_top1,
        "cosine_similarity": cosine,
        "relative_rmse": rel_rmse,
        "max_absolute_error": max_abs_error,
        "full_sequence": compare_full_sequence_logits(
            one_shot_full,
            chunked_full,
            input_ids=input_ids,
            attention_mask=attention_mask,
        ),
    }
    if capture_diagnostics:
        report["layer_types"] = list(getattr(model.config, "layer_types", []))
        report["stage_comparisons"] = compare_named_snapshots(
            chunked_stages, one_shot_stages
        )
        report["cache_comparisons"] = compare_named_snapshots(
            chunked_cache, one_shot_cache
        )
    return report


class LightweightGenerationObserver(LogitsProcessor):
    """Measure time-to-first-score and retain only four lightweight score rows."""

    def __init__(self) -> None:
        self.started_at: float | None = None
        self.first_score_seconds: float | None = None
        self.score_rows: dict[int, torch.Tensor] = {}
        self.last_step = -1
        self.last_row: torch.Tensor | None = None

    def start(self, started_at: float) -> None:
        self.started_at = started_at

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        del input_ids
        step = self.last_step + 1
        if self.first_score_seconds is None:
            if scores.device.type == "cuda":
                torch.cuda.synchronize(scores.device)
            if self.started_at is None:
                raise RuntimeError(
                    "Generation observer must be started before generate()"
                )
            self.first_score_seconds = time.perf_counter() - self.started_at
        row = scores[0].detach()
        if step < 3:
            self.score_rows[step] = row
        self.last_step = step
        self.last_row = row
        return scores

    def retained_rows(self) -> list[tuple[int, torch.Tensor]]:
        rows = dict(self.score_rows)
        if self.last_step >= 0 and self.last_row is not None:
            rows[self.last_step] = self.last_row
        return sorted(rows.items())


def synchronize_model_cuda_devices(model) -> None:
    devices = {
        tensor.device
        for tensor in list(model.parameters()) + list(model.buffers())
        if tensor.device.type == "cuda"
    }
    for device in sorted(devices, key=lambda item: item.index or 0):
        torch.cuda.synchronize(device)


def tensor_summary(value: torch.Tensor) -> dict[str, float | list[int]]:
    value = value.detach().float()
    finite = torch.isfinite(value)
    result: dict[str, float | list[int]] = {
        "shape": list(value.shape),
        "finite_fraction": float(finite.float().mean()),
    }
    if not finite.all():
        return result
    result.update(
        {
            "rms": float(value.square().mean().sqrt()),
            "std": float(value.std()),
            "abs_max": float(value.abs().max()),
        }
    )
    if value.ndim >= 3 and value.shape[1] > 1:
        token_delta = value[:, 1:] - value[:, :-1]
        result["adjacent_token_delta_rms"] = float(token_delta.square().mean().sqrt())
        first = value[:, 0].flatten(1)
        last = value[:, -1].flatten(1)
        result["first_last_cosine"] = float(
            torch.nn.functional.cosine_similarity(first, last).mean()
        )
    if value.ndim == 4 and value.shape[2] > 1:
        stream_delta = value[:, :, 1:] - value[:, :, :-1]
        result["adjacent_stream_delta_rms"] = float(stream_delta.square().mean().sqrt())
    return result


def run_demo(
    args: argparse.Namespace,
    *,
    fp16_reduction_policy: dict[str, bool | None],
) -> tuple[dict[str, object], dict[str, object], list[str]]:
    if args.target_prompt_tokens is not None and args.target_prompt_tokens <= 0:
        raise ValueError("--target-prompt-tokens must be positive")
    if args.max_new_tokens <= 0:
        raise ValueError("--max-new-tokens must be positive")
    if args.prefill_chunk_size < 0:
        raise ValueError("--prefill-chunk-size must be non-negative")
    if args.validate_prefill_chunk_equivalence_tokens < 0:
        raise ValueError(
            "--validate-prefill-chunk-equivalence-tokens must be non-negative"
        )
    if (
        args.diagnose_prefill_chunk_equivalence
        and not args.validate_prefill_chunk_equivalence_tokens
    ):
        raise ValueError(
            "--diagnose-prefill-chunk-equivalence requires "
            "--validate-prefill-chunk-equivalence-tokens"
        )
    if bool(args.dump_module_tensors) != bool(args.dump_module_tensors_output):
        raise ValueError(
            "--dump-module-tensors and --dump-module-tensors-output must be used together"
        )
    if args.temperature < 0:
        raise ValueError("--temperature must be non-negative")
    if not 0 < args.top_p <= 1:
        raise ValueError("--top-p must be in (0, 1]")
    transformers_logging.disable_progress_bar()
    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16}[args.dtype]
    backend = {
        "auto": BACKEND.AUTO,
        "torch": BACKEND.TORCH,
        "triton": BACKEND.TRITON,
    }[args.backend]

    started = time.perf_counter()
    wrapper = GPTQModel.load(
        str(args.model),
        device_map="balanced",
        backend=backend,
        dtype=dtype,
        trust_remote_code=True,
    )
    load_seconds = time.perf_counter() - started
    model = wrapper.model
    tokenizer = AutoTokenizer.from_pretrained(str(args.model), trust_remote_code=True)

    tensor_placement: dict[str, dict[str, int]] = {}
    non_cuda_tensors: list[str] = []
    for collection_name, tensors in (
        ("parameters", model.named_parameters()),
        ("buffers", model.named_buffers()),
    ):
        for name, tensor in tensors:
            device_name = str(tensor.device)
            stats = tensor_placement.setdefault(
                device_name,
                {"tensors": 0, "elements": 0, "bytes": 0},
            )
            stats["tensors"] += 1
            stats["elements"] += tensor.numel()
            stats["bytes"] += tensor.numel() * tensor.element_size()
            if tensor.device.type != "cuda" and len(non_cuda_tensors) < 20:
                non_cuda_tensors.append(f"{collection_name}:{name}@{device_name}")
    if args.require_all_gpu and non_cuda_tensors:
        raise RuntimeError(
            "Model is not fully GPU resident; first non-CUDA tensors: "
            + ", ".join(non_cuda_tensors)
        )

    dtype_observations: dict[str, str] = {}
    parameter_dtypes: dict[str, str] = {}
    for name in (
        "model.embed_tokens.weight",
        "model.layers.0.attn_hc.fn",
        "model.layers.0.input_layernorm.weight",
        "model.layers.0.self_attn.q_a_proj.qweight",
        "model.layers.0.self_attn.q_a_proj.scales",
        "model.layers.2.self_attn.compressor.kv_proj.weight",
        "model.hc_head.hc_fn",
        "model.norm.weight",
        "lm_head.weight",
    ):
        owner = model
        try:
            for part in name.split("."):
                owner = getattr(owner, part)
        except AttributeError:
            continue
        if isinstance(owner, torch.Tensor):
            parameter_dtypes[name] = str(owner.dtype)

    layer_summaries: dict[str, dict[str, float | list[int]]] = {}
    diagnostic_summaries: dict[str, object] = {}
    dumped_module_tensors: dict[str, object] = {}
    hooks = []
    for name, module in model.named_modules():
        if name.endswith("model.layers.0.self_attn.q_a_proj") or name.endswith(
            "layers.0.self_attn.q_a_proj"
        ):

            def observe_q_a(_module, inputs, _name=name):
                if inputs and isinstance(inputs[0], torch.Tensor):
                    dtype_observations[_name + ".input"] = str(inputs[0].dtype)

            def observe_q_a_output(_module, _inputs, output, _name=name):
                if isinstance(output, torch.Tensor):
                    dtype_observations[_name + ".output"] = str(output.dtype)

            hooks.append(module.register_forward_pre_hook(observe_q_a))
            hooks.append(module.register_forward_hook(observe_q_a_output))
            break

    capture_layer_summaries = (
        args.capture_layer_summaries
        if args.capture_layer_summaries is not None
        else args.target_prompt_tokens is None
    )
    if capture_layer_summaries:
        for name, module in model.named_modules():
            if name.startswith("model.layers.") and name.count(".") == 2:

                def observe_layer(_module, _inputs, output, _name=name):
                    if _name in layer_summaries:
                        return
                    value = output[0] if isinstance(output, tuple) else output
                    if isinstance(value, torch.Tensor):
                        layer_summaries[_name] = tensor_summary(value)

                hooks.append(module.register_forward_hook(observe_layer))

    if args.diagnostic_layer is not None:
        layer_prefix = f"model.layers.{args.diagnostic_layer}"
        diagnostic_paths = {
            layer_prefix,
            f"{layer_prefix}.attn_hc",
            f"{layer_prefix}.input_layernorm",
            f"{layer_prefix}.self_attn",
            f"{layer_prefix}.ffn_hc",
            f"{layer_prefix}.post_attention_layernorm",
            f"{layer_prefix}.mlp",
            f"{layer_prefix}.mlp.gate",
            f"{layer_prefix}.mlp.experts",
            f"{layer_prefix}.mlp.shared_experts",
        }

        def summarize_output(value):
            if isinstance(value, torch.Tensor):
                return tensor_summary(value)
            if isinstance(value, (tuple, list)):
                return [summarize_output(item) for item in value]
            return {"type": type(value).__name__}

        for name, module in model.named_modules():
            is_quant_linear = name.startswith(layer_prefix + ".") and hasattr(
                module, "qweight"
            )
            if name not in diagnostic_paths and not is_quant_linear:
                continue

            def observe_input(_module, inputs, _name=name):
                key = _name + ".input"
                if key not in diagnostic_summaries:
                    diagnostic_summaries[key] = summarize_output(inputs)

            def observe_output(_module, _inputs, output, _name=name):
                key = _name + ".output"
                if key not in diagnostic_summaries:
                    diagnostic_summaries[key] = summarize_output(output)
                if (
                    _name == f"{layer_prefix}.mlp.gate"
                    and isinstance(output, tuple)
                    and len(output) >= 3
                ):
                    weights, indices = output[1], output[2]
                    diagnostic_summaries[_name + ".selected_experts"] = sorted(
                        set(indices.detach().cpu().reshape(-1).tolist())
                    )
                    diagnostic_summaries[_name + ".selected_indices"] = (
                        indices.detach().cpu().tolist()
                    )
                    diagnostic_summaries[_name + ".selected_weights"] = (
                        weights.detach().float().cpu().tolist()
                    )

            hooks.append(module.register_forward_pre_hook(observe_input))
            hooks.append(module.register_forward_hook(observe_output))

    if args.dump_module_tensors:
        matching_modules = [
            (name, module)
            for name, module in model.named_modules()
            if name == args.dump_module_tensors
        ]
        if len(matching_modules) != 1:
            raise ValueError(
                f"Expected exactly one module named {args.dump_module_tensors!r}, "
                f"found {len(matching_modules)}"
            )
        dump_name, dump_module = matching_modules[0]

        def to_cpu(value):
            if isinstance(value, torch.Tensor):
                return value.detach().cpu()
            if isinstance(value, (tuple, list)):
                return type(value)(to_cpu(item) for item in value)
            if isinstance(value, dict):
                return {key: to_cpu(item) for key, item in value.items()}
            return value

        def dump_input(_module, inputs):
            if "inputs" not in dumped_module_tensors:
                dumped_module_tensors["module"] = dump_name
                dumped_module_tensors["inputs"] = to_cpu(inputs)

        def dump_output(_module, _inputs, output):
            if "output" not in dumped_module_tensors:
                dumped_module_tensors["output"] = to_cpu(output)

        hooks.append(dump_module.register_forward_pre_hook(dump_input))
        hooks.append(dump_module.register_forward_hook(dump_output))

    prompt_renderer, prompt_format = resolve_prompt_renderer(tokenizer, args.model)
    inputs, prompt_metadata = build_prompt_inputs(
        tokenizer,
        args.prompt,
        renderer=prompt_renderer,
        prompt_format=prompt_format,
        target_prompt_tokens=args.target_prompt_tokens,
    )
    input_device = model.get_input_embeddings().weight.device
    inputs = {key: value.to(input_device) for key, value in inputs.items()}
    prompt_tokens = int(inputs["input_ids"].shape[-1])
    if (
        args.target_prompt_tokens is not None
        and prompt_tokens != args.target_prompt_tokens
    ):
        raise RuntimeError(
            f"Expected exactly {args.target_prompt_tokens} prompt tokens, got {prompt_tokens}"
        )
    prefill_chunk_size, prefill_chunk_count = resolve_prefill_chunking(
        prompt_tokens,
        args.prefill_chunk_size,
    )
    if args.validate_prefill_chunk_equivalence_tokens > prompt_tokens:
        raise ValueError(
            "--validate-prefill-chunk-equivalence-tokens cannot exceed the prompt length"
        )
    if args.validate_prefill_chunk_equivalence_tokens and prefill_chunk_size is None:
        raise ValueError(
            "--validate-prefill-chunk-equivalence-tokens requires chunking; "
            "set --prefill-chunk-size to a positive value"
        )

    synchronize_model_cuda_devices(model)
    generation_observer = LightweightGenerationObserver()
    generation_started = time.perf_counter()
    generation_observer.start(generation_started)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    do_sample = args.temperature > 0
    sampling_kwargs = (
        {"do_sample": True, "temperature": args.temperature, "top_p": args.top_p}
        if do_sample
        else {"do_sample": False}
    )
    with torch.inference_mode():
        with torch.autocast(device_type="cuda", dtype=dtype):
            generated = wrapper.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                prefill_chunk_size=prefill_chunk_size,
                cache_implementation="dynamic",
                **sampling_kwargs,
                return_dict_in_generate=True,
                output_scores=False,
                logits_processor=LogitsProcessorList([generation_observer]),
            )
    synchronize_model_cuda_devices(model)
    generation_seconds = time.perf_counter() - generation_started
    for hook in hooks:
        hook.remove()
    sequence = generated.sequences[0]
    # The generation result also owns the completed 8K+decode cache. The
    # equivalence audit deliberately needs two *fresh* caches, so release the
    # completed cache first while retaining only the tiny token-id sequence.
    del generated

    prefill_chunk_equivalence = None
    equivalence_tokens = args.validate_prefill_chunk_equivalence_tokens
    if equivalence_tokens:
        synchronize_model_cuda_devices(model)
        equivalence_started = time.perf_counter()
        with torch.inference_mode():
            with torch.autocast(device_type="cuda", dtype=dtype):
                prefill_chunk_equivalence = compare_prefill_chunk_equivalence(
                    model,
                    input_ids=inputs["input_ids"][:, :equivalence_tokens],
                    attention_mask=inputs["attention_mask"][:, :equivalence_tokens],
                    chunk_size=prefill_chunk_size,
                    capture_diagnostics=args.diagnose_prefill_chunk_equivalence,
                )
        synchronize_model_cuda_devices(model)
        prefill_chunk_equivalence["seconds"] = time.perf_counter() - equivalence_started
        rel_rmse = prefill_chunk_equivalence["relative_rmse"]
        cosine = prefill_chunk_equivalence["cosine_similarity"]
        prefill_chunk_equivalence["passes_basic_check"] = bool(
            prefill_chunk_equivalence["one_shot_finite_fraction"] == 1.0
            and prefill_chunk_equivalence["chunked_finite_fraction"] == 1.0
            and prefill_chunk_equivalence["top1_match"]
            and cosine is not None
            and cosine >= 0.999
            and rel_rmse is not None
            and rel_rmse <= 0.01
        )

    continuation = sequence[prompt_tokens:]
    continuation_ids = continuation.detach().cpu().tolist()
    text = tokenizer.decode(
        continuation, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )

    score_summaries = []
    for step, retained_row in generation_observer.retained_rows():
        row = retained_row.float()
        finite = torch.isfinite(row)
        top_values, top_ids = torch.topk(row, k=min(5, row.numel()))
        score_summaries.append(
            {
                "step": step,
                "finite_fraction": float(finite.float().mean()),
                "min": float(row[finite].min()) if finite.any() else None,
                "max": float(row[finite].max()) if finite.any() else None,
                "std": float(row[finite].std()) if finite.any() else None,
                "top_ids": top_ids.cpu().tolist(),
                "top_values": top_values.cpu().tolist(),
            }
        )

    prefill_seconds = generation_observer.first_score_seconds
    if prefill_seconds is None:
        raise RuntimeError(
            "Generation produced no score step; prefill timing is unavailable"
        )
    decode_seconds = max(0.0, generation_seconds - prefill_seconds)
    decode_tokens = max(0, len(continuation_ids) - 1)

    longest_run = 0
    current_run = 0
    previous_id = None
    for token_id in continuation_ids:
        if token_id == previous_id:
            current_run += 1
        else:
            current_run = 1
            previous_id = token_id
        longest_run = max(longest_run, current_run)
    degeneration_reasons = []
    if len(continuation_ids) >= 16 and len(set(continuation_ids)) <= 2:
        degeneration_reasons.append("at_most_two_unique_ids")
    if len(continuation_ids) >= 16 and longest_run >= max(
        8, len(continuation_ids) // 2
    ):
        degeneration_reasons.append("long_repeated_token_run")

    report = {
        "model": str(args.model.resolve()),
        "backend": args.backend,
        "requested_activation_dtype": args.dtype,
        "autocast_enabled": True,
        "fp16_reduced_precision_reduction": fp16_reduction_policy,
        "observed_first_q_a_dtypes": dtype_observations,
        "parameter_dtypes": parameter_dtypes,
        "hf_device_map": {
            str(key): str(value)
            for key, value in (getattr(model, "hf_device_map", {}) or {}).items()
        },
        "tensor_placement": tensor_placement,
        "non_cuda_tensors": non_cuda_tensors,
        "capture_layer_summaries": capture_layer_summaries,
        "prefill_layer_summaries": layer_summaries,
        "diagnostic_layer": args.diagnostic_layer,
        "diagnostic_summaries": diagnostic_summaries,
        "load_seconds": load_seconds,
        "generation_seconds": generation_seconds,
        "prefill_seconds": prefill_seconds,
        "prefill_chunk_size": prefill_chunk_size,
        "prefill_chunk_count": prefill_chunk_count,
        "cache_implementation": "dynamic",
        "prefill_chunk_equivalence": prefill_chunk_equivalence,
        "decode_seconds": decode_seconds,
        "prefill_tokens_per_second": (
            prompt_tokens / prefill_seconds if prefill_seconds > 0 else None
        ),
        "decode_tokens_timed": decode_tokens,
        "decode_tokens_per_second": (
            decode_tokens / decode_seconds
            if decode_tokens and decode_seconds > 0
            else None
        ),
        "timing_semantics": (
            "prefill covers all prompt chunks and ends at the first logits-processor "
            "invocation after synchronizing the score device; the first generated token "
            "comes from the final prompt-chunk logits, and decode is synchronized total "
            "generation time minus prefill"
        ),
        "cache_semantics": (
            "Transformers DynamicCache persists sliding/CSA/HCA state across prompt "
            "chunks; generation then forwards each sampled token exactly once with the "
            "existing cache and monotonically increasing position_ids"
        ),
        "prompt": args.prompt,
        "prompt_tokens": prompt_tokens,
        "requested_target_prompt_tokens": args.target_prompt_tokens,
        "prompt_construction": prompt_metadata,
        "requested_max_new_tokens": args.max_new_tokens,
        "sampling": {
            "do_sample": do_sample,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "seed": args.seed,
        },
        "generated_tokens": len(continuation_ids),
        "continuation_ids": continuation_ids,
        "unique_continuation_ids": len(set(continuation_ids)),
        "longest_identical_token_run": longest_run,
        "degenerate": bool(degeneration_reasons),
        "degeneration_reasons": degeneration_reasons,
        "continuation_text": text,
        "full_generation_scores_retained": False,
        "retained_score_rows": len(score_summaries),
        "score_summaries": score_summaries,
    }
    return report, dumped_module_tensors, degeneration_reasons


def main() -> None:
    args = parse_args()
    with fp16_reduced_precision_reduction_policy(
        allow=args.allow_fp16_reduced_precision_reduction,
    ) as reduction_policy:
        report, dumped_module_tensors, degeneration_reasons = run_demo(
            args,
            fp16_reduction_policy=reduction_policy,
        )

    # Serialize only after leaving the policy context so the report proves the
    # process-global setting was restored, not merely that restoration was
    # requested.
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    print(rendered, flush=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    if args.dump_module_tensors_output is not None:
        args.dump_module_tensors_output.parent.mkdir(parents=True, exist_ok=True)
        torch.save(dumped_module_tensors, args.dump_module_tensors_output)
    if degeneration_reasons and not args.allow_degenerate:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
