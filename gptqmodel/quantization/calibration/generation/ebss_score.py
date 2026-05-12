from __future__ import annotations

from typing import Iterable, List, Optional

import torch


def update_cumulative_expert_counts(
    cumulative_counts: List[Optional[torch.Tensor]],
    step_counts: Iterable[Optional[torch.Tensor]],
) -> List[Optional[torch.Tensor]]:
    updated = list(cumulative_counts)
    for layer_index, step_count in enumerate(step_counts):
        if step_count is None:
            continue

        if updated[layer_index] is None:
            updated[layer_index] = step_count.detach().clone()
        else:
            updated[layer_index] = updated[layer_index] + step_count.to(updated[layer_index].device)

    return updated


def reorder_cumulative_expert_counts(
    cumulative_counts: List[Optional[torch.Tensor]],
    beam_idx: torch.Tensor,
) -> List[Optional[torch.Tensor]]:
    reordered: List[Optional[torch.Tensor]] = []
    for layer_count in cumulative_counts:
        if layer_count is None:
            reordered.append(None)
            continue
        reordered.append(layer_count.index_select(0, beam_idx.to(layer_count.device)))
    return reordered


def compute_beam_variance_scores(
    cumulative_counts: Iterable[Optional[torch.Tensor]],
    *,
    normalize: bool = True,
    use_frequency: bool = False,
    eps: float = 1e-6,
) -> torch.Tensor:
    valid_layers = []
    for layer_count in cumulative_counts:
        if layer_count is None:
            continue

        layer_tensor = layer_count.to(dtype=torch.float32)
        if use_frequency:
            totals = layer_tensor.sum(dim=-1, keepdim=True)
            layer_tensor = layer_tensor / totals.clamp_min(eps)
        valid_layers.append(torch.var(layer_tensor, dim=-1))

    if not valid_layers:
        return torch.zeros(0, dtype=torch.float32)

    scores = torch.stack(valid_layers, dim=0).mean(dim=0)
    if not normalize:
        return scores

    score_sum = scores.sum()
    if score_sum <= 0:
        return torch.zeros_like(scores)
    return scores / score_sum


def apply_ebss_penalty(
    running_beam_scores: torch.Tensor,
    expert_scores: torch.Tensor,
    *,
    penalty_strength: float = 1.0,
) -> torch.Tensor:
    if expert_scores.numel() == 0:
        return running_beam_scores
    return running_beam_scores - penalty_strength * expert_scores.unsqueeze(0)


def apply_ebss_penalty_to_log_probs(
    log_probs: torch.Tensor,
    running_beam_scores: torch.Tensor,
    expert_scores: torch.Tensor,
    *,
    penalty_strength: float = 1.0,
) -> torch.Tensor:
    adjusted_scores = apply_ebss_penalty(
        running_beam_scores,
        expert_scores,
        penalty_strength=penalty_strength,
    )
    return log_probs + adjusted_scores[:, :, None]
