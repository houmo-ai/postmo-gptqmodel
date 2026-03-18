from __future__ import annotations

from types import MethodType

import torch

from .ebss_score import (
    apply_ebss_penalty_to_log_probs,
    compute_beam_variance_scores,
    reorder_cumulative_expert_counts,
    update_cumulative_expert_counts,
)


class EBSSBeamSearchState:
    """State container for EBSS during beam search.

    This class tracks cumulative expert routing counts across decoding steps,
    keeps those counts aligned with beam reordering, and exposes helper methods
    for turning those counts into EBSS penalties.
    """

    def __init__(self, adapter):
        self.adapter = adapter
        self.cumulative_counts = []

    def initialize_from_adapter(self) -> None:
        step_counts = self.adapter.get_step_counts()
        self.cumulative_counts = [None for _ in step_counts]

    def update_from_step_counts(self) -> None:
        self.cumulative_counts = update_cumulative_expert_counts(
            self.cumulative_counts,
            self.adapter.get_step_counts(),
        )

    def clear_step_counts(self) -> None:
        self.adapter.clear_step_counts()

    def reorder(self, beam_idx) -> None:
        self.cumulative_counts = reorder_cumulative_expert_counts(self.cumulative_counts, beam_idx)
        self.adapter.reorder_state(beam_idx)

    def compute_expert_scores(
        self,
        *,
        normalize: bool = True,
        use_frequency: bool = False,
    ) -> torch.Tensor:
        return compute_beam_variance_scores(
            self.cumulative_counts,
            normalize=normalize,
            use_frequency=use_frequency,
        )

    def apply_penalty_to_log_probs(
        self,
        log_probs: torch.Tensor,
        running_beam_scores: torch.Tensor,
        *,
        penalty_strength: float = 1.0,
        normalize: bool = True,
        use_frequency: bool = False,
    ) -> torch.Tensor:
        expert_scores = self.compute_expert_scores(
            normalize=normalize,
            use_frequency=use_frequency,
        )
        if expert_scores.numel() == 0:
            return log_probs + running_beam_scores[:, :, None]
        return apply_ebss_penalty_to_log_probs(
            log_probs,
            running_beam_scores,
            expert_scores,
            penalty_strength=penalty_strength,
        )


def bind_local_ebss_beam_search(model, beam_search_fn, adapter):
    original = getattr(model, "_beam_search", None)
    model._beam_search = MethodType(beam_search_fn, model)
    model._ebss_adapter = adapter
    return original


def restore_local_ebss_beam_search(model, original_beam_search) -> None:
    if original_beam_search is None:
        if hasattr(model, "_beam_search"):
            delattr(model, "_beam_search")
    else:
        model._beam_search = original_beam_search

    if hasattr(model, "_ebss_adapter"):
        delattr(model, "_ebss_adapter")

