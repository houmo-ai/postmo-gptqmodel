# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

import torch

from gptqmodel.quantization.calibration.generation.ebss_score import (
    apply_ebss_penalty,
    apply_ebss_penalty_to_log_probs,
    compute_beam_variance_scores,
    reorder_cumulative_expert_counts,
    update_cumulative_expert_counts,
)


def test_update_cumulative_expert_counts_initializes_and_accumulates():
    cumulative = [None, torch.tensor([[1.0, 1.0], [2.0, 0.0]])]
    step = [torch.tensor([[1.0, 0.0], [0.0, 1.0]]), torch.tensor([[2.0, 1.0], [1.0, 3.0]])]

    updated = update_cumulative_expert_counts(cumulative, step)

    assert torch.equal(updated[0], torch.tensor([[1.0, 0.0], [0.0, 1.0]]))
    assert torch.equal(updated[1], torch.tensor([[3.0, 2.0], [3.0, 3.0]]))


def test_reorder_cumulative_expert_counts_matches_beam_index():
    cumulative = [torch.tensor([[1.0, 0.0], [0.0, 1.0]])]
    beam_idx = torch.tensor([1, 0])

    reordered = reorder_cumulative_expert_counts(cumulative, beam_idx)
    assert torch.equal(reordered[0], torch.tensor([[0.0, 1.0], [1.0, 0.0]]))


def test_compute_beam_variance_scores_zero_count_stability():
    scores = compute_beam_variance_scores([torch.zeros((2, 4), dtype=torch.float32)])
    assert torch.equal(scores, torch.zeros(2, dtype=torch.float32))


def test_compute_beam_variance_scores_layer_average_and_normalize():
    counts = [
        torch.tensor([[3.0, 0.0], [1.0, 1.0]]),
        torch.tensor([[5.0, 1.0], [2.0, 2.0]]),
    ]
    scores = compute_beam_variance_scores(counts)

    assert scores.shape == (2,)
    assert torch.isclose(scores.sum(), torch.tensor(1.0))
    assert scores[0] > scores[1]


def test_compute_beam_variance_scores_frequency_mode():
    counts = [torch.tensor([[10.0, 0.0], [5.0, 5.0]])]
    scores = compute_beam_variance_scores(counts, use_frequency=True, normalize=False)
    assert scores[0] > scores[1]


def test_apply_ebss_penalty_in_log_score_form():
    running_beam_scores = torch.tensor([[0.0, -1.0]])
    expert_scores = torch.tensor([0.75, 0.25])

    adjusted = apply_ebss_penalty(running_beam_scores, expert_scores, penalty_strength=2.0)
    assert torch.equal(adjusted, torch.tensor([[-1.5, -1.5]]))


def test_apply_ebss_penalty_to_log_probs_broadcasts_over_vocab():
    log_probs = torch.zeros((1, 2, 3), dtype=torch.float32)
    running_beam_scores = torch.tensor([[0.0, -1.0]], dtype=torch.float32)
    expert_scores = torch.tensor([0.5, 0.0], dtype=torch.float32)

    adjusted = apply_ebss_penalty_to_log_probs(log_probs, running_beam_scores, expert_scores)

    assert adjusted.shape == log_probs.shape
    assert torch.allclose(adjusted[0, 0], torch.full((3,), -0.5))
    assert torch.allclose(adjusted[0, 1], torch.full((3,), -1.0))