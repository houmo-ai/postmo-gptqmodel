# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

import torch

from gptqmodel.quantization.calibration.generation.ebss_beam_search import (
    EBSSBeamSearchState,
    bind_local_ebss_beam_search,
    restore_local_ebss_beam_search,
)


class _DummyAdapter:
    def __init__(self):
        self.step_counts = [torch.tensor([[1.0, 0.0], [0.0, 1.0]])]
        self.reordered = None
        self.cleared = False

    def get_step_counts(self):
        return [tensor.clone() if tensor is not None else None for tensor in self.step_counts]

    def clear_step_counts(self):
        self.cleared = True

    def reorder_state(self, beam_idx):
        self.reordered = beam_idx.clone()


class _DummyModel:
    pass


def _dummy_beam_search(self, *args, **kwargs):
    return self


def test_ebss_beam_search_state_updates_and_reorders():
    adapter = _DummyAdapter()
    state = EBSSBeamSearchState(adapter)
    state.initialize_from_adapter()
    state.update_from_step_counts()

    assert torch.equal(state.cumulative_counts[0], torch.tensor([[1.0, 0.0], [0.0, 1.0]]))

    beam_idx = torch.tensor([1, 0])
    state.reorder(beam_idx)
    assert torch.equal(state.cumulative_counts[0], torch.tensor([[0.0, 1.0], [1.0, 0.0]]))
    assert torch.equal(adapter.reordered, beam_idx)

    state.clear_step_counts()
    assert adapter.cleared is True


def test_bind_and_restore_local_ebss_beam_search():
    model = _DummyModel()
    adapter = _DummyAdapter()

    original = bind_local_ebss_beam_search(model, _dummy_beam_search, adapter)
    assert original is None
    assert hasattr(model, "_beam_search")
    assert model._ebss_adapter is adapter

    restore_local_ebss_beam_search(model, original)
    assert not hasattr(model, "_ebss_adapter")


def test_ebss_beam_search_state_computes_scores_and_applies_penalty():
    adapter = _DummyAdapter()
    state = EBSSBeamSearchState(adapter)
    state.initialize_from_adapter()
    state.update_from_step_counts()

    scores = state.compute_expert_scores()
    assert scores.shape == (2,)
    assert torch.isclose(scores.sum(), torch.tensor(1.0))

    log_probs = torch.zeros((1, 2, 3), dtype=torch.float32)
    running_beam_scores = torch.tensor([[0.0, -1.0]], dtype=torch.float32)
    adjusted = state.apply_penalty_to_log_probs(log_probs, running_beam_scores)

    assert adjusted.shape == log_probs.shape