# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

import torch

from gptqmodel.quantization.calibration.generation.adapters.glm4_moe_lite import Glm4MoeLiteExpertStatsAdapter


class _DummyGate:
    n_routed_experts = 4

    def __call__(self, hidden_states):
        return torch.zeros((hidden_states.shape[0] * hidden_states.shape[1], self.n_routed_experts), dtype=torch.float32)


class _DummyExperts:
    def __call__(self, hidden_states, topk_indices, topk_weights):
        return torch.zeros_like(hidden_states)


class _DummySharedExperts:
    def __call__(self, residuals):
        return torch.zeros_like(residuals)


class _DummyMlp:
    def __init__(self):
        self.gate = _DummyGate()
        self.experts = _DummyExperts()
        self.shared_experts = _DummySharedExperts()
        self.n_routed_experts = 4

    def route_tokens_to_experts(self, router_logits):
        topk_indices = torch.tensor([[0, 1], [1, 2], [2, 3], [3, 0]])
        topk_weights = torch.ones((4, 2), dtype=torch.float32)
        return topk_indices, topk_weights

    def forward(self, hidden_states):
        return hidden_states


class _DummyLayer:
    def __init__(self):
        self.mlp = _DummyMlp()


class _DummyInnerModel:
    def __init__(self):
        self.layers = [_DummyLayer()]


class _DummyModel:
    def __init__(self):
        self.model = _DummyInnerModel()


def test_glm4_moe_lite_adapter_patches_and_counts():
    adapter = Glm4MoeLiteExpertStatsAdapter(model=_DummyModel(), num_beams=2)
    adapter.attach()
    adapter.enable_collection()

    hidden_states = torch.zeros((2, 2, 8), dtype=torch.float32)
    adapter.layers[0].forward(hidden_states)
    counts = adapter.get_step_counts()

    assert counts[0] is not None
    assert counts[0].shape == (2, 4)
    assert torch.equal(counts[0][0], torch.tensor([1.0, 2.0, 1.0, 0.0]))
    assert torch.equal(counts[0][1], torch.tensor([1.0, 0.0, 1.0, 2.0]))