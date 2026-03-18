# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

import torch

from gptqmodel.quantization.calibration.generation.adapters.qwen3_next import Qwen3NextExpertStatsAdapter


class _DummyMlp:
    def __init__(self):
        self.save_expert_cnt = False
        self.top_cnt = torch.tensor([[2.0, 1.0], [1.0, 2.0]])

    def clear_expert_cnt(self):
        self.top_cnt.zero_()


class _DummyLayer:
    def __init__(self):
        self.mlp = _DummyMlp()


class _DummyModel:
    def __init__(self):
        self.layers = [_DummyLayer()]


def test_qwen3_next_adapter_enable_get_clear_disable(monkeypatch):
    monkeypatch.setattr(
        "gptqmodel.quantization.calibration.generation.adapters.qwen3_next.patch_qwen3_next_sparse_moe_block",
        lambda: None,
    )

    adapter = Qwen3NextExpertStatsAdapter(model=_DummyModel(), num_beams=2)
    adapter.attach()
    adapter.enable_collection()

    assert adapter.layers[0].save_expert_cnt is True
    counts = adapter.get_step_counts()
    assert len(counts) == 1
    assert torch.equal(counts[0], torch.tensor([[2.0, 1.0], [1.0, 2.0]]))

    adapter.clear_step_counts()
    assert torch.equal(adapter.layers[0].top_cnt, torch.zeros((2, 2)))

    adapter.disable_collection()
    assert adapter.layers[0].save_expert_cnt is False