from types import SimpleNamespace

import pytest

from scripts_qwen35moe.quantize_moe import resolve_device_map


class _DummyModel:
    def __init__(self, num_layers=40):
        self.config = SimpleNamespace(text_config=SimpleNamespace(num_hidden_layers=num_layers))


@pytest.fixture
def cuda_2way(monkeypatch):
    monkeypatch.setattr("torch.cuda.is_available", lambda: True)
    monkeypatch.setattr("torch.cuda.device_count", lambda: 2)


def test_balanced_defaults_to_blockwise_auto(cuda_2way):
    resolved, summary = resolve_device_map("balanced", _DummyModel())

    assert resolved == "0,1"
    assert "block-wise split via AutoRound auto mapper" in summary
    assert "cuda:0, cuda:1" in summary


def test_balanced_manual_keeps_whole_layer_chunks(cuda_2way):
    resolved, summary = resolve_device_map("balanced:manual", _DummyModel())

    assert isinstance(resolved, dict)
    assert resolved[r"^model\.language_model\.embed_tokens(?:\.|$)"] == "cuda:0"
    assert resolved[r"^model\.language_model\.norm(?:\.|$)"] == "cuda:1"
    assert resolved[r"^lm_head(?:\.|$)"] == "cuda:1"
    assert "balanced manual split" in summary
