# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import torch

from gptqmodel.quantization.calibration.generation.moe_ebss_provider import MoeEbssProvider


class _DummyModel:
    def __init__(self):
        self.config = SimpleNamespace(vocab_size=8)
        self.device = torch.device("cpu")

    def generate(self, *args, **kwargs):
        return torch.tensor([[1, 2, 3]])


class _DummyTokenizer:
    def batch_decode(self, outputs, skip_special_tokens=True):
        return ["dummy text"]


class _DummyAdapter:
    def enable_collection(self):
        return None

    def disable_collection(self):
        return None


def test_moe_ebss_provider_native_returns_rows_and_exports(monkeypatch, tmp_path):
    provider = MoeEbssProvider()
    cfg = SimpleNamespace(
        output_path=str(tmp_path / "calib.jsonl"),
        overwrite=True,
        nsamples=2,
        seqlen=8,
        num_beams=2,
        model_patch="qwen3_moe",
    )
    model = SimpleNamespace(config=SimpleNamespace(_name_or_path="dummy", model_type="qwen3_moe"))

    monkeypatch.setattr(provider, "_load_generation_model", lambda model_path, resolved_model_type, num_beams=4: _DummyModel())
    monkeypatch.setattr(
        "gptqmodel.quantization.calibration.generation.moe_ebss_provider.create_expert_stats_adapter",
        lambda generation_model, num_beams, model_type=None: _DummyAdapter(),
    )
    monkeypatch.setattr(
        "gptqmodel.quantization.calibration.generation.moe_ebss_provider.bind_local_ebss_beam_search",
        lambda generation_model, beam_search_fn, adapter: None,
    )
    monkeypatch.setattr(
        "gptqmodel.quantization.calibration.generation.moe_ebss_provider.restore_local_ebss_beam_search",
        lambda generation_model, original_beam_search: None,
    )
    monkeypatch.setattr(
        "gptqmodel.quantization.calibration.generation.moe_ebss_provider.random.sample",
        lambda seq, n: [0, 1],
    )
    monkeypatch.setattr(
        "gptqmodel.quantization.calibration.generation.moe_ebss_provider.random.randint",
        lambda low, high: 3,
    )

    artifact = provider.generate(model=model, tokenizer=_DummyTokenizer(), cfg=cfg)
    assert len(artifact["rows"]) == 2
    assert artifact["rows"][0] == {"text": "dummy text"}
    assert artifact["output_path"] == str(tmp_path / "calib.jsonl")
