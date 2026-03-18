# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

from gptqmodel.quantization.config import QuantizeConfig
from gptqmodel.quantization.calibration.generation.runtime import resolve_calibration_source


class _DummyProvider:
    name = "dummy"

    def generate(self, model, tokenizer, cfg):
        return "agent/dummy.jsonl"


def test_resolve_calibration_source_disabled_returns_original():
    calibration = [{"text": "x"}]
    qcfg = QuantizeConfig()
    resolved = resolve_calibration_source(model=None, tokenizer=None, quantize_config=qcfg, calibration=calibration)
    assert resolved is calibration


def test_resolve_calibration_source_enabled_uses_provider(monkeypatch):
    qcfg = QuantizeConfig(
        calibration_generation={
            "enabled": True,
            "provider": "dummy",
            "output_path": "agent/out.jsonl",
            "nsamples": 2,
            "seqlen": 8,
            "num_beams": 2,
            "allow_override": True,
        }
    )

    monkeypatch.setattr(
        "gptqmodel.quantization.calibration.generation.runtime._register_default_providers",
        lambda: None,
    )
    monkeypatch.setattr(
        "gptqmodel.quantization.calibration.generation.runtime.get_calibration_provider",
        lambda _name: _DummyProvider(),
    )
    monkeypatch.setattr(
        "gptqmodel.quantization.calibration.generation.runtime.load_calibration_jsonl",
        lambda _path: [{"text": "generated"}],
    )

    resolved = resolve_calibration_source(model=None, tokenizer=None, quantize_config=qcfg, calibration=[{"text": "old"}])
    assert resolved == [{"text": "generated"}]
