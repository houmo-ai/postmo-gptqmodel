# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

from gptqmodel.quantization.calibration.generation.runtime import resolve_calibration_source
from gptqmodel.quantization.config import QuantizeConfig


def test_runtime_prefers_in_memory_rows(monkeypatch):
    qcfg = QuantizeConfig(
        calibration_generation={
            "enabled": True,
            "provider": "moe_ebss",
            "output_path": "agent/out.jsonl",
            "nsamples": 2,
            "seqlen": 8,
            "num_beams": 2,
            "allow_override": True,
        }
    )

    class _Provider:
        def generate(self, model, tokenizer, cfg):
            return {"rows": [{"text": "generated"}], "output_path": None}

    monkeypatch.setattr(
        "gptqmodel.quantization.calibration.generation.runtime._register_default_providers",
        lambda: None,
    )
    monkeypatch.setattr(
        "gptqmodel.quantization.calibration.generation.runtime.get_calibration_provider",
        lambda _name: _Provider(),
    )

    resolved = resolve_calibration_source(model=None, tokenizer=None, quantize_config=qcfg, calibration=None)
    assert resolved == [{"text": "generated"}]


def test_runtime_falls_back_to_export_path(monkeypatch):
    qcfg = QuantizeConfig(
        calibration_generation={
            "enabled": True,
            "provider": "moe_ebss",
            "output_path": "agent/out.jsonl",
            "nsamples": 2,
            "seqlen": 8,
            "num_beams": 2,
            "allow_override": True,
        }
    )

    class _Provider:
        def generate(self, model, tokenizer, cfg):
            return {"rows": None, "output_path": "agent/out.jsonl"}

    monkeypatch.setattr(
        "gptqmodel.quantization.calibration.generation.runtime._register_default_providers",
        lambda: None,
    )
    monkeypatch.setattr(
        "gptqmodel.quantization.calibration.generation.runtime.get_calibration_provider",
        lambda _name: _Provider(),
    )
    monkeypatch.setattr(
        "gptqmodel.quantization.calibration.generation.runtime.load_calibration_jsonl",
        lambda _path: [{"text": "loaded"}],
    )

    resolved = resolve_calibration_source(model=None, tokenizer=None, quantize_config=qcfg, calibration=None)
    assert resolved == [{"text": "loaded"}]