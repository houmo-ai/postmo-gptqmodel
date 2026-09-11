# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

import pytest

from gptqmodel.quantization.config import QuantizeConfig


def test_calibration_generation_default_disabled():
    cfg = QuantizeConfig()
    assert cfg.calibration_generation is None


def test_calibration_generation_roundtrip():
    cfg = QuantizeConfig(
        calibration_generation={
            "enabled": True,
            "provider": "moe_ebss",
            "output_path": "agent/tmp-calib.jsonl",
            "nsamples": 8,
            "seqlen": 64,
            "num_beams": 2,
            "overwrite": True,
        }
    )
    payload = cfg.to_dict()
    reloaded = QuantizeConfig.from_quant_config(payload)
    assert reloaded.calibration_generation is not None
    assert reloaded.calibration_generation.enabled is True
    assert reloaded.calibration_generation.provider == "moe_ebss"


def test_calibration_generation_invalid_positive_fields():
    with pytest.raises(ValueError, match="nsamples"):
        QuantizeConfig(
            calibration_generation={
                "enabled": True,
                "provider": "moe_ebss",
                "output_path": "agent/tmp-calib.jsonl",
                "nsamples": 0,
                "seqlen": 64,
                "num_beams": 2,
            }
        )