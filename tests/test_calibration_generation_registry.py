# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

import pytest

from gptqmodel.quantization.calibration.generation.base import (
    CalibrationProviderBase,
    get_calibration_provider,
    register_calibration_provider,
)


class _DummyProvider(CalibrationProviderBase):
    name = "dummy"

    def generate(self, model, tokenizer, cfg):
        return "agent/dummy.jsonl"


def test_register_and_get_provider():
    register_calibration_provider(_DummyProvider())
    provider = get_calibration_provider("dummy")
    assert provider.name == "dummy"


def test_unknown_provider_should_fail():
    with pytest.raises(ValueError, match="not registered"):
        get_calibration_provider("not_exists")