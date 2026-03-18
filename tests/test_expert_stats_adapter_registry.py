# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

import pytest

from gptqmodel.quantization.calibration.generation.adapters.qwen3_moe import Qwen3MoeExpertStatsAdapter
from gptqmodel.quantization.calibration.generation.adapters.registry import (
    create_expert_stats_adapter,
    get_expert_stats_adapter_cls,
)


class _Config:
    model_type = "qwen3_moe"


class _Layer:
    def __init__(self):
        self.mlp = None


class _Model:
    def __init__(self):
        self.config = _Config()
        self.layers = [_Layer()]


def test_get_expert_stats_adapter_cls_known_type():
    assert get_expert_stats_adapter_cls("qwen3_moe") is Qwen3MoeExpertStatsAdapter


def test_get_expert_stats_adapter_cls_unknown_type_raises():
    with pytest.raises(ValueError, match="unsupported"):
        get_expert_stats_adapter_cls("unknown")


def test_create_expert_stats_adapter_uses_model_config_type():
    adapter = create_expert_stats_adapter(_Model(), num_beams=4)
    assert isinstance(adapter, Qwen3MoeExpertStatsAdapter)
