# SPDX-FileCopyrightText: 2024-2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import MagicMock

import torch
from torch import nn

from gptqmodel.looper.stage_inputs_capture import StageInputsCapture


class FakeLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(2, 2)


class FakeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([FakeLayer()])
        self.config = MagicMock(model_type="test")


class ModelWhosePathsMustNotBeInspected(FakeModel):
    def named_modules(self, *args, **kwargs):  # pragma: no cover - must not be called
        raise AssertionError("input capture must not resolve turtle checkpoint paths")


def make_capture(layer, *, model=None):
    quantize_config = MagicMock()
    quantize_config.device = torch.device("cpu")

    gptq_model = MagicMock()
    gptq_model.model = model or FakeModel()
    gptq_model.quantize_config = quantize_config
    gptq_model.get_base_modules.return_value = []
    gptq_model.shell_module_materialize.return_value = layer
    gptq_model.ATTENTION_MASKS_REQUIRED_FOR_INPUT = False
    gptq_model.quant_region_timer = None

    looper = MagicMock()
    looper.gptq_model = gptq_model
    return StageInputsCapture(looper), [layer], gptq_model


def test_cache_inputs_delegates_path_resolution_to_materializer():
    layer = FakeLayer()
    capture, layers, gptq_model = make_capture(
        layer,
        model=ModelWhosePathsMustNotBeInspected(),
    )

    capture.cache_inputs(
        layers=layers,
        calibration_data=[],
        use_cache=False,
    )
    assert gptq_model.shell_module_materialize.call_args.kwargs == {
        "target_submodule": layer,
        "device": torch.device("cpu"),
    }
