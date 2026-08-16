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


def materialization_path(capture, layers):
    capture.cache_inputs(
        layers=layers,
        calibration_data=[],
        use_cache=False,
    )
    return capture.gptq_model.shell_module_materialize.call_args.kwargs["module_path"]


def test_cache_inputs_resolves_layer_identity_from_model_tree():
    model = FakeModel()
    layer = model.layers[0]
    capture, layers, _ = make_capture(layer, model=model)
    assert materialization_path(capture, layers) == "layers.0"


def test_orphan_class_name_is_not_used_as_checkpoint_path():
    layer = FakeLayer()
    capture, layers, _ = make_capture(layer, model=FakeModel())
    assert materialization_path(capture, layers) is None


def test_cache_inputs_falls_back_to_full_name():
    layer = FakeLayer()
    layer.full_name = "custom.path.layer_0"
    capture, layers, _ = make_capture(layer, model=FakeModel())
    assert materialization_path(capture, layers) == "custom.path.layer_0"


def test_model_tree_path_wins_over_stale_full_name():
    layer = FakeLayer()
    layer.full_name = "legacy.path.layer_0"
    model = FakeModel()
    model.layers[0] = layer
    capture, layers, _ = make_capture(layer, model=model)
    assert materialization_path(capture, layers) == "layers.0"
