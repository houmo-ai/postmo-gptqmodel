# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

import importlib.util
import json
import sys

import pytest
import torch
from safetensors.torch import save_file
from torch import nn
from transformers import PretrainedConfig, PreTrainedModel

from gptqmodel.utils.model import (
    _try_load_checkpoint_with_transformers_conversion,
)


class _TinyConfig(PretrainedConfig):
    model_type = "gptqmodel_conversion_test"


class _TinyModel(PreTrainedModel):
    config_class = _TinyConfig
    base_model_prefix = ""

    def __init__(self, config):
        super().__init__(config)
        self.frontend = nn.Linear(2, 2, bias=False)


class _MixedTinyModel(_TinyModel):
    def __init__(self, config):
        super().__init__(config)
        self.register_buffer("exact_buffer", torch.zeros(2))


@pytest.mark.skipif(
    importlib.util.find_spec("transformers.core_model_loading") is None,
    reason="Transformers checkpoint conversion loader is unavailable",
)
def test_transformers_checkpoint_conversion_loads_renamed_frontend(tmp_path):
    checkpoint = tmp_path / "model.safetensors"
    expected = torch.arange(4, dtype=torch.float32).reshape(2, 2)
    save_file({"legacy_frontend.weight": expected}, checkpoint)

    model = _TinyModel(_TinyConfig())
    with torch.no_grad():
        model.frontend.weight.fill_(-1)

    converted = _try_load_checkpoint_with_transformers_conversion(
        model,
        checkpoint=str(checkpoint),
        device_map={"": "cpu"},
        dtype=torch.float32,
        offload_buffers=False,
        hf_key_mapping={"legacy_frontend.weight": "frontend.weight"},
    )

    assert converted is True
    torch.testing.assert_close(model.frontend.weight, expected)


@pytest.mark.skipif(
    importlib.util.find_spec("transformers.core_model_loading") is None,
    reason="Transformers checkpoint conversion loader is unavailable",
)
def test_transformers_checkpoint_conversion_keeps_exact_key_fast_path(tmp_path):
    checkpoint = tmp_path / "model.safetensors"
    save_file({"frontend.weight": torch.ones(2, 2)}, checkpoint)

    model = _TinyModel(_TinyConfig())
    converted = _try_load_checkpoint_with_transformers_conversion(
        model,
        checkpoint=str(checkpoint),
        device_map={"": "cpu"},
        dtype=torch.float32,
        offload_buffers=False,
        # A family-wide mapping may target another model variant.  Match
        # Transformers' fallback to the existing exact runtime key instead of
        # forcing conversion merely because the source pattern matched.
        hf_key_mapping={"frontend.weight": "missing_frontend.weight"},
    )

    assert converted is False


@pytest.mark.skipif(
    importlib.util.find_spec("transformers.core_model_loading") is None,
    reason="Transformers checkpoint conversion loader is unavailable",
)
def test_transformers_checkpoint_conversion_preserves_mixed_exact_keys(tmp_path):
    checkpoint = tmp_path / "model.safetensors"
    expected_frontend = torch.arange(4, dtype=torch.float32).reshape(2, 2)
    expected_exact = torch.tensor([3.0, 4.0], dtype=torch.float32)
    save_file(
        {
            "legacy_frontend.weight": expected_frontend,
            "exact_buffer": expected_exact,
        },
        checkpoint,
    )

    model = _MixedTinyModel(_TinyConfig())
    converted = _try_load_checkpoint_with_transformers_conversion(
        model,
        checkpoint=checkpoint,
        device_map={"": "cpu"},
        dtype=torch.float32,
        offload_buffers=False,
        hf_key_mapping={"legacy_frontend.weight": "frontend.weight"},
    )

    assert converted is True
    torch.testing.assert_close(model.frontend.weight, expected_frontend)
    torch.testing.assert_close(model.exact_buffer, expected_exact)


@pytest.mark.skipif(
    importlib.util.find_spec("transformers.core_model_loading") is None,
    reason="Transformers checkpoint conversion loader is unavailable",
)
def test_transformers_checkpoint_conversion_runs_exact_key_converter(
    monkeypatch,
    tmp_path,
):
    from transformers.conversion_mapping import Transpose, WeightConverter
    import transformers.modeling_utils

    checkpoint = tmp_path / "model.safetensors"
    source = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    save_file({"frontend.weight": source}, checkpoint)

    def get_conversion_mapping(model, key_mapping=None):
        del model, key_mapping
        return [
            WeightConverter(
                source_patterns="frontend.weight",
                target_patterns="frontend.weight",
                operations=[Transpose()],
            )
        ]

    monkeypatch.setattr(
        transformers.modeling_utils,
        "get_model_conversion_mapping",
        get_conversion_mapping,
    )
    model = _TinyModel(_TinyConfig())
    converted = _try_load_checkpoint_with_transformers_conversion(
        model,
        checkpoint=str(checkpoint),
        device_map={"": "cpu"},
        dtype=torch.float32,
        offload_buffers=False,
        hf_key_mapping=None,
    )

    assert converted is True
    torch.testing.assert_close(model.frontend.weight, source.T)


@pytest.mark.skipif(
    importlib.util.find_spec("transformers.core_model_loading") is None,
    reason="Transformers checkpoint conversion loader is unavailable",
)
def test_transformers_checkpoint_conversion_loads_multiple_shards(tmp_path):
    expected_frontend = torch.arange(4, dtype=torch.float32).reshape(2, 2)
    expected_exact = torch.tensor([3.0, 4.0], dtype=torch.float32)
    shard_1 = "model-00001-of-00002.safetensors"
    shard_2 = "model-00002-of-00002.safetensors"
    save_file({"legacy_frontend.weight": expected_frontend}, tmp_path / shard_1)
    save_file({"exact_buffer": expected_exact}, tmp_path / shard_2)
    checkpoint = tmp_path / "model.safetensors.index.json"
    checkpoint.write_text(
        json.dumps(
            {
                "metadata": {
                    "total_size": (
                        expected_frontend.numel() * expected_frontend.element_size()
                        + expected_exact.numel() * expected_exact.element_size()
                    )
                },
                "weight_map": {
                    "legacy_frontend.weight": shard_1,
                    "exact_buffer": shard_2,
                },
            }
        ),
        encoding="utf-8",
    )

    model = _MixedTinyModel(_TinyConfig())
    converted = _try_load_checkpoint_with_transformers_conversion(
        model,
        checkpoint=checkpoint,
        device_map={"": "cpu"},
        dtype=torch.float32,
        offload_buffers=False,
        hf_key_mapping={"legacy_frontend.weight": "frontend.weight"},
    )

    assert converted is True
    torch.testing.assert_close(model.frontend.weight, expected_frontend)
    torch.testing.assert_close(model.exact_buffer, expected_exact)


@pytest.mark.skipif(
    importlib.util.find_spec("transformers.core_model_loading") is None,
    reason="Transformers checkpoint conversion loader is unavailable",
)
def test_transformers_checkpoint_conversion_without_mapping_uses_fast_path(
    tmp_path,
):
    checkpoint = tmp_path / "model.safetensors"
    save_file({"frontend.weight": torch.ones(2, 2)}, checkpoint)

    converted = _try_load_checkpoint_with_transformers_conversion(
        _TinyModel(_TinyConfig()),
        checkpoint=checkpoint,
        device_map={"": "cpu"},
        dtype=torch.float32,
        offload_buffers=False,
        hf_key_mapping=None,
    )

    assert converted is False


def test_transformers_checkpoint_conversion_unavailable_uses_fast_path(
    monkeypatch,
    tmp_path,
):
    checkpoint = tmp_path / "model.safetensors"
    save_file({"legacy_frontend.weight": torch.ones(2, 2)}, checkpoint)
    monkeypatch.setitem(sys.modules, "transformers.core_model_loading", None)

    converted = _try_load_checkpoint_with_transformers_conversion(
        _TinyModel(_TinyConfig()),
        checkpoint=checkpoint,
        device_map={"": "cpu"},
        dtype=torch.float32,
        offload_buffers=False,
        hf_key_mapping={"legacy_frontend.weight": "frontend.weight"},
    )

    assert converted is False
