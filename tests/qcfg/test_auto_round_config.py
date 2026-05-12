# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import torch

from gptqmodel.quantization import FORMAT, QUANT_METHOD
from gptqmodel.quantization.config import AutoRoundVersion, METHOD, QuantizeConfig
from gptqmodel.utils import BACKEND
from gptqmodel.utils.importer import select_quant_linear
from gptqmodel.models.auto import _is_supported_quantization_config
from gptqmodel.nn_modules.qlinear.torch import TorchQuantLinear


def test_autoround_alias_and_config_roundtrip():
    cfg = QuantizeConfig(auto_round_enabled=True, 
        bits=4,
        group_size=128,
        sym=True,
        format=FORMAT.GPTQ,
        iters=32,
        lr=0.1,
        auto_round_version=AutoRoundVersion.V2,
    )

    payload = cfg.to_dict()

    assert QUANT_METHOD.AUTO_ROUND == METHOD.AUTO_ROUND
    assert payload["quant_method"] == METHOD.AUTO_ROUND
    assert payload["checkpoint_format"] == FORMAT.GPTQ
    assert payload["iters"] == 32
    assert payload["lr"] == 0.1
    assert payload["auto_round_version"] == "v2"
    assert payload["enable_alg_ext"] is True

    restored = QuantizeConfig.from_quant_config(payload)
    assert restored.quant_method == METHOD.AUTO_ROUND
    assert restored.iters == 32
    assert restored.lr == 0.1
    assert restored.auto_round_version == AutoRoundVersion.V2
    assert restored.enable_alg_ext is True


def test_autoround_selects_gptq_compatible_kernel():
    kernel = select_quant_linear(
        bits=4,
        group_size=128,
        desc_act=False,
        sym=True,
        backend=BACKEND.TORCH,
        device="cpu",
        format=FORMAT.GPTQ,
        quant_method=METHOD.AUTO_ROUND,
        pack=True,
        pack_dtype=torch.int32,
    )

    assert kernel is TorchQuantLinear


def test_autoround_quantization_config_is_supported_for_load():
    config = SimpleNamespace(
        quantization_config={
            "quant_method": METHOD.AUTO_ROUND,
            "checkpoint_format": FORMAT.GPTQ,
        }
    )

    assert _is_supported_quantization_config(config) is True
