from __future__ import annotations

import torch

from gptqmodel.nn_modules.qlinear.tritonv2 import TritonV2QuantLinear


def test_triton_forward_preserves_quant_matmul_output_dtype(monkeypatch) -> None:
    """Do not promote an autocast GEMM result back to an FP32 input dtype."""

    module = TritonV2QuantLinear(
        bits=4,
        group_size=64,
        desc_act=False,
        sym=True,
        in_features=64,
        out_features=64,
        pack_dtype=torch.int32,
        bias=False,
    ).eval()

    def fake_apply(inputs, *_args):
        return torch.zeros(
            (inputs.shape[0], module.out_features),
            dtype=torch.float16,
            device=inputs.device,
        )

    from gptqmodel.nn_modules.triton_utils.dequant import QuantLinearFunction

    monkeypatch.setattr(QuantLinearFunction, "apply", fake_apply)
    output = module(torch.zeros((2, 64), dtype=torch.float32))

    assert output.dtype is torch.float16
    assert output.shape == (2, 64)


def test_triton_forward_casts_fp32_input_to_autocast_dtype(monkeypatch) -> None:
    """FP32 RMSNorm outputs must not force FP32 weight dequantization."""

    module = TritonV2QuantLinear(
        bits=4,
        group_size=64,
        desc_act=False,
        sym=True,
        in_features=64,
        out_features=64,
        pack_dtype=torch.int32,
        bias=False,
    ).eval()
    observed = {}

    def fake_apply(inputs, *_args):
        observed["input_dtype"] = inputs.dtype
        return torch.zeros(
            (inputs.shape[0], module.out_features),
            dtype=inputs.dtype,
            device=inputs.device,
        )

    from gptqmodel.nn_modules.triton_utils.dequant import QuantLinearFunction

    monkeypatch.setattr(QuantLinearFunction, "apply", fake_apply)
    original_is_enabled = torch.is_autocast_enabled
    original_get_dtype = torch.get_autocast_dtype
    monkeypatch.setattr(
        torch,
        "is_autocast_enabled",
        lambda device_type=None: True if device_type == "cpu" else original_is_enabled(device_type),
    )
    monkeypatch.setattr(
        torch,
        "get_autocast_dtype",
        lambda device_type: torch.float16 if device_type == "cpu" else original_get_dtype(device_type),
    )

    output = module(torch.zeros((2, 64), dtype=torch.float32))

    assert observed["input_dtype"] is torch.float16
    assert output.dtype is torch.float16
