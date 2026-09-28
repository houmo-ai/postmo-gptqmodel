"""Mixed-bit zero-point conversion must use each physical layer's width."""

import pytest
import torch

from gptqmodel.nn_modules.qlinear.torch import TorchQuantLinear
from gptqmodel.quantization.config import QuantizeConfig
from gptqmodel.utils.model import convert_gptq_v1_to_v2_format, convert_gptq_v2_to_v1_format


@pytest.mark.parametrize("global_bits", [4, 8])
def test_mixed_symmetric_zero_points_round_trip(global_bits):
    model = torch.nn.ModuleList([
        TorchQuantLinear(bits=bits, group_size=64, sym=True, desc_act=False,
                         in_features=64, out_features=32, bias=False)
        for bits in (2, 4, 8)
    ])
    originals = []
    for layer in model:
        word = sum((2 ** (layer.bits - 1)) << shift for shift in range(0, 32, layer.bits))
        signed = word - 2**32 if word >= 2**31 else word
        layer.qzeros.fill_(signed)
        layer.qzero_format(format=2)
        originals.append(layer.qzeros.clone())
    config = QuantizeConfig(bits=global_bits, group_size=64, sym=True)
    convert_gptq_v2_to_v1_format(model, config, TorchQuantLinear)
    for layer in model:
        codes = (layer.qzeros.long().unsqueeze(-1) >> torch.arange(0, 32, layer.bits)) & layer.maxq
        assert torch.all(codes == 2 ** (layer.bits - 1) - 1)
    convert_gptq_v1_to_v2_format(model, config, TorchQuantLinear)
    for layer, expected in zip(model, originals):
        torch.testing.assert_close(layer.qzeros, expected, rtol=0, atol=0)
