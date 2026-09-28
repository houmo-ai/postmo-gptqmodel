"""Preserve every channel when dimensions align to a packed word, not 32."""

import math

import pytest
import torch
from safetensors.torch import load_file, save_file

from gptqmodel.nn_modules.qlinear.torch import TorchQuantLinear


@pytest.mark.parametrize("bits", [2, 4, 8])
@pytest.mark.parametrize("shape", [(64, 48), (48, 64), (64, 4304), (4304, 64), (64, 64)])
@pytest.mark.parametrize("packer", ["pack_original", "pack_block"])
def test_pack_save_reload_preserves_tail_channels(bits, shape, packer, tmp_path, monkeypatch):
    monkeypatch.setenv("GPTQMODEL_DISABLE_PACK_EXT", "1")
    monkeypatch.setenv("GPTQ_TORCH_TRITON_DEQUANT", "0")
    in_features, out_features = shape
    groups = math.ceil(in_features / 64)
    kwargs = dict(bits=bits, group_size=64, sym=False, desc_act=False,
                  in_features=in_features, out_features=out_features, bias=True)
    linear = torch.nn.Linear(in_features, out_features, dtype=torch.float16)
    codes = torch.arange(out_features * in_features).reshape(out_features, in_features) % (2**bits)
    zeros = (torch.arange(out_features * groups).reshape(out_features, groups) % (2**bits)).float()
    scales = torch.full_like(zeros, 0.125)
    g_idx = torch.arange(in_features, dtype=torch.int32) // 64
    expected = ((codes - zeros[:, g_idx.long()]) * scales[:, g_idx.long()]).half()
    with torch.no_grad():
        linear.weight.copy_(expected)
        linear.bias.copy_(torch.arange(out_features) % 7)
    packed = TorchQuantLinear(**kwargs)
    getattr(packed, packer)(linear, scales, zeros, g_idx)
    assert packed.qweight.shape == (in_features * bits // 32, out_features)
    assert packed.qzeros.shape == (groups, out_features * bits // 32)
    path = tmp_path / "model.safetensors"
    save_file(packed.state_dict(), path)
    # Fresh constructor buffers exercise the loader's expected tensor shapes.
    restored = TorchQuantLinear(**kwargs)
    restored.load_state_dict(load_file(path), strict=True)
    for name, value in packed.state_dict().items():
        assert torch.equal(value, restored.state_dict()[name]), name
    torch.testing.assert_close(restored.dequantize_weight().T, expected, rtol=0, atol=0)
    # Exercise both the regular and eval cached torch unpack paths.
    restored.eval()
    torch.testing.assert_close(restored.dequantize_weight().T, expected, rtol=0, atol=0)
    inputs = torch.zeros(2, in_features, dtype=torch.float16)
    inputs[0, -1] = 1
    inputs[1, -16:] = 0.125
    torch.testing.assert_close(restored(inputs), torch.nn.functional.linear(inputs, expected, linear.bias),
                               rtol=0, atol=0)


@pytest.mark.parametrize("packer", ["pack_original", "pack_block"])
def test_unaligned_channels_are_rejected_instead_of_truncated(packer):
    layer = torch.nn.Linear(65, 32, bias=False)
    module = TorchQuantLinear(bits=8, group_size=64, sym=True, desc_act=False,
                              in_features=65, out_features=32, bias=False)
    with pytest.raises(ValueError, match="divisible by 4"):
        getattr(module, packer)(layer, torch.ones(32, 2), torch.full((32, 2), 128.),
                                torch.arange(65, dtype=torch.int32) // 64)
