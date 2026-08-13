import torch
import torch.nn as nn

from gptqmodel.nn_modules.qlinear.torch import TorchQuantLinear


def _unpack_rows(packed: torch.Tensor, bits: int) -> torch.Tensor:
    pack_factor = 32 // bits
    unsigned = packed.to(torch.int64) & 0xFFFF_FFFF
    return torch.stack(
        [((unsigned >> (index * bits)) & ((1 << bits) - 1)).to(torch.int32) for index in range(pack_factor)],
        dim=1,
    ).reshape(packed.shape[0] * pack_factor, packed.shape[1])


def _make_out_of_range_case() -> tuple[nn.Linear, torch.Tensor, torch.Tensor, torch.Tensor]:
    linear = nn.Linear(64, 32, bias=False, dtype=torch.float32)
    with torch.no_grad():
        # With scale=1 and zero=8 these map to -92 and 108.  Both must
        # saturate before W4 packing instead of wrapping modulo 16.
        linear.weight[:, :32].fill_(-100.0)
        linear.weight[:, 32:].fill_(100.0)
    scales = torch.ones((32, 1), dtype=torch.float32)
    zeros = torch.full((32, 1), 8.0, dtype=torch.float32)
    g_idx = torch.zeros(64, dtype=torch.int32)
    return linear, scales, zeros, g_idx


def _make_qlinear() -> TorchQuantLinear:
    return TorchQuantLinear(
        bits=4,
        group_size=64,
        sym=True,
        desc_act=False,
        in_features=64,
        out_features=32,
        bias=False,
        pack_dtype=torch.int32,
        register_buffers=False,
    )


def _assert_saturated(module: TorchQuantLinear) -> None:
    unpacked = _unpack_rows(module.qweight, bits=4)
    assert torch.equal(unpacked[:32], torch.zeros_like(unpacked[:32]))
    assert torch.equal(unpacked[32:], torch.full_like(unpacked[32:], 15))


def test_python_block_packer_clamps_before_bitpacking(monkeypatch):
    monkeypatch.setenv("GPTQMODEL_DISABLE_PACK_EXT", "1")
    module = _make_qlinear()
    module.pack_block(*_make_out_of_range_case())
    _assert_saturated(module)


def test_legacy_packer_clamps_before_bitpacking():
    module = _make_qlinear()
    module.pack_original(*_make_out_of_range_case())
    _assert_saturated(module)
