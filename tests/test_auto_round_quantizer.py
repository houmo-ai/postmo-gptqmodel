import torch

from gptqmodel.quantization.auto_round import (
    auto_round_loss,
    clamp_auto_round_state_,
    build_input_hessian,
    build_qparams_for_pack,
    build_qparams_from_best_params,
    build_round_g_idx,
    init_auto_round_state,
    quant_tensor_asym,
    quant_tensor_sym_gptq,
    search_scales,
)
from gptqmodel.quantization.config import AutoRoundVersion, FORMAT, QuantizeConfig


def test_search_scales_returns_per_group_scale():
    data = torch.tensor([[1.0, -2.0, 0.5, -0.25]], dtype=torch.float32)
    weights = torch.ones_like(data)

    scales = search_scales(data, bits=4, qw=weights)

    assert scales.shape == (1,)
    assert scales.dtype == torch.float32
    assert torch.all(scales > 0)


def test_quant_tensor_sym_gptq_preserves_shape_and_returns_midpoint_zero():
    tensor = torch.tensor([[1.0, -2.0, 0.5, -0.25]], dtype=torch.float32)
    qdq, scale, zp = quant_tensor_sym_gptq(tensor, bits=4, group_size=2, v=0)

    assert qdq.shape == tensor.shape
    assert scale.shape == (2, 1)
    assert zp.shape == (2, 1)
    assert torch.allclose(zp, torch.full_like(zp, 7.5))


def test_quant_tensor_asym_preserves_shape_and_returns_groupwise_zero_point():
    tensor = torch.tensor([[0.0, 1.0, 2.0, 3.0]], dtype=torch.float32)
    qdq, scale, zp = quant_tensor_asym(tensor, bits=4, group_size=2, v=0)

    assert qdq.shape == tensor.shape
    assert scale.shape == (2, 1)
    assert zp.shape == (2, 1)
    assert torch.all(scale > 0)


def test_build_input_hessian_returns_feature_energy():
    inputs = [
        torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float32),
        torch.tensor([[2.0, 0.0, 1.0]], dtype=torch.float32),
    ]

    hessian = build_input_hessian(inputs)

    assert hessian.shape == (3,)
    assert torch.all(hessian > 0)
    assert hessian[0] > hessian[1]


def test_build_round_g_idx_matches_group_layout():
    g_idx = build_round_g_idx(in_features=8, group_size=2)
    assert torch.equal(g_idx, torch.tensor([0, 0, 1, 1, 2, 2, 3, 3], dtype=torch.int32))


def test_build_qparams_for_pack_reshapes_groupwise_outputs():
    tensor = torch.tensor([[1.0, -2.0, 0.5, -0.25]], dtype=torch.float32)
    qdq, scale, zp = quant_tensor_sym_gptq(tensor, bits=4, group_size=2, v=0)

    q_scales, q_zeros, q_g_idx = build_qparams_for_pack(
        qdq_weight=qdq,
        scale=scale,
        zero=zp,
        rows=1,
        in_features=4,
        group_size=2,
    )

    assert q_scales.shape == (1, 2)
    assert q_zeros.shape == (1, 2)
    assert torch.equal(q_g_idx, torch.tensor([0, 0, 1, 1], dtype=torch.int32))


def test_build_qparams_from_best_params_uses_hessian_init_scale():
    weight = torch.tensor([[1.0, -2.0, 0.5, -0.25]], dtype=torch.float32)
    hessian = torch.tensor([4.0, 2.0, 1.0, 1.0], dtype=torch.float32)
    init_scale = search_scales(weight.reshape(2, 2), bits=4, qw=hessian.reshape(2, 2))

    qdq, scale, zero = build_qparams_from_best_params(
        weight=weight,
        bits=4,
        group_size=2,
        sym=True,
        value=torch.zeros(2, 2),
        min_scale=torch.ones(2),
        max_scale=torch.ones(2),
        init_scale=init_scale,
    )

    assert qdq.shape == weight.shape
    assert scale.shape == (2, 1)
    assert zero.shape == (2, 1)


def test_auto_round_loss_uses_versioned_behavior():
    pred = torch.tensor([[1.0, 2.0, 0.0, 0.0]], dtype=torch.float32)
    ref = torch.zeros_like(pred)
    cfg_v1 = QuantizeConfig(auto_round_enabled=True, bits=4, group_size=4, sym=True, format=FORMAT.GPTQ, auto_round_version=AutoRoundVersion.V1)
    cfg_v2 = QuantizeConfig(auto_round_enabled=True, bits=4, group_size=4, sym=True, format=FORMAT.GPTQ, auto_round_version=AutoRoundVersion.V2)

    loss_v1 = auto_round_loss(pred, ref, qcfg=cfg_v1)
    loss_v2 = auto_round_loss(pred, ref, qcfg=cfg_v2)

    assert loss_v1.item() == 1.25
    assert loss_v2.item() == 1.0


def test_clamp_auto_round_state_respects_version_scale_bounds():
    weight = torch.randn(2, 8, dtype=torch.float32)
    cfg_v1 = QuantizeConfig(auto_round_enabled=True, bits=4, group_size=4, sym=True, format=FORMAT.GPTQ, auto_round_version=AutoRoundVersion.V1)
    cfg_v2 = QuantizeConfig(auto_round_enabled=True, bits=4, group_size=4, sym=True, format=FORMAT.GPTQ, auto_round_version=AutoRoundVersion.V2)

    state_v1 = init_auto_round_state(weight, cfg_v1)
    state_v2 = init_auto_round_state(weight, cfg_v2)

    with torch.no_grad():
        state_v1.max_scale.fill_(1.5)
        state_v2.max_scale.fill_(1.5)

    clamp_auto_round_state_(state_v1)
    clamp_auto_round_state_(state_v2)

    assert torch.all(state_v1.max_scale <= 1.0)
    assert torch.all(state_v2.max_scale <= 2.0)
    assert torch.any(torch.isclose(state_v2.max_scale, torch.tensor(1.5)))
