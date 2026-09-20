"""Mathematics, tuning state and public recipe of the XH Hessian mode."""
import json
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch

from auto_round.xh_hessian import (
    GroupHessian, XHHessianConfig, XHHessianLinear, batch_mask, configure, search_positive_scales,
)
from gptqmodel.recipes.qwen35_autoround import quantize_qwen35_autoround


def test_group_hessian_matches_output_error_and_ignores_padding():
    torch.manual_seed(7)
    x = torch.randn(3, 9, 128)
    x[..., 1] = 2 * x[..., 0]  # Correlation must contribute to the score.
    mask = torch.ones(3, 9, dtype=torch.bool)
    mask[0, -3:] = False
    stats = GroupHessian(128, token_chunk=4)
    stats.add(x[:1], mask[:1])
    stats.add(x[1:], mask[1:])
    h = stats.finish()
    e = torch.randn(5, 64)
    actual = (x[mask][:, :64] @ e.T).square().mean(0)
    torch.testing.assert_close((e @ h[0] * e).sum(-1), actual)
    assert stats.count == 24
    assert not torch.allclose((e.square() * h[0].diag()).sum(-1), actual)


def test_scale_search_scores_real_group_outputs_and_keeps_xh_candidate():
    torch.manual_seed(42)
    w, x = torch.randn(7, 128) * 0.04, torch.randn(37, 128)
    x[:, 1] = 3 * x[:, 0]
    stats = GroupHessian(128)
    stats.add(x)
    s, report = search_positive_scales(w, stats.finish(), XHHessianConfig(candidates=9, row_chunk=3))
    assert torch.isfinite(s).all() and (s > 0).all()
    assert torch.equal(s, s.half().float())
    error = ((w.reshape(7, 2, 64) / s[..., None]).round().clamp(-8, 7) * s[..., None]
             - w.reshape(7, 2, 64))
    direct = sum((x[:, g*64:(g+1)*64] @ error[:, g].T).square().mean(0).sum()
                 for g in range(2))
    assert report['searched_score'] == pytest.approx(direct.item(), rel=2e-6)
    assert report['searched_score'] <= report['xh_initial_score']


def test_positive_scale_gradient_checkpoint_restore_and_export_grid():
    torch.manual_seed(8)
    layer = torch.nn.Linear(128, 64, bias=False)
    layer.requires_grad_(False)
    # Earlier GPTQ loader tests can disable torch.nn.init globally. Make the
    # numerical fixture explicit instead of depending on Linear's initializer.
    layer.weight.copy_(torch.randn(64, 128) * 0.04)
    for key, value in dict(bits=4, group_size=64, data_type='int_sym_xh', sym=True,
                           act_bits=16, scale_dtype=torch.float16).items():
        setattr(layer, key, value)
    inputs = torch.randn(3, 11, 128)
    layer._xh_hessian = GroupHessian(128)
    layer._xh_hessian.add(inputs)
    wrapper = XHHessianLinear(layer, xh_config=XHHessianConfig(candidates=3), report=[])
    loss = (wrapper(inputs) - layer(inputs)).square().mean()
    loss.backward()
    assert wrapper.log_scale.grad.abs().sum() > 0
    assert wrapper.value.grad.abs().sum() > 0
    assert set(wrapper.params) == {'value', 'log_scale'}
    with torch.no_grad():
        wrapper.log_scale.fill_(0.1)
        wrapper.value.fill_(0.05)
    expected = wrapper(inputs).detach()
    snapshot = {k: v.detach().clone() for k, v in wrapper.params.items()}
    with torch.no_grad():
        wrapper.log_scale.fill_(-0.2)
        wrapper.value.fill_(-0.1)
    restored = wrapper.unwrapper(snapshot)
    torch.testing.assert_close(restored(inputs), expected, rtol=0, atol=0)
    assert (restored.scale > 0).all() and (restored.zp == 8).all()
    integer = restored.weight.reshape(64, 2, 64) / restored.scale[..., None]
    torch.testing.assert_close(integer, integer.round(), rtol=0, atol=1e-6)
    assert integer.min() >= -8 and integer.max() <= 7


def test_hessian_mask_excludes_holdout_and_padding():
    owner = SimpleNamespace(xh_hessian_config=XHHessianConfig(validation_samples=2),
                            batch_dim=0, attention_mask=[])
    mask = batch_mask(owner, torch.tensor([1, 3, 4]), 5, torch.ones(3, 4, 64))
    assert mask.tolist() == [[True]*4, [False]*4, [False]*4]
    owner.attention_mask = [torch.tensor([[1, 1, 0, 0]])] * 5
    mask = batch_mask(owner, torch.tensor([0, 4]), 5, torch.ones(2, 4, 64))
    assert mask.tolist() == [[True, True, False, False], [False]*4]


def test_zero_and_singular_groups_need_no_inverse_or_damping():
    w = torch.zeros(3, 64)
    h = torch.ones(1, 64, 64)  # Rank one, non-invertible.
    s, score = search_positive_scales(w, h, XHHessianConfig(candidates=3))
    assert torch.isfinite(s).all() and (s > 0).all()
    assert score['searched_score'] == score['xh_initial_score'] == 0


@pytest.mark.parametrize('overrides', [{'scale_dtype': torch.float32}, {'quant_nontext_module': True}])
def test_backend_rejects_unverified_storage_or_vision_mode(overrides):
    owner = SimpleNamespace(bits=4, group_size=64, data_type='int_sym_xh', sym=True,
                            act_bits=16, iters=4, is_moe_model=False, is_auto_scheme=False,
                            enable_alg_ext=False, enable_norm_bias_tuning=False,
                            enable_minmax_tuning=True, scale_dtype=torch.float16,
                            quant_nontext_module=False)
    owner.__dict__.update(overrides)
    with pytest.raises(ValueError, match='FP16 scale storage and text-only'):
        configure(owner, {})


def test_recipe_forwards_mode_and_resolved_config(tmp_path):
    result = quantize_qwen35_autoround(
        model_dir='unused', output_dir=str(tmp_path / 'out'), dry_run=True, topology='dense',
        data_type='int_sym_xh', optimization_mode='xh_hessian',
        xh_hessian_config={'candidates': 17},
    )
    command = result.provenance['command']
    assert command[command.index('--optimization_mode') + 1] == 'xh_hessian'
    config = json.loads(command[command.index('--xh_hessian_config') + 1])
    assert config['candidates'] == 17 and config['validation_samples'] == 16
    assert result.provenance['xh_hessian_config'] == config


@pytest.mark.parametrize('topology, mode', [('dense', 'llm-only'), ('DENSE', None), ('dense', 'llm_only')])
def test_named_dtype_selects_same_solver_defaults_as_explicit_mode(tmp_path, topology, mode):
    common = dict(model_dir='unused', output_dir=str(tmp_path / 'out'), dry_run=True,
                  topology=topology, mode=mode)
    named = quantize_qwen35_autoround(**common, data_type='int_sym_xh_hessian')
    explicit = quantize_qwen35_autoround(**common, data_type='int_sym_xh', optimization_mode='xh_hessian')
    assert named.provenance['optimization_mode'] == 'xh_hessian'
    assert named.provenance['storage_data_type'] == 'int_sym_xh'
    assert named.provenance['xh_hessian_config'] == explicit.provenance['xh_hessian_config']
    command = list(named.provenance['command'])
    assert command[command.index('--data_type') + 1] == 'int_sym_xh_hessian'
    command[command.index('--data_type') + 1] = 'int_sym_xh'
    assert command == explicit.provenance['command']


@pytest.mark.parametrize('data_type, mode', [
    ('int_sym_xh_hessian', 'xh_hessian'), ('int_sym_xh', 'standard'), ('int_sym_gptq', 'standard'),
])
def test_cli_selects_solver_from_dtype_alone(monkeypatch, data_type, mode):
    script = Path(__file__).resolve().parents[2] / 'third_party/auto-round/scripts/quantize.py'
    spec = importlib.util.spec_from_file_location('xh_hessian_cli_test', script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(sys, 'argv', [str(script), '--model', 'unused', '--output_dir', 'unused-out',
                                    '--mode', 'llm-only', '--data_type', data_type])
    args = module.parse_args()
    assert args.optimization_mode == mode
    assert args.data_type == data_type
    assert args.xh_hessian_config is None  # Backend supplies defaults.


@pytest.mark.parametrize('data_type', ['int_sym_xh_hessian', 'int_sym_xh', 'int_sym_gptq'])
def test_direct_autoround_resolves_named_solver_before_scheme_validation(data_type):
    from auto_round import AutoRound
    from transformers import LlamaConfig, LlamaForCausalLM

    model = LlamaForCausalLM(LlamaConfig(
        vocab_size=64, hidden_size=128, intermediate_size=256, num_hidden_layers=1,
        num_attention_heads=2, num_key_value_heads=2,
    ))
    owner = AutoRound(model, tokenizer=SimpleNamespace(), data_type=data_type,
                      bits=4, group_size=64, iters=4, nsamples=24, batch_size=2,
                      device_map='cpu', low_cpu_mem_usage=False)
    if data_type == 'int_sym_xh_hessian':
        assert owner.data_type == 'int_sym_xh'
        assert owner.requested_data_type == data_type
        assert owner.optimization_mode == 'xh_hessian'
        assert owner.xh_hessian_config == XHHessianConfig()
    else:
        assert owner.data_type == data_type
        assert owner.optimization_mode == 'standard'
        assert owner.xh_hessian_config is None


@pytest.mark.parametrize('override', [
    {'bits': 8}, {'group_size': 128}, {'topology': 'moe'}, {'sym': False}, {'iters': 0},
    {'nsamples': 16}, {'head': {'bits': 4}}, {'mtp': {}}, {'auto_mixed_precision': True},
    {'optimization_mode': 'typo'},
])
def test_named_dtype_rejects_unsupported_combinations(tmp_path, override):
    kwargs = dict(model_dir='unused', output_dir=str(tmp_path / 'out'), dry_run=True,
                  topology='dense', data_type='int_sym_xh_hessian')
    kwargs.update(override)
    with pytest.raises(ValueError):
        quantize_qwen35_autoround(**kwargs)


@pytest.mark.parametrize('override', [
    {'group_size': 128}, {'bits': 8}, {'data_type': 'int_sym_gptq'}, {'topology': 'moe'},
    {'iters': 0}, {'nsamples': 16}, {'xh_hessian_config': {'scale_min': -1}},
])
def test_recipe_rejects_unsupported_mode_inputs(tmp_path, override):
    kwargs = dict(model_dir='unused', output_dir=str(tmp_path / 'out'), dry_run=True, topology='dense',
                  data_type='int_sym_xh', optimization_mode='xh_hessian')
    kwargs.update(override)
    with pytest.raises(ValueError):
        quantize_qwen35_autoround(**kwargs)
