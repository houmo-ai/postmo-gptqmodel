"""Qwen3 fused expert adaptation preserves the linear checkpoint contract."""

import pytest
import torch
from transformers import Qwen3MoeConfig
from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeSparseMoeBlock

from auto_round.modeling.fused_moe.qwen3_moe import LinearQwen3MoeSparseMoeBlock


@pytest.mark.parametrize("top_k", [1, 2])
@pytest.mark.parametrize("normalize", [False, True])
@pytest.mark.parametrize("shape", [(1, 1, 8), (2, 3, 8)])
def test_fused_to_linear_forward_backward_and_reload(top_k, normalize, shape):
    torch.manual_seed(42)
    config = Qwen3MoeConfig(
        hidden_size=8, moe_intermediate_size=16, num_experts=4,
        num_experts_per_tok=top_k, norm_topk_prob=normalize,
    )
    original = Qwen3MoeSparseMoeBlock(config).eval()
    with torch.no_grad():
        for parameter in original.parameters():
            parameter.normal_(std=0.2)
    replacement = LinearQwen3MoeSparseMoeBlock.from_original(original, config)
    hidden = torch.randn(shape, requires_grad=True)
    expected = original(hidden)
    actual = replacement(hidden)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(
        torch.autograd.grad(actual.sum(), hidden, retain_graph=True)[0],
        torch.autograd.grad(expected.sum(), hidden)[0],
    )
    reloaded = LinearQwen3MoeSparseMoeBlock(config).eval()
    reloaded.load_state_dict(replacement.state_dict(), strict=True)
    assert "gate.weight" in replacement.state_dict()
    assert "experts.0.gate_proj.weight" in replacement.state_dict()
    torch.testing.assert_close(reloaded(hidden), actual)
    assert not replacement.training
    assert original.experts.gate_up_proj.device.type == "cpu"
