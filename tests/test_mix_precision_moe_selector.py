# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

import sys
from pathlib import Path

import torch.nn as nn


repo_root = Path(__file__).resolve().parents[1]
repo_str = str(repo_root)
if repo_str not in sys.path:
    sys.path.insert(0, repo_str)

from gptqmodel.models.definitions.glm4_moe_lite import Glm4MoeLiteQModel
from gptqmodel.quantization.mix_precision.layer_selector import LayerSelector


class _ExpertBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_proj = nn.Linear(8, 16, bias=False)
        self.up_proj = nn.Linear(8, 16, bias=False)
        self.down_proj = nn.Linear(16, 8, bias=False)
        # Routed MoE implementations may include expert-local gate/router like names.
        # This should not be included in expert mix-precision search groups.
        self.gate = nn.Linear(8, 4, bias=False)


class _SharedExpertBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_proj = nn.Linear(8, 16, bias=False)
        self.up_proj = nn.Linear(8, 16, bias=False)
        self.down_proj = nn.Linear(16, 8, bias=False)


class _IndexFirstMoE(nn.Module):
    def __init__(self):
        super().__init__()
        self.experts = nn.ModuleList([_ExpertBlock(), _ExpertBlock()])
        self.shared_experts = _SharedExpertBlock()
        self.gate = nn.Linear(8, 2, bias=False)


class _ProjFirstExperts(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_proj = nn.ModuleList([nn.Linear(8, 16, bias=False) for _ in range(2)])
        self.up_proj = nn.ModuleList([nn.Linear(8, 16, bias=False) for _ in range(2)])
        self.down_proj = nn.ModuleList([nn.Linear(16, 8, bias=False) for _ in range(2)])
        self.gate = nn.Linear(8, 2, bias=False)


class _ProjFirstMoE(nn.Module):
    def __init__(self):
        super().__init__()
        self.experts = _ProjFirstExperts()
        self.shared_experts = _SharedExpertBlock()
        self.gate = nn.Linear(8, 2, bias=False)


class _ToyModel(nn.Module):
    def __init__(self, moe: nn.Module):
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([nn.Module()])
        self.model.layers[0].mlp = moe
        self.model.layers[0].dense = nn.Linear(8, 8, bias=False)


class _Glm4SelfAttn(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_a_proj = nn.Linear(8, 8, bias=False)
        self.q_b_proj = nn.Linear(8, 8, bias=False)
        self.kv_a_proj_with_mqa = nn.Linear(8, 8, bias=False)
        self.kv_b_proj = nn.Linear(8, 8, bias=False)
        self.o_proj = nn.Linear(8, 8, bias=False)


class _Glm4MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.experts = _ProjFirstExperts()
        self.shared_experts = _SharedExpertBlock()
        self.gate = nn.Linear(8, 2, bias=False)
        self.gate_proj = nn.Linear(8, 16, bias=False)
        self.up_proj = nn.Linear(8, 16, bias=False)
        self.down_proj = nn.Linear(16, 8, bias=False)


class _Glm4ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([nn.Module()])
        self.model.layers[0].self_attn = _Glm4SelfAttn()
        self.model.layers[0].mlp = _Glm4MLP()
        self.lm_head = nn.Linear(8, 8, bias=False)


def _assert_expert_group_only_has_three_proj(model: nn.Module) -> None:
    selector = LayerSelector(moe_unified=True)
    result = selector.select(model)

    group_key = "model.layers.0.mlp.experts"
    assert group_key in result.candidates
    assert group_key in result.moe_groups

    members = result.moe_groups[group_key]
    assert members, "expected routed expert members"

    for name in members:
        assert ".experts." in name
        assert "gate_proj" in name or "up_proj" in name or "down_proj" in name
        assert ".gate" not in name.replace("gate_proj", "")


def test_moe_expert_group_filters_router_modules_index_first():
    model = _ToyModel(_IndexFirstMoE())
    _assert_expert_group_only_has_three_proj(model)


def test_moe_expert_group_filters_router_modules_proj_first():
    model = _ToyModel(_ProjFirstMoE())
    _assert_expert_group_only_has_three_proj(model)


def test_fixed_bits_none_skips_module():
    model = nn.Module()
    model.linear = nn.Linear(8, 8, bias=False)
    model.lm_head = nn.Linear(8, 8, bias=False)

    selector = LayerSelector(fixed_bits={"lm_head": None})
    result = selector.select(model)

    assert "linear" in result.candidates
    assert "lm_head" not in result.candidates
    assert result.fixed_bits["lm_head"] is None


def test_glm4_mix_precision_template_targets_only_quantizable_modules():
    model = _Glm4ToyModel()
    config = Glm4MoeLiteQModel.mix_precision_config_template()

    selector = LayerSelector(
        include_patterns=config.get("layer_include"),
        exclude_patterns=config.get("layer_exclude"),
        fixed_bits=config.get("fixed_bits"),
        moe_unified=config.get("moe_unified", True),
    )
    result = selector.select(model)

    expected_regular = {
        "model.layers.0.self_attn.q_a_proj",
        "model.layers.0.self_attn.q_b_proj",
        "model.layers.0.self_attn.kv_a_proj_with_mqa",
        "model.layers.0.self_attn.kv_b_proj",
        "model.layers.0.self_attn.o_proj",
        "model.layers.0.mlp.shared_experts.gate_proj",
        "model.layers.0.mlp.shared_experts.up_proj",
        "model.layers.0.mlp.shared_experts.down_proj",
        "model.layers.0.mlp.gate_proj",
        "model.layers.0.mlp.up_proj",
        "model.layers.0.mlp.down_proj",
    }

    for module_name in expected_regular:
        assert module_name in result.regular_modules
        assert module_name in result.candidates

    expert_group = "model.layers.0.mlp.experts"
    assert expert_group in result.candidates
    assert expert_group in result.moe_groups
    assert len(result.moe_groups[expert_group]) == 6

    assert "model.layers.0.mlp.gate" not in result.candidates
    assert "model.layers.0.mlp.gate" not in result.regular_modules
    assert "lm_head" not in result.candidates


def test_glm4_diceq_is_disabled_by_default():
    config = Glm4MoeLiteQModel.diceq_config_template()

    assert config["enabled"] is False
