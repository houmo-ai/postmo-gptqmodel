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
