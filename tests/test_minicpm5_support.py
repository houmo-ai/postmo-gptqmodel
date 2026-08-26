from types import SimpleNamespace

from torch import nn

from gptqmodel.models import auto
from gptqmodel.models.definitions.minicpm5 import MiniCPM5MoEQModel
from gptqmodel.models.moe_lifecycle import GateUpDownMoELifecycleHooks


def test_minicpm5_model_type_selects_definition(monkeypatch):
    fake_config = SimpleNamespace(model_type="minicpm5_moe")
    monkeypatch.setattr(
        auto.AutoConfig,
        "from_pretrained",
        lambda *args, **kwargs: fake_config,
    )

    assert auto.check_and_get_model_definition("/tmp/minicpm5") is MiniCPM5MoEQModel
    assert "minicpm5_moe" in auto.SUPPORTED_MODELS


def test_minicpm5_definition_matches_remote_code_contract():
    assert MiniCPM5MoEQModel.require_trust_remote_code is True
    assert MiniCPM5MoEQModel.require_fast_init is False
    assert MiniCPM5MoEQModel.layer_modules_strict is False
    assert MiniCPM5MoEQModel.dynamic_expert_index == "n_routed_experts"
    assert MiniCPM5MoEQModel.pre_lm_head_norm_module == "model.norm"
    assert isinstance(MiniCPM5MoEQModel.moe_lifecycle_hooks, GateUpDownMoELifecycleHooks)
    assert MiniCPM5MoEQModel.get_moe_module_name() == "mlp"


def test_minicpm5_module_tree_covers_dense_and_routed_mlp():
    layer_modules = MiniCPM5MoEQModel.simple_layer_modules(
        model_config=SimpleNamespace(n_routed_experts=3),
        quantize_config=SimpleNamespace(dynamic=None),
    )
    flat_modules = {name for block in layer_modules for name in block}

    assert "self_attn.q_proj" in flat_modules
    assert "self_attn.k_proj" in flat_modules
    assert "self_attn.v_proj" in flat_modules
    assert "self_attn.o_proj" in flat_modules
    assert "mlp.gate_proj" in flat_modules
    assert "mlp.up_proj" in flat_modules
    assert "mlp.down_proj" in flat_modules
    assert "mlp.experts.2.gate_proj" in flat_modules
    assert "mlp.experts.2.up_proj" in flat_modules
    assert "mlp.experts.2.down_proj" in flat_modules
    assert "mlp.shared_experts.gate_proj" in flat_modules
    assert "mlp.shared_experts.up_proj" in flat_modules
    assert "mlp.shared_experts.down_proj" in flat_modules
    assert "mlp.gate" not in flat_modules


def test_minicpm5_moe_lifecycle_finds_routed_and_shared_experts():
    moe = nn.Module()
    moe.experts = nn.ModuleList([nn.Identity(), nn.Identity()])
    moe.shared_experts = nn.Identity()
    hooks = MiniCPM5MoEQModel.moe_lifecycle_hooks

    assert hooks.get_experts_module(moe, MiniCPM5MoEQModel) is moe.experts
    assert hooks.get_shared_experts_module(moe, MiniCPM5MoEQModel) is moe.shared_experts
