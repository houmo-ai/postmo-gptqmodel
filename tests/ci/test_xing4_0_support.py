import json
from types import SimpleNamespace

import pytest

from torch import nn

from gptqmodel.models import auto
from gptqmodel.models.definitions.xing4_0 import Xing4_0QModel
from gptqmodel.models.moe_lifecycle import GateUpDownMoELifecycleHooks
from gptqmodel.recipes.xing4_0 import build_xing4_0_dynamic, quantize_xing4_0


def test_xing4_0_model_type_selects_definition(monkeypatch):
    fake_config = SimpleNamespace(model_type="xing4_0")
    monkeypatch.setattr(
        auto.AutoConfig,
        "from_pretrained",
        lambda *args, **kwargs: fake_config,
    )

    assert auto.check_and_get_model_definition("/tmp/xing4_0") is Xing4_0QModel
    assert "xing4_0" in auto.SUPPORTED_MODELS


def test_xing4_0_definition_matches_remote_code_contract():
    assert Xing4_0QModel.require_trust_remote_code is True
    assert Xing4_0QModel.require_fast_init is False
    assert Xing4_0QModel.layer_modules_strict is False
    assert Xing4_0QModel.dynamic_expert_index == "n_routed_experts"
    assert Xing4_0QModel.pre_lm_head_norm_module == "model.norm"
    assert isinstance(Xing4_0QModel.moe_lifecycle_hooks, GateUpDownMoELifecycleHooks)
    assert Xing4_0QModel.get_moe_module_name() == "mlp"


def test_xing4_0_module_tree_covers_mla_dense_and_routed_mlp():
    layer_modules = Xing4_0QModel.simple_layer_modules(
        model_config=SimpleNamespace(n_routed_experts=3),
        quantize_config=SimpleNamespace(dynamic=None),
    )
    flat_modules = {name for block in layer_modules for name in block}

    assert "self_attn.q_a_proj" in flat_modules
    assert "self_attn.q_b_proj" in flat_modules
    assert "self_attn.q_absorbed_proj" in flat_modules
    assert "self_attn.q_rope_proj" in flat_modules
    assert "self_attn.kv_a_proj_with_mqa" in flat_modules
    assert "self_attn.kv_b_proj" in flat_modules
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
    assert not any(name.startswith("attn_hc.") for name in flat_modules)
    assert not any(name.startswith("ffn_hc.") for name in flat_modules)


def test_xing4_0_moe_lifecycle_finds_routed_and_shared_experts():
    moe = nn.Module()
    moe.experts = nn.ModuleList([nn.Identity(), nn.Identity()])
    moe.shared_experts = nn.Identity()
    hooks = Xing4_0QModel.moe_lifecycle_hooks

    assert hooks.get_experts_module(moe, Xing4_0QModel) is moe.experts
    assert hooks.get_shared_experts_module(moe, Xing4_0QModel) is moe.shared_experts


def test_xing4_0_dynamic_marks_experts_w4_and_attention_w8():
    dynamic = build_xing4_0_dynamic(
        self_attn_bits=8,
        dense_mlp_bits=8,
        shared_expert_bits=8,
        expert_bits=4,
        group_size=64,
        sym=True,
    )
    assert (
        dynamic[
            r"model\.layers\.\d+\.self_attn\.(q_proj|q_a_proj|q_b_proj|q_absorbed_proj|q_rope_proj|kv_a_proj_with_mqa|kv_b_proj|o_proj)$"
        ]["bits"]
        == 8
    )
    assert (
        dynamic[
            r"model\.layers\.\d+\.mlp\.experts\.\d+\.(gate_proj|up_proj|down_proj)$"
        ]["bits"]
        == 4
    )


def test_xing4_0_recipe_dry_run(tmp_path):
    result = quantize_xing4_0(
        model_dir=str(tmp_path / "missing_model"),
        output_dir=str(tmp_path / "out"),
        method="gptq",
        dry_run=True,
        calibration_data=["hello world"] * 8,
        nsamples=8,
        seqlen=16,
    )
    assert result.model_family == "xing4_0"
    assert result.precision["routed_experts"] == "W4"
    assert result.precision["self_attn"] == "W8"
    assert result.precision["mla_weight_layout"] == "xh2_absorbed_v1"
    assert result.provenance["offload_to_disk"] is False
    assert result.provenance["dry_run"] is True


@pytest.mark.parametrize("head_alias", [False, True])
@pytest.mark.parametrize("rope_type", ["default", "yarn"])
def test_saved_xing_config_preserves_source_serialization(tmp_path, head_alias, rope_type):
    from gptqmodel.models.definitions.xing4_0 import _preserve_source_config

    source, output = tmp_path / "source", tmp_path / "output"
    source.mkdir()
    output.mkdir()
    scaling = {"type": "yarn", "factor": 64} if rope_type == "yarn" else None
    config = {"qk_rope_head_dim": 64, "rope_theta": 10000, "rope_scaling": scaling}
    if head_alias:
        config["head_dim"] = 64
    source_text = json.dumps(config)
    (source / "config.json").write_text(source_text)
    rope = {"rope_type": rope_type, "rope_theta": 10000}
    if scaling:
        rope.update(scaling)
    saved = {
        "qk_rope_head_dim": 64, "head_dim": 64, "rope_parameters": rope,
        "quantization_config": {"bits": 4, "quant_method": "gptq"}, "mla_weight_layout": "xh2_absorbed_v1",
    }
    (output / "config.json").write_text(json.dumps(saved))
    (output / "model.safetensors").write_bytes(b"unchanged-weight-payload")
    _preserve_source_config(source, output)
    result = json.loads((output / "config.json").read_text())
    assert result == {
        **config, "quantization_config": saved["quantization_config"], "mla_weight_layout": "xh2_absorbed_v1",
    }
    assert (source / "config.json").read_text() == source_text
    assert (output / "model.safetensors").read_bytes() == b"unchanged-weight-payload"


@pytest.mark.parametrize("changed", ["head_dim", "qk_rope_head_dim", "factor", "extra"])
def test_preserving_xing_config_rejects_changed_behavior(tmp_path, changed):
    from gptqmodel.models.definitions.xing4_0 import _preserve_source_config

    source, output = tmp_path / "source", tmp_path / "output"
    source.mkdir()
    output.mkdir()
    (source / "config.json").write_text(json.dumps({
        "qk_rope_head_dim": 64, "rope_theta": 10000, "rope_scaling": {"type": "yarn", "factor": 64},
    }))
    saved = {"qk_rope_head_dim": 64, "head_dim": 64, "rope_parameters": {
        "rope_type": "yarn", "rope_theta": 10000, "factor": 64,
    }}
    if changed in {"head_dim", "qk_rope_head_dim"}:
        saved[changed] = 128
    else:
        saved["rope_parameters"][changed] = 2
    original = json.dumps(saved)
    (output / "config.json").write_text(original)
    with pytest.raises(ValueError):
        _preserve_source_config(source, output)
    assert (output / "config.json").read_text() == original
