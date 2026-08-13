from types import SimpleNamespace

import torch

from gptqmodel.models import auto
from gptqmodel.models.definitions.deepseek_v4 import DeepSeekV4QModel
from gptqmodel.nn_modules.hooked_linear import HookedLinear, replace_module_with_hooked_legacy
from gptqmodel.utils.model import nested_move_to


def test_deepseek_v4_model_type_selects_definition(monkeypatch):
    fake_config = SimpleNamespace(model_type="deepseek_v4")

    monkeypatch.setattr(
        auto.AutoConfig,
        "from_pretrained",
        lambda *args, **kwargs: fake_config,
    )

    assert auto.check_and_get_model_definition("/tmp/deepseek-v4") is DeepSeekV4QModel


def test_deepseek_v4_module_tree_matches_attention_and_experts():
    layer_modules = DeepSeekV4QModel.simple_layer_modules(
        model_config=SimpleNamespace(n_routed_experts=256),
        quantize_config=SimpleNamespace(dynamic=None),
    )
    flat_modules = {name for block in layer_modules for name in block}

    assert "self_attn.q_a_proj" in flat_modules
    assert "self_attn.q_b_proj" in flat_modules
    assert "self_attn.kv_proj" in flat_modules
    assert "self_attn.o_b_proj" in flat_modules
    assert "self_attn.o_a_proj" not in flat_modules
    assert "mlp.experts.99.gate_proj" in flat_modules
    assert "mlp.experts.99.up_proj" in flat_modules
    assert "mlp.experts.99.down_proj" in flat_modules
    assert "mlp.shared_experts.gate_proj" in flat_modules


def test_installed_defuser_has_official_deepseek_v4_support():
    from defuser.model_registry import MODEL_CONFIG

    assert "deepseek_v4" in MODEL_CONFIG


def test_deepseek_v4_replays_mixed_precision_layers_under_bfloat16_autocast():
    assert DeepSeekV4QModel.replay_autocast_dtype is torch.bfloat16


def test_deepseek_v4_disables_full_turtle_auto_reload():
    model = DeepSeekV4QModel.__new__(DeepSeekV4QModel)
    assert model._resolve_turtle_reload_threshold() == 0


def test_nested_move_to_moves_deepseek_v4_position_embedding_dict():
    position_embeddings = {
        "main": (torch.ones(1), torch.zeros(1)),
        "compress": (torch.ones(1), torch.zeros(1)),
    }

    moved = nested_move_to(position_embeddings, device=torch.device("meta"))

    assert isinstance(moved, dict)
    assert isinstance(moved["main"], tuple)
    assert all(tensor.device.type == "meta" for pair in moved.values() for tensor in pair)


def test_hook_replacement_preserves_non_quantized_custom_linear_subclass():
    class GroupedLinear(torch.nn.Linear):
        pass

    root = torch.nn.Module()
    root.grouped = GroupedLinear(4, 4, bias=False)
    root.plain = torch.nn.Linear(4, 4, bias=False)

    replace_module_with_hooked_legacy(root, skip_module_paths={"grouped"})

    assert isinstance(root.grouped, GroupedLinear)
    assert type(root.grouped) is GroupedLinear
    assert isinstance(root.plain, HookedLinear)
