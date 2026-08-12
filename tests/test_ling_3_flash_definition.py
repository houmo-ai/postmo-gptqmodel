import sys
import types

import torch

from gptqmodel.models.auto import MODEL_MAP
from gptqmodel.models.definitions.ling_3_flash import (
    Ling3FlashQModel,
    _shield_short_convolutions_from_gptq_hooks,
    patch_ling_3_flash_remote_code,
)


def test_ling_3_flash_is_registered_with_hybrid_quantization_tree():
    assert MODEL_MAP["bailing_hybrid"] is Ling3FlashQModel
    assert Ling3FlashQModel.require_trust_remote_code is True

    blocks = Ling3FlashQModel.build_layer_modules(Ling3FlashQModel.module_tree)
    flattened = [name for block in blocks for name in block]

    # KDA state-sensitive decay/beta paths stay floating point.
    for name in (
        "attention.f_proj:!",
        "attention.f_a_proj:!",
        "attention.f_b_proj:!",
        "attention.b_proj:!",
    ):
        assert name in flattened

    # KDA projections, MLA boundaries, dense/shared/routed experts are all
    # available for the caller's W8/W4 dynamic policy.
    for name in (
        "attention.q_proj",
        "attention.kv_a_proj_with_mqa",
        "attention.kv_b_proj",
        "attention.o_proj",
        "attention.dense",
        "mlp.gate_proj",
        "mlp.shared_experts.gate_proj",
        "mlp.experts.{expert_index}.gate_proj",
    ):
        assert name in flattened

    assert "mlp.gate:!" in flattened


def test_ling_short_convolution_is_shielded_without_gptq_core_changes():
    class ShortConvolution(torch.nn.Conv1d):
        def forward(self, x, cache=None, output_final_state=False):
            output = torch.nn.functional.conv1d(x, self.weight, self.bias)
            return output, cache if output_final_state else None

    class Block(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.short_conv = ShortConvolution(2, 2, 1)

    block = Block()
    state_keys = list(block.state_dict())
    original_cls = type(block.short_conv)

    assert _shield_short_convolutions_from_gptq_hooks(block) == 1
    assert isinstance(block.short_conv, torch.nn.Module)
    assert not isinstance(block.short_conv, torch.nn.Conv1d)
    assert type(block.short_conv)._gptqmodel_ling_original_class is original_cls
    assert list(block.state_dict()) == state_keys

    cache = torch.ones(1)
    output, actual_cache = block.short_conv(
        torch.randn(1, 2, 3),
        cache=cache,
        output_final_state=True,
    )
    assert output.shape == (1, 2, 3)
    torch.testing.assert_close(actual_cache, cache)


def test_ling_3_flash_remote_metadata_compatibility():
    config_module_name = "test_ling.configuration_bailing_moe_v3"
    model_module_name = "test_ling.modeling_bailing_moe_v3"
    config_module = types.ModuleType(config_module_name)
    model_module = types.ModuleType(model_module_name)

    class FakeConfig:
        model_type = ""

    FakeConfig.__module__ = config_module_name
    config_module.BailingMoeV3Config = FakeConfig
    FakeModel = type(
        "FakeModel",
        (),
        {
            "__module__": model_module_name,
            "_tied_weights_keys": ["lm_head.weight"],
        },
    )
    model_module.FakeModel = FakeModel

    sys.modules[config_module_name] = config_module
    sys.modules[model_module_name] = model_module
    try:
        patch_ling_3_flash_remote_code()
        config = FakeConfig()
        config.rope_scaling = {"rope_type": "default"}

        assert FakeConfig.model_type == "bailing_hybrid"
        assert config.rope_scaling is None
        assert FakeModel._tied_weights_keys == {}
    finally:
        sys.modules.pop(config_module_name, None)
        sys.modules.pop(model_module_name, None)
