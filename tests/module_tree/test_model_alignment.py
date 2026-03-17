# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

import sys
from pathlib import Path

import defuser
import defuser.model_registry as defuser_model_registry
import torch
from accelerate import init_empty_weights
from transformers import AutoConfig, AutoModelForCausalLM, Qwen3NextConfig, Qwen3NextForCausalLM

from gptqmodel.models.definitions.dots1 import Dots1QModel
from gptqmodel.models.definitions.qwen3 import Qwen3QModel
from gptqmodel.models.definitions.qwen3_next import Qwen3NextGPTQ
from gptqmodel.quantization.rotation.model import MODEL_ROTATION_REGISTRY


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "models"))
from model_test import ModelTest  # noqa: E402
from torch import nn


class TestDots1Struct(ModelTest):
    NATIVE_MODEL_ID = "/monster/data/model/dots.llm1.inst"
    TRUST_REMOTE_CODE = True

    def test_module_tree_alignment(self):
        config = AutoConfig.from_pretrained(self.NATIVE_MODEL_ID, trust_remote_code=True)
        with init_empty_weights(include_buffers=True):
            shell = AutoModelForCausalLM.from_config(config, trust_remote_code=True)

        dense_layer = shell.model.layers[0]
        self.assertTrue(hasattr(dense_layer.mlp, "gate_proj"))
        self.assertFalse(hasattr(dense_layer.mlp, "experts"))

        moe_layer = shell.model.layers[config.first_k_dense_replace]
        self.assertTrue(hasattr(moe_layer.mlp, "experts"))
        if isinstance(moe_layer.mlp.experts, nn.ModuleList):
            expert_num = len(moe_layer.mlp.experts)
        else:
            expert_num = moe_layer.mlp.experts.num_experts
        self.assertEqual(expert_num, config.n_routed_experts)
        self.assertTrue(hasattr(moe_layer.mlp, "shared_experts"))

        self.assertIn("q_norm:!", Dots1QModel.module_tree[3]["self_attn"])
        self.assertIn("k_norm:!", Dots1QModel.module_tree[3]["self_attn"])
        mlp_key = next(
            key for key in Dots1QModel.module_tree[3].keys()
            if Dots1QModel._parse_module_flags(key)[0] == "mlp"
        )
        self.assertIn("gate_proj:0", Dots1QModel.module_tree[3][mlp_key][""])
        self.assertIn("#", Dots1QModel.module_tree[3][mlp_key]["experts"])
        self.assertIn("shared_experts", Dots1QModel.module_tree[3][mlp_key])


class TestMarinModuleTree(ModelTest):
    NATIVE_MODEL_ID = "/monster/data/model/marin-32b-base"

    def test_marin_module_tree(self):
        config = AutoConfig.from_pretrained(self.NATIVE_MODEL_ID, trust_remote_code=True)
        with init_empty_weights(include_buffers=True):
            shell = AutoModelForCausalLM.from_config(config, trust_remote_code=True)

        decoder_layer = shell.model.layers[0]
        self.assertTrue(hasattr(decoder_layer.self_attn, "q_norm"))
        self.assertTrue(hasattr(decoder_layer.self_attn, "k_norm"))
        self.assertTrue(hasattr(decoder_layer.self_attn, "q_proj"))
        self.assertTrue(hasattr(decoder_layer.self_attn, "o_proj"))
        self.assertIn("q_norm:!", Qwen3QModel.module_tree[3]["self_attn"])
        self.assertIn("k_norm:!", Qwen3QModel.module_tree[3]["self_attn"])


class TestMarinAwqModuleTree(ModelTest):
    NATIVE_MODEL_ID = "/monster/data/model/marin-32b-base"

    def test_marin_module_tree(self):
        config = AutoConfig.from_pretrained(self.NATIVE_MODEL_ID, trust_remote_code=True)
        with init_empty_weights(include_buffers=True):
            shell = AutoModelForCausalLM.from_config(config, trust_remote_code=True)

        decoder_layer = shell.model.layers[0]
        self.assertTrue(hasattr(decoder_layer.self_attn, "q_norm"))
        self.assertTrue(hasattr(decoder_layer.self_attn, "k_norm"))
        self.assertTrue(hasattr(decoder_layer.self_attn, "q_proj"))
        self.assertTrue(hasattr(decoder_layer.self_attn, "o_proj"))
        self.assertIn("q_norm:!", Qwen3QModel.module_tree[3]["self_attn"])
        self.assertIn("k_norm:!", Qwen3QModel.module_tree[3]["self_attn"])


def test_qwen3_next_module_tree_alignment():
    config = Qwen3NextConfig(
        vocab_size=128,
        hidden_size=32,
        intermediate_size=64,
        moe_intermediate_size=48,
        shared_expert_intermediate_size=40,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        linear_num_key_heads=2,
        linear_num_value_heads=4,
        linear_key_head_dim=8,
        linear_value_head_dim=8,
        decoder_sparse_step=2,
        num_experts=4,
        num_experts_per_tok=2,
        layer_types=[
            "linear_attention",
            "linear_attention",
            "linear_attention",
            "full_attention",
        ],
        dtype=torch.float32,
    )
    with init_empty_weights(include_buffers=True):
        shell = Qwen3NextForCausalLM(config)

    dense_layer = next(layer for layer in shell.model.layers if hasattr(layer.mlp, "gate_proj"))
    moe_layer = next(layer for layer in shell.model.layers if hasattr(layer.mlp, "experts"))
    full_attn_layer = next(layer for layer in shell.model.layers if hasattr(layer, "self_attn"))
    linear_attn_layer = next(layer for layer in shell.model.layers if hasattr(layer, "linear_attn"))

    assert hasattr(dense_layer.mlp, "down_proj")
    assert hasattr(moe_layer.mlp, "shared_expert")
    assert hasattr(moe_layer.mlp, "shared_expert_gate")
    assert hasattr(full_attn_layer.self_attn, "q_norm")
    assert hasattr(full_attn_layer.self_attn, "k_norm")
    assert hasattr(linear_attn_layer.linear_attn, "in_proj_qkvz")
    assert hasattr(linear_attn_layer.linear_attn, "in_proj_ba")

    layer_tree = Qwen3NextGPTQ.module_tree[3]
    assert "q_norm:!" in layer_tree["self_attn"]
    assert "k_norm:!" in layer_tree["self_attn"]
    assert "in_proj_qkvz:0" in layer_tree["linear_attn"]
    assert "in_proj_ba:!" in layer_tree["linear_attn"]
    assert "mlp" in layer_tree

    moe_key = next(
        key for key in layer_tree.keys()
        if Qwen3NextGPTQ._parse_module_flags(key)[0] == "mlp"
        and "moe" in Qwen3NextGPTQ._parse_module_flags(key)[1]
    )
    assert "shared_expert" in layer_tree[moe_key]
    assert "shared_expert_gate" in layer_tree[moe_key]
    assert "#" in layer_tree[moe_key]["experts"]

    blocks = Qwen3NextGPTQ.build_layer_modules(Qwen3NextGPTQ.module_tree)
    assert any("self_attn.q_proj" in block for block in blocks)
    assert any("linear_attn.in_proj_qkvz" in block for block in blocks)
    assert any("mlp.gate_proj" in block for block in blocks)
    assert any("mlp.shared_expert.gate_proj" in block for block in blocks)
    assert any("mlp.experts.{expert_index}.gate_proj" in block for block in blocks)

    awq_blocks = Qwen3NextGPTQ.full_layer_modules(model_config=config, is_awq_quantize=True)
    assert any("mlp.experts.0.gate_proj" in block for block in awq_blocks)
    assert any("mlp.shared_expert.gate_proj" in block for block in awq_blocks)

    assert any(entry["cls"] is Qwen3NextGPTQ for entry in MODEL_ROTATION_REGISTRY)


def test_qwen3_next_before_model_load_registers_defuser_and_unfuses_experts():
    original_entry = defuser_model_registry.MODEL_CONFIG.pop("qwen3_next", None)
    original_flag = getattr(defuser_model_registry, "_gptqmodel_qwen3_next_registry_patch", None)

    try:
        if hasattr(defuser_model_registry, "_gptqmodel_qwen3_next_registry_patch"):
            delattr(defuser_model_registry, "_gptqmodel_qwen3_next_registry_patch")

        Qwen3NextGPTQ.before_model_load(Qwen3NextGPTQ, load_quantized_model=False)

        assert Qwen3NextGPTQ.support_offload_to_disk is False
        assert defuser_model_registry.MODEL_CONFIG["qwen3_next"]["min_transformers_version"] == "5.2.0"

        config = Qwen3NextConfig(
            vocab_size=128,
            hidden_size=32,
            intermediate_size=64,
            moe_intermediate_size=24,
            shared_expert_intermediate_size=16,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=8,
            linear_key_head_dim=8,
            linear_value_head_dim=8,
            linear_num_key_heads=2,
            linear_num_value_heads=4,
            decoder_sparse_step=1,
            num_experts=4,
            num_experts_per_tok=2,
            mlp_only_layers=[],
            dtype=torch.float32,
        )
        model = Qwen3NextForCausalLM(config)

        experts = model.model.layers[0].mlp.experts
        assert hasattr(experts, "gate_up_proj")
        assert hasattr(experts, "down_proj")

        defuser.convert_model(model, cleanup_original=False)

        experts = model.model.layers[0].mlp.experts
        assert not hasattr(experts, "gate_up_proj")
        assert not hasattr(experts, "down_proj")
        assert hasattr(experts, "0")
        assert hasattr(experts, "1")
        assert hasattr(experts, "2")
        assert hasattr(experts, "3")
        expert_zero = getattr(experts, "0")
        assert hasattr(expert_zero, "gate_proj")
        assert hasattr(expert_zero, "up_proj")
        assert hasattr(expert_zero, "down_proj")
        assert getattr(model.config, "_experts_implementation", None) == "linear_loop"
    finally:
        if original_entry is not None:
            defuser_model_registry.MODEL_CONFIG["qwen3_next"] = original_entry
        else:
            defuser_model_registry.MODEL_CONFIG.pop("qwen3_next", None)

        if original_flag is not None:
            defuser_model_registry._gptqmodel_qwen3_next_registry_patch = original_flag
        elif hasattr(defuser_model_registry, "_gptqmodel_qwen3_next_registry_patch"):
            delattr(defuser_model_registry, "_gptqmodel_qwen3_next_registry_patch")
