import copy
from types import SimpleNamespace

import defuser
import torch
from torch import nn
from transformers import Qwen3NextConfig, Qwen3NextForCausalLM
from transformers.models.qwen3_next.modeling_qwen3_next import Qwen3NextRMSNormGated

from gptqmodel.models.definitions.llama import LlamaQModel
from gptqmodel.models.definitions.qwen3 import Qwen3QModel
from gptqmodel.models.definitions.qwen3_next import Qwen3NextGPTQ
from gptqmodel.quantization.rotation.model import MODEL_ROTATION_REGISTRY
from gptqmodel.quantization.rotation.rotation import (
    _get_norm_scale,
    _resolve_attention_head_dim,
    fuse_layer_norms_qwen3_next,
    rotate_model_qwen3_next,
)
from gptqmodel.utils.structure import capture_rotation_preserved_modules, restore_rotation_preserved_modules


def test_resolve_attention_head_dim_prefers_explicit_config_value():
    config = SimpleNamespace(head_dim=128)

    assert _resolve_attention_head_dim(config, hidden_size=2560, num_heads=32) == 128


def test_resolve_attention_head_dim_falls_back_to_hidden_size_ratio():
    config = SimpleNamespace()

    assert _resolve_attention_head_dim(config, hidden_size=4096, num_heads=32) == 128


def test_qwen3_rotation_registry_precedes_llama_fallback():
    classes = [entry["cls"] for entry in MODEL_ROTATION_REGISTRY]

    assert classes.index(Qwen3QModel) < classes.index(LlamaQModel)
    assert next(entry for entry in MODEL_ROTATION_REGISTRY if entry["cls"] is Qwen3QModel)["rotate_fn"].__name__ == "rotate_model_qwen3_5"


class _TinyRotationModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Module()
        self.model.embed_tokens = nn.Embedding(8, 4)
        self.model.norm = nn.LayerNorm(4)
        self.lm_head = nn.Linear(4, 8, bias=False)


def test_restore_rotation_preserved_modules_rehydrates_meta_modules():
    torch.manual_seed(0)
    model = _TinyRotationModel()
    model.model.embed_tokens.weight.data.uniform_(-1.0, 1.0)
    model.model.norm.weight.data.uniform_(0.5, 1.5)

    preserved = capture_rotation_preserved_modules(model, "model.norm", "lm_head")

    expected_embed = preserved["model.embed_tokens"].weight.detach().clone()
    expected_norm = preserved["model.norm"].weight.detach().clone()

    model.model.embed_tokens = nn.Embedding(8, 4, device="meta")
    model.model.norm = nn.LayerNorm(4, device="meta")

    restored = restore_rotation_preserved_modules(model, preserved)

    assert restored == 2
    assert torch.equal(model.model.embed_tokens.weight.detach().cpu(), expected_embed)
    assert torch.equal(model.model.norm.weight.detach().cpu(), expected_norm)


def test_qwen3_next_rmsnorm_gated_uses_direct_weight_scale():
    norm = Qwen3NextRMSNormGated(hidden_size=4)
    norm.weight.data = torch.tensor([1.5, 0.5, 2.0, 3.0])

    scale = _get_norm_scale(norm)

    assert torch.equal(scale, norm.weight.double())


def test_qwen3_next_rotation_preserves_logits_on_tiny_model():
    torch.manual_seed(0)
    config = Qwen3NextConfig(
        vocab_size=64,
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
        mlp_only_layers=[0],
        layer_types=["full_attention", "full_attention"],
        tie_word_embeddings=False,
    )
    baseline = Qwen3NextForCausalLM(copy.deepcopy(config)).eval()
    rotated = Qwen3NextForCausalLM(copy.deepcopy(config)).eval()
    rotated.load_state_dict(baseline.state_dict())

    Qwen3NextGPTQ.before_model_load(Qwen3NextGPTQ, load_quantized_model=False)
    defuser.convert_model(rotated, cleanup_original=False)

    input_ids = torch.tensor([[1, 2, 3, 4]])
    with torch.no_grad():
        reference_logits = baseline(input_ids=input_ids, use_cache=False).logits

    fuse_layer_norms_qwen3_next(
        rotated,
        pre_lm_head_norm_module_name="model.norm",
        layers_node="model.layers",
        lm_head_name="lm_head",
    )
    rotate_model_qwen3_next(
        rotated,
        rotate_mode="hadamard",
        device=torch.device("cpu"),
        lm_head_name="lm_head",
        layers_node="model.layers",
    )

    with torch.no_grad():
        rotated_logits = rotated(input_ids=input_ids, use_cache=False).logits

    assert torch.allclose(reference_logits, rotated_logits, atol=1e-4, rtol=1e-4)
