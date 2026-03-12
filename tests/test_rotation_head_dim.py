from types import SimpleNamespace

import torch
from torch import nn

from gptqmodel.models.definitions.llama import LlamaQModel
from gptqmodel.models.definitions.qwen3 import Qwen3QModel
from gptqmodel.quantization.rotation.model import MODEL_ROTATION_REGISTRY
from gptqmodel.quantization.rotation.rotation import _resolve_attention_head_dim
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
