"""Real W5 endpoint calibration over Dense and hybrid MoE QDQ backbones."""
import hashlib
import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file


@pytest.fixture(params=[False, True], ids=["dense", "moe"])
def qdq(tmp_path, request):
    return _make_qdq(tmp_path, request.param)


def _make_qdq(tmp_path, moe):
    from transformers import Qwen2Tokenizer
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM
    from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import Qwen3_5MoeTextConfig
    from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeForCausalLM
    from gptqmodel.recipes.minicpm_v_4_7 import _emit_causal_float_bin

    torch.manual_seed(47)
    torch.set_num_threads(2)
    cfg = dict(vocab_size=128, hidden_size=128, intermediate_size=256, num_hidden_layers=2,
               num_attention_heads=4, num_key_value_heads=2, head_dim=32,
               linear_num_key_heads=2, linear_num_value_heads=4, linear_key_head_dim=32,
               linear_value_head_dim=32, linear_conv_kernel_dim=4,
               layer_types=["linear_attention", "full_attention"], max_position_embeddings=128,
               tie_word_embeddings=False)
    if moe:
        model = Qwen3_5MoeForCausalLM(Qwen3_5MoeTextConfig(
            **cfg, num_experts=4, num_experts_per_tok=2,
            moe_intermediate_size=128, shared_expert_intermediate_size=128))
    else:
        model = Qwen3_5ForCausalLM(Qwen3_5TextConfig(**cfg))
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            parameter.copy_(torch.ones_like(parameter) if 'norm' in name
                            else torch.randn_like(parameter) * .02)
    source = tmp_path / "qdq"
    _emit_causal_float_bin(model, source)
    Qwen2Tokenizer(vocab={"<|endoftext|>": 0, "t": 1, "Ġ": 2,
                          **{str(i): i + 3 for i in range(10)}}, merges=[]).save_pretrained(source)
    data = tmp_path / "calibration.jsonl"
    data.write_text("\n".join(json.dumps({"text": "t " * 32}) for _ in range(2)))
    return source, data


@pytest.mark.parametrize("algorithm", ["gptq", "autoround"])
def test_qdq_head_calibration_preserves_body_and_reloads(qdq, tmp_path, algorithm):
    from gptqmodel import GPTQModel
    from gptqmodel.recipes.minicpm_v47_qdq_head import _stage_qdq_checkpoint, quantize_minicpm_v47_head
    from gptqmodel.quantization.lm_head import HeadQuantizationConfig, inspect_checkpoint_head
    from gptqmodel.utils.backend import BACKEND

    source, data = qdq
    original = hashlib.sha256((source / "pytorch_model.bin").read_bytes()).hexdigest()
    stage = tmp_path / "reference"
    stage.mkdir()
    _stage_qdq_checkpoint(source, source, stage, algorithm)
    body = load_file(stage / "model-backbone.safetensors")
    target = tmp_path / "head_w5"
    result = quantize_minicpm_v47_head(
        model_dir=str(source), tokenizer_dir=str(source), output_dir=str(target),
        head={"bits": 5, "algorithm": algorithm, "iters": 2,
              "row_chunk_size": 128, "batch_tokens": 16},
        calibration_jsonl=str(data), nsamples=2, seqlen=16, device="cpu")
    assert result.output_dir == str(target)
    assert hashlib.sha256((source / "pytorch_model.bin").read_bytes()).hexdigest() == original
    saved = load_file(target / "model-backbone.safetensors")
    assert saved.keys() == body.keys()
    for name in body:
        torch.testing.assert_close(saved[name], body[name], rtol=0, atol=0)
    actual = inspect_checkpoint_head(target, HeadQuantizationConfig(bits=5))
    assert actual["actual"]["bits"] == 5
    model = GPTQModel.load(str(target), device="cpu", backend=BACKEND.TORCH)
    assert model.model.lm_head.maxq == 31
    with torch.inference_mode():
        assert torch.isfinite(model.model(torch.ones(1, 16, dtype=torch.long)).logits).all()
    reused = quantize_minicpm_v47_head(model_dir=str(target), tokenizer_dir=str(source),
                                     head={"bits": 5}, device="cpu")
    assert reused.output_dir == str(target) and reused.method == "skip"


@pytest.mark.parametrize("algorithm", ["gptq", "autoround"])
def test_full_recipe_w5_plan_is_one_final_checkpoint(tmp_path, algorithm):
    from gptqmodel.recipes.minicpm_v_4_7 import (
        quantize_minicpm_v_4_7_autoround, quantize_minicpm_v_4_7_gptq,
    )
    recipe = quantize_minicpm_v_4_7_gptq if algorithm == "gptq" else quantize_minicpm_v_4_7_autoround
    result = recipe(model_dir=str(tmp_path / "native"), output_dir=str(tmp_path / "final"),
                    head={"bits": 5}, dry_run=True)
    assert result.output_dir == str(tmp_path / "final")
    assert result.precision["head_bits"] == 5
    assert result.provenance["head"]["algorithm"] == algorithm
    assert result.provenance["stages"] == ["backbone", "head"]
    assert not (tmp_path / "final").exists()


def test_one_call_runs_dense_backbone_and_w5_head(tmp_path):
    from gptqmodel.recipes.minicpm_v_4_7 import quantize_minicpm_v_4_7_gptq

    source, data = _make_qdq(tmp_path, False)
    original = torch.load(source / 'pytorch_model.bin', weights_only=True)
    target = tmp_path / 'full_w5'
    result = quantize_minicpm_v_4_7_gptq(
        model_dir=str(source), llm_backbone_dir=str(source), output_dir=str(target),
        calibration_jsonl=str(data), nsamples=2, seqlen=16, device='cpu',
        head={'bits': 5, 'algorithm': 'gptq', 'iters': 2})
    assert result.output_dir == str(target)
    config = json.loads((target / 'config.json').read_text())
    assert config['quantization_config']['meta']['lm_head_quantization']['bits'] == 5
    body = load_file(target / 'model-backbone.safetensors')
    name = 'model.layers.1.self_attn.q_proj.weight'
    assert not torch.equal(body[name].float(), original[name].float())
    torch.testing.assert_close(body['model.embed_tokens.weight'].float(),
                               original['model.embed_tokens.weight'].to(body['model.embed_tokens.weight'].dtype).float(),
                               rtol=0, atol=0)
    assert not list(tmp_path.glob('full_w5.backbone-*'))
