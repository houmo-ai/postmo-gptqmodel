import tempfile

import torch
import pytest
from transformers import Qwen3MoeConfig, Qwen3MoeForCausalLM

from gptqmodel import BACKEND, GPTQModel
from gptqmodel.nn_modules.qlinear.torch import TorchQuantLinear
from gptqmodel.quantization import FORMAT
from gptqmodel.quantization.config import AutoRoundQuantizeConfig, AutoRoundVersion, ExpertsRoutingBypass, METHOD, MoEConfig


def _build_tiny_qwen3_moe():
    cfg = Qwen3MoeConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        moe_intermediate_size=64,
        num_attention_heads=4,
        num_key_value_heads=4,
        num_hidden_layers=1,
        num_experts=4,
        num_experts_per_tok=2,
        max_position_embeddings=32,
    )
    model = Qwen3MoeForCausalLM(cfg).eval()
    calibration = [
        {
            "input_ids": torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]], dtype=torch.long),
            "attention_mask": torch.ones((1, 12), dtype=torch.long),
        },
        {
            "input_ids": torch.tensor([[12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1]], dtype=torch.long),
            "attention_mask": torch.ones((1, 12), dtype=torch.long),
        },
    ]
    return cfg, model, calibration


@pytest.mark.parametrize("version", [AutoRoundVersion.V1, AutoRoundVersion.V2])
def test_auto_round_quantizes_tiny_qwen3_moe_end_to_end(version):
    torch.manual_seed(0)

    cfg, model, calibration = _build_tiny_qwen3_moe()

    with tempfile.TemporaryDirectory() as native_dir:
        model.save_pretrained(native_dir)

        qcfg = AutoRoundQuantizeConfig(
            bits=4,
            group_size=16,
            sym=True,
            format=FORMAT.GPTQ,
            desc_act=False,
            iters=1,
            lr=0.05,
            minmax_lr=0.05,
            device="cpu",
            auto_round_version=version,
            moe=MoEConfig(routing=ExpertsRoutingBypass(batch_size=1)),
            offload_to_disk=False,
        )
        qmodel = GPTQModel.load(native_dir, quantize_config=qcfg)
        qmodel.quantize(calibration, batch_size=1)

        attn_proj = qmodel.model.model.layers[0].self_attn.q_proj
        expert_proj = qmodel.model.model.layers[0].mlp.experts[0].gate_proj

        assert isinstance(attn_proj, TorchQuantLinear)
        assert isinstance(expert_proj, TorchQuantLinear)
        assert hasattr(attn_proj, "bias")
        assert attn_proj.bias is None

        output = qmodel.model(
            input_ids=torch.tensor([[1, 2, 3, 4]], dtype=torch.long),
            attention_mask=torch.ones((1, 4), dtype=torch.long),
        )

        with tempfile.TemporaryDirectory() as quant_dir:
            qmodel.save(quant_dir)
            reloaded = GPTQModel.load(quant_dir, backend=BACKEND.TORCH, device="cpu")
            reloaded_output = reloaded.model(
                input_ids=torch.tensor([[1, 2, 3, 4]], dtype=torch.long),
                attention_mask=torch.ones((1, 4), dtype=torch.long),
            )

        assert qmodel.quantized is True
        assert len(qmodel.quant_log) == 16
        assert qmodel.quantize_config.auto_round_version == version
        assert tuple(output.logits.shape) == (1, 4, cfg.vocab_size)
        assert torch.isfinite(output.logits).all()
        assert reloaded.quantize_config.quant_method == METHOD.AUTO_ROUND
        assert reloaded.quantize_config.format == FORMAT.GPTQ
        assert reloaded.quantize_config.auto_round_version == version
        assert isinstance(reloaded.model.model.layers[0].self_attn.q_proj, TorchQuantLinear)
        assert torch.isfinite(reloaded_output.logits).all()
        assert tuple(reloaded_output.logits.shape) == tuple(output.logits.shape)


def test_auto_round_quantizes_tiny_qwen3_moe_with_rotation():
    torch.manual_seed(0)

    cfg, model, calibration = _build_tiny_qwen3_moe()

    with tempfile.TemporaryDirectory() as native_dir:
        model.save_pretrained(native_dir)

        qcfg = AutoRoundQuantizeConfig(
            bits=4,
            group_size=16,
            sym=True,
            format=FORMAT.GPTQ,
            desc_act=False,
            iters=0,
            rotation="hadamard",
            device="cpu",
            moe=MoEConfig(routing=ExpertsRoutingBypass(batch_size=1)),
            offload_to_disk=False,
        )
        qmodel = GPTQModel.load(native_dir, quantize_config=qcfg)
        qmodel.quantize(calibration, batch_size=1)
        router = qmodel.model.model.layers[0].mlp.gate

        output = qmodel.model(
            input_ids=torch.tensor([[1, 2, 3, 4]], dtype=torch.long),
            attention_mask=torch.ones((1, 4), dtype=torch.long),
        )

        assert qmodel.quantized is True
        assert qmodel.quantize_config.rotation == "hadamard"
        assert isinstance(qmodel.model.model.layers[0].self_attn.q_proj, TorchQuantLinear)
        assert router.weight.is_inference() is False
        assert tuple(output.logits.shape) == (1, 4, cfg.vocab_size)
        assert torch.isfinite(output.logits).all()
