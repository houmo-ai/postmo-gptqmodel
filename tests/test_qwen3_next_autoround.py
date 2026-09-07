"""Real tiny Qwen3-Next AutoRound training/packing/loading, plus precision policy."""

from __future__ import annotations

import json
import subprocess

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from gptqmodel.recipes.qwen3_next import quantize_qwen3_next
from gptqmodel.recipes.qwen3_next_autoround import build_layer_config, build_parser


@pytest.fixture(autouse=True)
def restore_loading_globals(monkeypatch):
    # GPTQModel's existing resave path disables initialization globally. Keep
    # later tiny-model tests independent of its construction/loading patches.
    from transformers import modeling_utils
    from transformers.models.qwen3_next import modeling_qwen3_next

    for name in ("normal_", "uniform_", "kaiming_uniform_"):
        monkeypatch.setattr(torch.nn.init, name, getattr(torch.nn.init, name))
    monkeypatch.setattr(
        modeling_utils,
        "_init_weights",
        getattr(modeling_utils, "_init_weights", True),
        raising=False,
    )
    monkeypatch.setattr(
        modeling_qwen3_next,
        "Qwen3NextSparseMoeBlock",
        modeling_qwen3_next.Qwen3NextSparseMoeBlock,
    )


def _tiny_model():
    import defuser
    from transformers import Qwen3NextConfig, Qwen3NextForCausalLM

    from gptqmodel.models.definitions.qwen3_next import (
        _patch_defuser_qwen3_next_registry,
    )

    config = Qwen3NextConfig(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        moe_intermediate_size=128,
        shared_expert_intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=32,
        linear_num_key_heads=2,
        linear_num_value_heads=2,
        linear_key_head_dim=32,
        linear_value_head_dim=32,
        num_experts=2,
        num_experts_per_tok=2,
        decoder_sparse_step=2,
        layer_types=["linear_attention", "full_attention"],
        max_position_embeddings=128,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
    )
    model = Qwen3NextForCausalLM(config).eval()
    _patch_defuser_qwen3_next_registry()
    defuser.convert_model(model, cleanup_original=False)
    return model


@pytest.mark.parametrize(
    "overrides", [[], ["--expert-bits", "8", "--self-attn-bits", "16"]]
)
def test_precision_policy_covers_hybrid_attention_and_moe(overrides):
    model = _tiny_model()
    args = build_parser().parse_args(["--model", "raw", "--out", "out", *overrides])
    policy = build_layer_config(model, args)
    prefix = "model.layers."
    assert policy[prefix + "0.linear_attn.in_proj_qkvz"]["bits"] == 8
    assert policy[prefix + "0.linear_attn.out_proj"]["bits"] == 8
    assert policy[prefix + "0.linear_attn.in_proj_ba"] == {
        "bits": 16,
        "data_type": "fp",
    }
    assert policy[prefix + "0.mlp.down_proj"]["bits"] == 4
    assert policy[prefix + "1.self_attn.q_proj"]["bits"] == args.self_attn_bits
    assert policy[prefix + "1.mlp.experts.0.gate_proj"]["bits"] == args.expert_bits
    assert policy[prefix + "1.mlp.shared_expert.gate_proj"]["bits"] == 8
    for name in (prefix + "1.mlp.shared_expert_gate", "lm_head"):
        assert policy[name] == {"bits": 16, "data_type": "fp"}
    # Transformers/defuser may use a custom TopKRouter, not nn.Linear.
    router = model.model.layers[1].mlp.gate
    assert router.weight.is_floating_point() and not hasattr(router, "qweight")
    if isinstance(router, torch.nn.Linear):
        assert policy[prefix + "1.mlp.gate"] == {"bits": 16, "data_type": "fp"}
    assert all(
        value["data_type"] == "int_sym_xh"
        for value in policy.values()
        if value["bits"] < 16
    )


def _save_tiny_checkpoint(path, with_mtp):
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import PreTrainedTokenizerFast

    torch.manual_seed(42)
    model = _tiny_model().to(torch.bfloat16)
    model.save_pretrained(path)
    tokenizer = Tokenizer(
        WordLevel(vocab={str(i): i for i in range(128)}, unk_token="3")
    )
    tokenizer.pre_tokenizer = Whitespace()
    PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        pad_token="0",
        bos_token="1",
        eos_token="2",
        unk_token="3",
    ).save_pretrained(path)
    if with_mtp:
        weights = load_file(path / "model.safetensors")
        weights["mtp.fc.weight"] = torch.arange(16, dtype=torch.bfloat16).reshape(4, 4)
        save_file(weights, path / "model.safetensors", metadata={"format": "pt"})


def test_incomplete_source_weights_fail_before_quantization(tmp_path):
    source, output = tmp_path / "source", tmp_path / "quantized"
    _save_tiny_checkpoint(source, with_mtp=False)
    weights = load_file(source / "model.safetensors")
    weights.pop("model.layers.1.mlp.experts.0.gate_proj.weight")
    save_file(weights, source / "model.safetensors", metadata={"format": "pt"})
    with pytest.raises(subprocess.CalledProcessError):
        quantize_qwen3_next(
            model_dir=str(source),
            output_dir=str(output),
            method="autoround",
            iters=0,
            device_map="cpu",
        )
    log = (output / "qwen3_next_autoround_quantize.log").read_text()
    assert "Incomplete Qwen3-Next checkpoint load" in log
    assert "model.layers.1.mlp.experts.0.gate_proj.weight" in log
    assert not (output / "model.safetensors").exists()


@pytest.mark.cuda
@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="Real AutoRound GatedDeltaNet smoke requires CUDA",
)
@pytest.mark.parametrize(
    "with_mtp,self_attn_bits,expert_bits", [(False, 8, 4), (True, 8, 4), (False, 16, 8)]
)
def test_real_autoround_quantize_save_load(
    tmp_path, with_mtp, self_attn_bits, expert_bits
):
    from gptqmodel import BACKEND, GPTQModel

    source, output = tmp_path / "source", tmp_path / "quantized"
    _save_tiny_checkpoint(source, with_mtp)
    calibration = tmp_path / "calibration.jsonl"
    calibration.write_text(
        "\n".join(
            json.dumps({"text": " ".join(str((i + j) % 120 + 4) for i in range(64))})
            for j in range(4)
        )
        + "\n"
    )
    result = quantize_qwen3_next(
        model_dir=str(source),
        output_dir=str(output),
        method="autoround",
        data_type="int_sym_xh",
        iters=2,
        nsamples=2,
        seqlen=32,
        batch_size=1,
        self_attn_bits=self_attn_bits,
        expert_bits=expert_bits,
        calibration_jsonl=str(calibration),
        device_map="0",
        deterministic=with_mtp,
        env={"OMP_NUM_THREADS": "4", "TOKENIZERS_PARALLELISM": "false"},
    )
    assert result.has_mtp is with_mtp
    assert result.provenance["mtp_tensor_count"] == int(with_mtp)
    if with_mtp:
        assert result.provenance["attention_implementation"] == "eager"
        index = json.loads((output / "model.safetensors.index.json").read_text())[
            "weight_map"
        ]
        with safe_open(output / index["mtp.fc.weight"], framework="pt") as reader:
            assert torch.equal(
                reader.get_tensor("mtp.fc.weight"),
                torch.arange(16, dtype=torch.bfloat16).reshape(4, 4),
            )
    config = json.loads((output / "config.json").read_text())["quantization_config"]
    assert config["data_type"] == "int_sym_xh"
    assert config["checkpoint_format"] == "gptq_v2"
    assert config["quant_method"] == "gptq"
    assert config["iters"] == 2
    assert config["nsamples"] == 2

    # The existing GPTQModel writer materializes a loaded model in FP16. Use
    # the same inference dtype on both sides to test serialization, not BF16
    # versus FP16 arithmetic.
    loaded = GPTQModel.load(
        str(output), device="cuda:0", backend=BACKEND.TORCH, dtype=torch.float16
    )
    assert loaded.quantize_config.data_type == "int_sym_xh"
    layers = loaded.model.model.layers
    for projection, bits in (
        (layers[0].linear_attn.in_proj_qkvz, 8),
        (layers[0].mlp.gate_proj, 4),
        (layers[1].self_attn.q_proj, self_attn_bits),
        (layers[1].mlp.experts[0].gate_proj, expert_bits),
        (layers[1].mlp.shared_expert.gate_proj, 8),
    ):
        if bits == 16:
            assert isinstance(projection, torch.nn.Linear)
            assert not hasattr(projection, "qweight")
            continue
        assert projection.bits == bits
        assert torch.isfinite(projection.scales).all()
        assert (projection.scales > 0).all()
        shifts = torch.arange(32 // bits, device=projection.qzeros.device) * bits
        zero_points = (projection.qzeros.to(torch.int64).unsqueeze(-1) >> shifts) & (
            (1 << bits) - 1
        )
        assert (zero_points == 1 << (bits - 1)).all()
    assert isinstance(layers[0].linear_attn.in_proj_ba, torch.nn.Linear)
    assert layers[1].mlp.gate.weight.is_floating_point()
    assert not hasattr(layers[1].mlp.gate, "qweight")
    assert isinstance(layers[1].mlp.shared_expert_gate, torch.nn.Linear)
    inputs = torch.tensor([[4, 5, 6, 7]], device="cuda:0")
    with torch.no_grad():
        logits = loaded.model(input_ids=inputs, use_cache=False).logits
    assert logits.shape == (1, 4, 128)
    assert torch.isfinite(logits).all()
    packed = {
        name: value.detach().cpu().clone()
        for name, value in loaded.model.state_dict().items()
        if name.endswith((".qweight", ".qzeros", ".scales", ".g_idx"))
    }

    resaved = tmp_path / "resaved"
    loaded.save(str(resaved))
    if with_mtp:
        index = json.loads((resaved / "model.safetensors.index.json").read_text())[
            "weight_map"
        ]
        with safe_open(resaved / index["mtp.fc.weight"], framework="pt") as reader:
            assert torch.equal(
                reader.get_tensor("mtp.fc.weight"),
                torch.arange(16, dtype=torch.bfloat16).reshape(4, 4),
            )
    again = GPTQModel.load(
        str(resaved), device="cuda:0", backend=BACKEND.TORCH, dtype=torch.float16
    )
    assert again.quantize_config.data_type == "int_sym_xh"
    for name, original in packed.items():
        assert torch.equal(again.model.state_dict()[name].cpu(), original), name
    with torch.no_grad():
        logits_again = again.model(input_ids=inputs, use_cache=False).logits
    torch.testing.assert_close(logits_again, logits, rtol=0, atol=0)
