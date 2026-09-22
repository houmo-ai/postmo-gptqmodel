"""Real dense MTP calibration, packed reload, and preservation of the base."""
import json
import pytest
import torch
from safetensors.torch import save_file
from gptqmodel.quantization.qwen35_mtp import (
    Qwen35MTP, MTPQuantizationConfig, CheckpointTensors,
    calibrate_mtp_experts, inspect_mtp_checkpoint, save_mtp_checkpoint,
)


@pytest.mark.parametrize("bits", [4, 5, 8])
@pytest.mark.parametrize("algorithm", ["gptq", "autoround"])
def test_dense_calibration_roundtrip(tmp_path, bits, algorithm):
    from transformers import Qwen3_5TextConfig
    torch.set_num_threads(2)
    torch.manual_seed(31)
    cfg = Qwen3_5TextConfig(hidden_size=128, intermediate_size=256, num_attention_heads=4,
                          num_key_value_heads=2, head_dim=32, vocab_size=256,
                          num_hidden_layers=1, layer_types=["full_attention"],
                          rope_parameters={"rope_type": "default", "rope_theta": 10000,
                                           "partial_rotary_factor": 0.5})
    config = {"model_type": "qwen3_5", "text_config": cfg.to_dict()}
    model = Qwen35MTP(config).half().eval()
    with torch.no_grad():
        for name, param in model.named_parameters():
            param.copy_(torch.zeros_like(param) if "norm" in name else torch.randn_like(param) * 0.025)
    source = tmp_path / "source"
    source.mkdir()
    state = {"mtp." + k: v.contiguous() for k, v in model.state_dict().items()}
    state["lm_head.weight"] = torch.randn(256, 128).half()
    state["model.language_model.embed_tokens.weight"] = torch.randn(256, 128).half()
    save_file(state, source / "model.safetensors")
    (source / "config.json").write_text(json.dumps(config))
    model = Qwen35MTP.from_pretrained(source)
    records = [{"embedding": torch.randn(1, 16, 128).half(), "hidden": torch.randn(1, 16, 128).half()}]
    options = MTPQuantizationConfig(block_bits=bits, algorithm=algorithm, iters=2, batch_tokens=16)
    payload, metrics = calibrate_mtp_experts(model, records, options, device="cpu")
    assert len(metrics) == 7
    assert all(item["algorithm"] == algorithm for item in metrics.values())
    target = tmp_path / "packed"
    save_mtp_checkpoint(source, target, payload, options, {"test": True})
    assert inspect_mtp_checkpoint(target, options)["block_bits"] == bits
    loaded = Qwen35MTP.from_pretrained(target)
    for name, value in model.named_parameters():
        other = dict(loaded.named_parameters())[name]
        assert torch.equal(value, other), name
        assert value.stride() == other.stride(), name
    with torch.no_grad():
        expected = model(records[0]["embedding"], records[0]["hidden"])
        actual = loaded(records[0]["embedding"], records[0]["hidden"])
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    before, after = CheckpointTensors(source), CheckpointTensors(target)
    for key in before.mapping:
        if key in after.mapping:
            assert torch.equal(before.get(key), after.get(key)), key
    with pytest.raises(ValueError, match="contract"):
        inspect_mtp_checkpoint(target, MTPQuantizationConfig(block_bits=5 if bits != 5 else 4, algorithm=algorithm))
