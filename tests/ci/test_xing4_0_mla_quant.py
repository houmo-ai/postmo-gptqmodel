"""Actual Xing GPTQ/AutoRound absorption, packing and checkpoint reload."""

import json

import pytest
import torch

from _xing_fixture import save_tiny_xing
from gptqmodel.models.definitions.xing4_0_mla import XH2_MLA_LAYOUT


@pytest.mark.parametrize("method", ["gptq", "auto_round"])
def test_real_absorbed_quantize_load_resave(tmp_path, method, monkeypatch):
    from transformers import modeling_utils
    from gptqmodel import BACKEND, GPTQModel
    from gptqmodel.recipes import quantize_xing4_0

    assert torch.cuda.is_available(), "quant lane requires a dedicated CUDA device"
    # The writer changes global init functions; restore them for other fixtures.
    for name in ("normal_", "uniform_", "kaiming_uniform_"):
        monkeypatch.setattr(torch.nn.init, name, getattr(torch.nn.init, name))
    monkeypatch.setattr(
        modeling_utils,
        "_init_weights",
        getattr(modeling_utils, "_init_weights", True),
        raising=False,
    )
    source, output = tmp_path / "source", tmp_path / "quantized"
    save_tiny_xing(source)
    source_config_path = source / "config.json"
    source_config = json.loads(source_config_path.read_text())
    source_config.pop("head_dim", None)
    source_config_path.write_text(json.dumps(source_config))
    code = (source / "modeling_xing4_0.py").read_bytes()
    result = quantize_xing4_0(
        model_dir=str(source),
        output_dir=str(output),
        method=method,
        calibration_data=[" ".join(str(i % 120 + 4) for i in range(256))] * 4,
        nsamples=4,
        seqlen=32,
        batch_size=1,
        device="cuda:0",
        auto_round_iters=2,
        offload_to_disk=False,
        hessian_mse=False,
    )
    assert result.precision["mla_weight_layout"] == XH2_MLA_LAYOUT
    assert "head_dim" not in json.loads((output / "config.json").read_text())
    assert (source / "modeling_xing4_0.py").read_bytes() == code
    loaded = GPTQModel.load(
        str(output),
        backend=BACKEND.TORCH,
        device="cpu",
        dtype=torch.float16,
        trust_remote_code=True,
    )
    assert loaded.model.config.mla_transform["weight_dtype"] == "torch.float16"
    bias = loaded.model.model.layers[1].mlp.gate.e_score_correction_bias
    assert bias.dtype == torch.float32
    assert torch.equal(bias, torch.tensor([0.1234567, -0.2345678], dtype=torch.float32))
    initial = {
        name: {
            key: getattr(layer, key).clone()
            for key in ("qweight", "qzeros", "scales", "g_idx")
        }
        for name, layer in loaded.model.named_modules()
        if hasattr(layer, "qweight")
    }
    assert initial
    for index in range(2):
        attention = loaded.model.model.layers[index].self_attn
        for name in ("q_absorbed_proj", "q_rope_proj", "o_proj"):
            assert f"model.layers.{index}.self_attn.{name}" in initial
            assert getattr(attention, name).bits == 8
    assert loaded.model.model.layers[1].mlp.experts[0].gate_proj.bits == 4
    for buffers in initial.values():
        assert torch.isfinite(buffers["scales"]).all() and (buffers["scales"] > 0).all()
    tokens = torch.tensor([[4, 5, 6, 7]])
    with torch.no_grad():
        expected = loaded.model(tokens).logits
    assert torch.isfinite(expected).all()
    from transformers.cache_utils import DynamicCache

    cache = DynamicCache()
    with torch.no_grad():
        decoded = torch.cat(
            [
                loaded.model(
                    tokens[:, i : i + 1], past_key_values=cache, use_cache=True
                ).logits
                for i in range(tokens.shape[1])
            ],
            dim=1,
        )
    torch.testing.assert_close(decoded, expected, atol=2e-3, rtol=2e-3)
    quant_config = loaded.model.config.quantization_config
    assert quant_config["meta"].get("auto_round", {}).get(
        "auto_round_enabled", False
    ) is (method == "auto_round")
    resaved = tmp_path / "resaved"
    loaded.save(str(resaved))
    restored = GPTQModel.load(
        str(resaved),
        backend=BACKEND.TORCH,
        device="cpu",
        dtype=torch.float16,
        trust_remote_code=True,
    )
    actual = dict(restored.model.named_modules())
    for name, buffers in initial.items():
        for key, value in buffers.items():
            assert torch.equal(value, getattr(actual[name], key)), (name, key)
    with torch.no_grad():
        torch.testing.assert_close(
            restored.model(tokens).logits, expected, atol=0, rtol=0
        )
