"""XH2 MLA math and real Ling quantize/save/reload coverage (no full weights)."""

import copy
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from gptqmodel.models.definitions.ling3_flash_mla import (
    XH2_MLA_LAYOUT,
    _build_absorbed,
    absorb_ling_mla,
    install_ling_mla_class,
    validate_mla_layout,
)


def _attention(q_rank=None, gate="head_wise", bias=True):
    from transformers.dynamic_module_utils import get_class_from_dynamic_module

    from gptqmodel.models.definitions.ling_3_flash import load_ling_3_flash_config

    source = Path(
        os.environ.get("LING3_FLASH_MODEL_DIR", "/data01/datasets/Ling-3.0-flash")
    )
    if not (source / "modeling_bailing_moe_v3.py").is_file():
        pytest.skip("Set LING3_FLASH_MODEL_DIR to Ling checkpoint-side config/code")
    config = load_ling_3_flash_config(str(source))
    for name, value in dict(
        hidden_size=64,
        num_attention_heads=2,
        num_key_value_heads=2,
        num_hidden_layers=2,
        kv_lora_rank=64,
        q_lora_rank=q_rank,
        qk_nope_head_dim=32,
        qk_rope_head_dim=16,
        qk_head_dim=48,
        v_head_dim=32,
        use_qkv_bias=bias,
        gated_attention_proj_granularity_type=gate,
        attention_dropout=0.0,
        rope_interleave=True,
    ).items():
        setattr(config, name, value)
    cls = get_class_from_dynamic_module(
        "modeling_bailing_moe_v3.BailingMoeV3MultiLatentAttention", str(source)
    )
    torch.manual_seed(41)
    return cls(config, 0).eval()


@pytest.mark.parametrize("q_rank", [None, 64])
@pytest.mark.parametrize("gate", [None, "head_wise"])
@pytest.mark.parametrize("bias", [False, True])
def test_float_prefill_and_cached_decode(q_rank, gate, bias):
    from transformers.cache_utils import DynamicCache

    raw = _attention(q_rank, gate, bias)
    fused = copy.deepcopy(raw)
    install_ling_mla_class(type(fused))
    _build_absorbed(fused, from_weights=True)
    raw_cache, fused_cache = DynamicCache(), DynamicCache()
    hidden = torch.randn(2, 5, 64)
    angles = torch.randn(2, 5, 8).repeat(1, 1, 2)
    cos, sin = angles.cos(), angles.sin()
    for begin, end in [(0, 3), (3, 4), (4, 5)]:
        mask = torch.zeros(1, 1, end - begin, end)
        future = torch.arange(end)[None, :] > torch.arange(begin, end)[:, None]
        mask.masked_fill_(future, torch.finfo(mask.dtype).min)
        args = (hidden[:, begin:end], (cos[:, begin:end], sin[:, begin:end]), mask)
        expected, expected_p, _ = raw(*args, past_key_values=raw_cache)
        actual, actual_p, _ = fused(*args, past_key_values=fused_cache)
        torch.testing.assert_close(actual_p, expected_p, atol=1e-6, rtol=1e-5)
        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
        cache = fused_cache.layers[0]
        assert tuple(cache.keys.shape) == (2, 1, end, 80)
        assert tuple(cache.values.shape) == (2, 1, end, 64)
        assert (
            cache.keys.untyped_storage().data_ptr()
            != cache.values.untyped_storage().data_ptr()
        )
    assert fused.scaling == raw.scaling
    with pytest.raises(ValueError, match="foreign cache"):
        fused(hidden[:, :1], (cos[:, :1], sin[:, :1]), None, past_key_values=raw_cache)
    # The reference remains trainable through the absorbed boundaries.
    fused.zero_grad()
    out = fused(hidden, (cos, sin), None)[0]
    out.square().mean().backward()
    for name in ("q_absorbed_proj", "q_rope_proj", "absorbed_dense"):
        gradient = getattr(fused, name).weight.grad
        assert gradient is not None and torch.isfinite(gradient).all()


def test_layout_rejects_noncommuting_gate_and_packed_source():
    with pytest.raises(ValueError, match="element_wise"):
        _build_absorbed(_attention(gate="element_wise"), from_weights=True)
    module = _attention()
    module.q_proj.quant_weight = torch.ones_like(module.q_proj.weight)
    with pytest.raises(ValueError, match="original floating weights"):
        _build_absorbed(module, from_weights=True)
    with pytest.raises(ValueError, match="Unsupported Ling MLA"):
        validate_mla_layout("xh2_absorbed_v2")


@pytest.mark.parametrize(
    "extra,match",
    [
        ({"data_type": "int_sym"}, "positive scales"),
        ({"offload_to_disk": True}, "materialized FP"),
        ({"mla_weight_layout": "xh2_absorbed_v999"}, "Unsupported Ling MLA"),
    ],
)
def test_xh2_recipe_rejects_incompatible_options(tmp_path, extra, match):
    from gptqmodel.recipes.ling3_flash import quantize_ling3_flash

    with pytest.raises(ValueError, match=match):
        quantize_ling3_flash(
            **{
                "model_dir": str(tmp_path / "model"),
                "output_dir": str(tmp_path / "out"),
                "method": "autoround",
                "mla_weight_layout": XH2_MLA_LAYOUT,
                "dry_run": True,
                **extra,
            }
        )
    assert not (tmp_path / "out").exists()


def test_absorbed_fp_checkpoint_loads_with_plain_hf_and_rebuilds_rope(tmp_path):
    from test_ling3_flash_autoround import _save_real_tiny_ling
    from transformers import AutoModelForCausalLM

    from gptqmodel.models.definitions.ling3_flash_mla import (
        finalize_ling_mla_checkpoint,
    )
    from gptqmodel.models.definitions.ling_3_flash import (
        load_ling_3_flash_config,
        refresh_ling_rotary_buffers,
    )

    source, output = tmp_path / "source", tmp_path / "fused"
    _save_real_tiny_ling(source, 1)
    model = AutoModelForCausalLM.from_pretrained(
        str(source),
        config=load_ling_3_flash_config(str(source)),
        trust_remote_code=True,
        dtype=torch.bfloat16,
    )
    rope = model.model.rotary_emb
    rope.inv_freq.fill_(float("nan"))
    refresh_ling_rotary_buffers(model)
    expected_freq, _ = rope.rope_init_fn(rope.config, torch.device("cpu"))
    torch.testing.assert_close(rope.inv_freq, expected_freq, atol=0, rtol=0)
    assert rope.original_inv_freq is rope.inv_freq
    absorb_ling_mla(model)
    model.save_pretrained(output)
    finalize_ling_mla_checkpoint(output)
    restored, info = AutoModelForCausalLM.from_pretrained(
        str(output),
        trust_remote_code=True,
        dtype=torch.bfloat16,
        output_loading_info=True,
    )
    assert not any(
        info.get(key)
        for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")
    )
    torch.testing.assert_close(
        restored.model.rotary_emb.inv_freq, expected_freq, atol=0, rtol=0
    )
    expected, actual = model.state_dict(), restored.state_dict()
    assert expected.keys() == actual.keys()
    assert all(torch.equal(tensor, actual[name]) for name, tensor in expected.items())


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="Real KDA requires CUDA")
@pytest.mark.parametrize("method", ["autoround", "gptq"])
def test_real_absorbed_quantize_load_resave(tmp_path, method, monkeypatch):
    from accelerate import init_empty_weights
    from safetensors.torch import load_file
    from test_ling3_flash_autoround import _save_real_tiny_ling
    from transformers import AutoModelForCausalLM, modeling_utils

    from gptqmodel import BACKEND, GPTQModel
    from gptqmodel.models.definitions.ling_3_flash import load_ling_3_flash_config

    # GPTQModel's writer replaces global HF/Torch initialization helpers.
    # Do not let a resave contaminate the next test's fresh FP source model.
    for name in ("normal_", "uniform_", "kaiming_uniform_"):
        monkeypatch.setattr(torch.nn.init, name, getattr(torch.nn.init, name))
    monkeypatch.setattr(
        modeling_utils,
        "_init_weights",
        getattr(modeling_utils, "_init_weights", True),
        raising=False,
    )

    source, output = tmp_path / "source", tmp_path / "out"
    # GPTQ's packed zero layout requires output widths aligned to 32, including
    # the gate. Keep the real checkpoint's 32 heads; shrink hidden/MoE instead.
    _save_real_tiny_ling(
        source,
        1,
        num_attention_heads=32,
        num_key_value_heads=32,
        qk_rope_head_dim=64,
        qk_head_dim=96,
    )
    source_code = (source / "modeling_bailing_moe_v3.py").read_bytes()
    calibration = tmp_path / "calibration.jsonl"
    calibration.write_text(
        "\n".join(
            json.dumps({"text": " ".join(str((i + j) % 120 + 4) for i in range(64))})
            for j in range(4)
        )
        + "\n"
    )
    script = (
        Path(__file__).resolve().parents[1]
        / "examples/quantization/examples/example_ling3_flash.py"
    )
    command = [
        sys.executable,
        str(script),
        "--model",
        str(source),
        "--out",
        str(output),
        "--method",
        method,
        "--mla-weight-layout",
        XH2_MLA_LAYOUT,
        "--calibration-jsonl",
        str(calibration),
        "--nsamples",
        "2",
        "--seqlen",
        "32",
        "--iters",
        "2",
        "--device-map",
        "0",
    ]
    result = subprocess.run(command, capture_output=True, text=True, timeout=600)
    (tmp_path / "quantize.log").write_text(result.stdout + result.stderr)
    assert result.returncode == 0, result.stdout[-10000:] + result.stderr[-10000:]
    assert (source / "modeling_bailing_moe_v3.py").read_bytes() == source_code
    source_state, state = (
        load_file(source / "model.safetensors"),
        load_file(output / "model.safetensors"),
    )
    mtp_names = [name for name in source_state if name.startswith("model.layers.2.")]
    assert mtp_names
    for name in mtp_names:
        assert torch.equal(source_state[name].float(), state[name].float()), name
    # A plain HF constructor (including meta/streamed load) must use the same ABI.
    config = load_ling_3_flash_config(str(output))
    with init_empty_weights():
        empty = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
    assert empty.model.layers[1].attention.q_absorbed_proj.weight.is_meta
    assert not hasattr(empty.model.layers[1].attention, "kv_b_proj")
    loaded = GPTQModel.load(
        str(output),
        backend=BACKEND.TORCH,
        device="cpu",
        dtype=torch.float16,
        trust_remote_code=True,
    )
    rope = loaded.model.model.rotary_emb
    expected_freq, _ = rope.rope_init_fn(rope.config, torch.device("cpu"))
    torch.testing.assert_close(rope.inv_freq, expected_freq, atol=0, rtol=0)
    initial = {
        name: {
            key: getattr(layer, key).clone()
            for key in ("qweight", "qzeros", "scales", "g_idx")
        }
        for name, layer in loaded.model.named_modules()
        if hasattr(layer, "qweight")
    }
    for name in ("q_absorbed_proj", "q_rope_proj", "absorbed_dense"):
        assert f"model.layers.1.attention.{name}" in initial
    assert loaded.model.model.layers[1].attention.q_absorbed_proj.bits == 8
    assert loaded.model.model.layers[1].mlp.experts[0].gate_proj.bits == 4
    assert all(
        torch.isfinite(values["scales"]).all() and (values["scales"] > 0).all()
        for values in initial.values()
    )
    assert not any(name.startswith("model.layers.2.") for name in initial)
    # Check loaded quantized MLA inference, including token-by-token decoding.
    from transformers.cache_utils import DynamicCache

    attention = loaded.model.model.layers[1].attention
    x = torch.randn(1, 3, 64, dtype=torch.float16)
    cos, sin = torch.ones(1, 3, 64, dtype=x.dtype), torch.zeros(1, 3, 64, dtype=x.dtype)
    prefill = attention(x, (cos, sin), None)[0]
    cache = DynamicCache()
    decode = torch.cat(
        [
            attention(
                x[:, i : i + 1],
                (cos[:, i : i + 1], sin[:, i : i + 1]),
                None,
                past_key_values=cache,
            )[0]
            for i in range(3)
        ],
        dim=1,
    )
    torch.testing.assert_close(prefill, decode, atol=2e-3, rtol=2e-3)
    resaved = tmp_path / "resaved"
    loaded.save(str(resaved))
    restored = GPTQModel.load(
        str(resaved),
        backend=BACKEND.TORCH,
        device="cpu",
        dtype=torch.float16,
        trust_remote_code=True,
    )
    actual = {
        name: layer
        for name, layer in restored.model.named_modules()
        if hasattr(layer, "qweight")
    }
    assert actual.keys() == initial.keys()
    for name, buffers in initial.items():
        for key, expected in buffers.items():
            assert torch.equal(expected, getattr(actual[name], key)), (name, key)
    assert (
        json.loads((resaved / "config.json").read_text())["ling_mla_layout"]
        == XH2_MLA_LAYOUT
    )
