"""Xing producer MLA math, checkpoint ABI and actual quantization lifecycle."""

import copy
import json

import pytest
import torch

from _xing_fixture import save_tiny_xing, tiny_xing_model
from gptqmodel.models.definitions.xing4_0_mla import (
    XH2_MLA_LAYOUT,
    absorb_xing_mla,
    finalize_xing_mla_checkpoint,
    validate_mla_config,
    validate_xing_mla_model,
)


@pytest.mark.parametrize("q_rank", [None, 64])
@pytest.mark.parametrize("interleave", [False, True])
@pytest.mark.parametrize("scaled", [False, True])
def test_absorption_preserves_native_prefill_decode(q_rank, interleave, scaled):
    from transformers.cache_utils import DynamicCache

    rope = {"rope_type": "default", "rope_theta": 10000.0}
    if scaled:
        rope.update(
            rope_type="yarn",
            factor=4.0,
            original_max_position_embeddings=32,
            beta_fast=32,
            beta_slow=1,
            mscale=1.0,
            mscale_all_dim=1.0,
        )
    raw = tiny_xing_model(
        q_lora_rank=q_rank,
        rope_interleave=interleave,
        rope_parameters=rope,
        attention_bias=True,
    )
    fused = absorb_xing_mla(copy.deepcopy(raw))
    source, target = raw.model.layers[0].self_attn, fused.model.layers[0].self_attn
    hidden = torch.randn(2, 5, 64)
    embeddings = raw.model.rotary_emb(hidden, torch.arange(5).expand(2, -1))
    caches = DynamicCache(), DynamicCache()
    for begin, end in ((0, 3), (3, 4), (4, 5)):
        mask = torch.zeros(1, 1, end - begin, end)
        mask.masked_fill_(
            torch.arange(end)[None] > torch.arange(begin, end)[:, None], float("-inf")
        )
        args = (hidden[:, begin:end], tuple(x[:, begin:end] for x in embeddings), mask)
        expected, expected_prob = source(*args, past_key_values=caches[0])
        actual, actual_prob = target(*args, past_key_values=caches[1])
        torch.testing.assert_close(actual_prob, expected_prob, atol=1e-6, rtol=1e-5)
        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
        key, value = caches[1].layers[0].keys, caches[1].layers[0].values
        assert key.shape == (2, 1, end, 96) and value.shape == (2, 1, end, 64)
        assert key.untyped_storage().data_ptr() != value.untyped_storage().data_ptr()
    assert target.scaling == source.scaling
    with pytest.raises(ValueError, match="foreign"):
        target(
            hidden[:, :1],
            tuple(x[:, :1] for x in embeddings),
            None,
            past_key_values=caches[0],
        )
    validate_xing_mla_model(fused)


def test_absorption_rejects_packed_meta_and_metadata_relabel():
    for kind in ("packed", "meta", "relabel"):
        model = tiny_xing_model()
        if kind == "packed":
            model.model.layers[0].self_attn.q_b_proj.quant_weight = torch.ones(1)
        elif kind == "meta":
            model.to("meta")
        else:
            model.config.mla_weight_layout = XH2_MLA_LAYOUT
        with pytest.raises(ValueError):
            absorb_xing_mla(model)
    model = absorb_xing_mla(tiny_xing_model())
    with pytest.raises(ValueError, match="original unquantized"):
        absorb_xing_mla(model)
    model.config.mla_transform["version"] = 2
    with pytest.raises(ValueError, match="version"):
        validate_mla_config(model.config)


def test_absorption_uses_float32_even_under_autocast():
    raw = tiny_xing_model().half()
    attention = raw.model.layers[0].self_attn
    q = attention.q_b_proj.weight.float().reshape(2, 64, 64)
    kv = attention.kv_b_proj.weight.float().reshape(2, 64, 64)
    output = attention.o_proj.weight.float().reshape(64, 2, 32)
    q_expected = (
        torch.einsum("hnc,hni->hci", kv[:, :32], q[:, :32]).reshape(128, 64).half()
    )
    o_expected = (
        torch.einsum("xhv,hvc->xhc", output, kv[:, 32:]).reshape(64, 128).half()
    )
    with torch.autocast("cpu", dtype=torch.bfloat16):
        fused = absorb_xing_mla(raw).model.layers[0].self_attn
    assert torch.equal(fused.q_absorbed_proj.weight, q_expected)
    assert torch.equal(fused.o_proj.weight, o_expected)


def test_absorbed_attention_accepts_padding_and_boolean_masks():
    raw = tiny_xing_model()
    fused = absorb_xing_mla(copy.deepcopy(raw))
    hidden = torch.randn(2, 3, 64)
    embeddings = raw.model.rotary_emb(hidden, torch.arange(3).expand(2, -1))
    padding = torch.tensor([[1, 1, 0], [1, 1, 1]])
    keep = (
        torch.ones(3, 3, dtype=torch.bool).tril()[None, None]
        & padding[:, None, None].bool()
    )
    additive = torch.zeros(2, 1, 3, 3).masked_fill(~keep, float("-inf"))
    expected = raw.model.layers[0].self_attn(hidden, embeddings, additive)[0]
    for mask in (padding, keep):
        actual = fused.model.layers[0].self_attn(hidden, embeddings, mask)[0]
        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize(
    "options",
    [
        {"mla_weight_layout": "xh2_absorbed_v2"},
        {"offload_to_disk": True},
        {"trust_remote_code": False},
    ],
)
def test_recipe_rejects_unsupported_absorption_options(tmp_path, options):
    from gptqmodel.recipes import quantize_xing4_0

    with pytest.raises(ValueError):
        quantize_xing4_0(
            model_dir=str(tmp_path / "source"),
            output_dir=str(tmp_path / "out"),
            dry_run=True,
            **options,
        )
    assert not (tmp_path / "out").exists()


def test_absorbed_fp_hf_save_load_resave(tmp_path):
    from accelerate import init_empty_weights
    from transformers import AutoConfig, AutoModelForCausalLM

    source = tmp_path / "source"
    raw = save_tiny_xing(source)
    original_code = (source / "modeling_xing4_0.py").read_bytes()
    model = absorb_xing_mla(raw).float()
    output, resaved = tmp_path / "absorbed", tmp_path / "resaved"
    model.save_pretrained(output)
    finalize_xing_mla_checkpoint(output)
    config = AutoConfig.from_pretrained(output, trust_remote_code=True)
    with init_empty_weights():
        empty = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
    assert empty.model.layers[0].self_attn.q_absorbed_proj.weight.is_meta
    loaded, info = AutoModelForCausalLM.from_pretrained(
        output, trust_remote_code=True, dtype=torch.float32, output_loading_info=True
    )
    assert not any(info.values())
    expected, actual = model.state_dict(), loaded.state_dict()
    assert expected.keys() == actual.keys()
    assert all(torch.equal(value, actual[key]) for key, value in expected.items())
    torch.testing.assert_close(
        loaded.model.rotary_emb.inv_freq,
        model.model.rotary_emb.inv_freq,
        atol=0,
        rtol=0,
    )
    tokens = torch.tensor([[4, 5, 6, 7]])
    torch.testing.assert_close(
        loaded(tokens).logits, model(tokens).logits, atol=0, rtol=0
    )
    loaded.save_pretrained(resaved)
    again = AutoModelForCausalLM.from_pretrained(
        resaved, trust_remote_code=True, dtype=torch.float32
    )
    torch.testing.assert_close(
        again(tokens).logits, model(tokens).logits, atol=0, rtol=0
    )
    assert (source / "modeling_xing4_0.py").read_bytes() == original_code
    # An old artifact cannot become absorbed by changing its config alone.
    bad_config = json.loads((source / "config.json").read_text())
    bad_config.update(
        mla_weight_layout=XH2_MLA_LAYOUT, mla_transform=model.config.mla_transform
    )
    (source / "config.json").write_text(json.dumps(bad_config))
    with pytest.raises(ValueError, match="checkpoint tensors|mixed layouts"):
        validate_xing_mla_model(
            AutoModelForCausalLM.from_pretrained(source, trust_remote_code=True)
        )
