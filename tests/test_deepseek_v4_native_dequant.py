import json

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file

from gptqmodel.utils.model_dequant import (
    dequantize_deepseek_v4_fp4_expert,
    dequantize_model,
)


def test_dequantize_deepseek_v4_fp4_expert_low_nibble_first():
    codes = torch.arange(16, dtype=torch.uint8).repeat(2)
    packed = (codes[0::2] | (codes[1::2] << 4)).view(1, 16).to(torch.int8)
    scale = torch.full((1, 1), 2.0, dtype=torch.float32)

    output = dequantize_deepseek_v4_fp4_expert(packed, scale)

    table = torch.tensor(
        [
            0.0,
            0.5,
            1.0,
            1.5,
            2.0,
            3.0,
            4.0,
            6.0,
            0.0,
            -0.5,
            -1.0,
            -1.5,
            -2.0,
            -3.0,
            -4.0,
            -6.0,
        ],
        dtype=torch.float32,
    )
    expected = (table.repeat(2).view(1, 32) * 2.0).to(torch.bfloat16)
    torch.testing.assert_close(output, expected)


@pytest.mark.skipif(
    not hasattr(torch, "float8_e8m0fnu"),
    reason="float8_e8m0fnu dtype is required",
)
def test_dequantize_model_handles_deepseek_v4_direct_fp8_and_fp4(tmp_path):
    model_dir = tmp_path / "native"
    output_dir = tmp_path / "bf16"
    model_dir.mkdir()
    config = {
        "model_type": "deepseek_v4",
        "quantization_config": {
            "quant_method": "fp8",
            "fmt": "e4m3",
            "scale_fmt": "ue8m0",
            "weight_block_size": [128, 128],
        },
    }
    (model_dir / "config.json").write_text(json.dumps(config), encoding="utf-8")

    fp8_key = "layers.0.attn.wkv.weight"
    fp8_scale_key = "layers.0.attn.wkv.scale"
    expert_key = "layers.0.ffn.experts.0.w1.weight"
    expert_scale_key = "layers.0.ffn.experts.0.w1.scale"
    fp8_weight = torch.ones((128, 128), dtype=torch.float32).to(torch.float8_e4m3fn)
    fp8_scale = torch.full((1, 1), 2.0, dtype=torch.float32).to(torch.float8_e8m0fnu)
    expert_weight = torch.zeros((1, 16), dtype=torch.int8)
    expert_scale = torch.ones((1, 1), dtype=torch.float32).to(torch.float8_e8m0fnu)
    tensors = {
        fp8_key: fp8_weight,
        fp8_scale_key: fp8_scale,
        expert_key: expert_weight,
        expert_scale_key: expert_scale,
        "layers.0.hc_attn_fn": torch.tensor(
            [[1.0001, 1.0002]],
            dtype=torch.float32,
        ),
    }
    shard = "model.safetensors"
    save_file(tensors, model_dir / shard)
    (model_dir / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {}, "weight_map": {key: shard for key in tensors}}),
        encoding="utf-8",
    )

    dequantize_model(model_dir, output_dir, target_dtype=torch.bfloat16)

    with safe_open(output_dir / shard, framework="pt", device="cpu") as reader:
        assert set(reader.keys()) == {
            fp8_key,
            expert_key,
            "layers.0.hc_attn_fn",
        }
        torch.testing.assert_close(
            reader.get_tensor(fp8_key),
            torch.full((128, 128), 2.0, dtype=torch.bfloat16),
        )
        assert reader.get_tensor(expert_key).shape == (1, 32)
        sensitive = reader.get_tensor("layers.0.hc_attn_fn")
        assert sensitive.dtype == torch.float32
        torch.testing.assert_close(
            sensitive,
            tensors["layers.0.hc_attn_fn"],
            rtol=0,
            atol=0,
        )
