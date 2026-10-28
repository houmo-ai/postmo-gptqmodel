# Copyright 2026 HOUMO AI
# SPDX-License-Identifier: Apache-2.0
"""Public Qwen3-VL recipe: real GPTQ/AutoRound weights and native feature reload."""

import json
import pytest
import torch
from test_qwen3_vl_calibration import model_and_input
from gptqmodel.recipes import quantize_qwen3_vl


def save_source(path):
    from transformers import Qwen2Tokenizer, Qwen3VLProcessor, Qwen2VLImageProcessor, Qwen3VLVideoProcessor
    quant, example = model_and_input()
    quant.model.save_pretrained(path)
    vocab = {f"t{i}": i for i in range(60)}
    vocab.update({"<|image_pad|>": 60, "<|video_pad|>": 61, "<|vision_start|>": 62, "<|vision_end|>": 63})
    tokenizer = Qwen2Tokenizer(vocab=vocab, merges=[], unk_token="t0", pad_token="t0", eos_token="t2",
                              additional_special_tokens=list(vocab)[-4:])
    Qwen3VLProcessor(tokenizer=tokenizer,
                    image_processor=Qwen2VLImageProcessor(patch_size=16, temporal_patch_size=2, merge_size=2),
                    video_processor=Qwen3VLVideoProcessor(patch_size=16, temporal_patch_size=2, merge_size=2)
                    ).save_pretrained(path)
    samples = []
    for index in range(2):
        row = {name: value.clone() for name, value in example.items()}
        for name in ("input_ids", "attention_mask", "mm_token_type_ids"):
            fill = 1 if name != "mm_token_type_ids" else 0
            row[name] = torch.nn.functional.pad(row[name], (0, 27), value=fill)
        row["pixel_values"] += index * .01
        samples.append(row)
    return samples


@pytest.mark.parametrize("method", ["gptq", "autoround"])
def test_real_feature_recipe_quantize_save_reload(tmp_path, method):
    from safetensors import safe_open
    samples = save_source(tmp_path / "source")
    if method == "autoround":
        text = {key: value.clone() for key, value in samples[0].items()
                if key in ("input_ids", "attention_mask", "mm_token_type_ids")}
        text["input_ids"].fill_(1)
        text["mm_token_type_ids"].zero_()
        # Native text position_ids is a tensor; visual presentations may pass
        # None. Exercise both transitions, including different sequence lengths.
        samples[1] = {key: (torch.nn.functional.pad(value, (0, 3), value=0 if key == "mm_token_type_ids" else 1)
                           if key in ("input_ids", "attention_mask", "mm_token_type_ids") else value)
                      for key, value in samples[1].items()}
        samples = [samples[0], text, samples[1]]
    # Optional processor fields must survive the actual quantization pipeline,
    # including mixed text/image batches and checkpoint reload.
    for sample in samples:
        sample.update(position_ids=None, pixel_values_videos=None, video_grid_thw=None)
    result = quantize_qwen3_vl(model_dir=tmp_path / "source", output_dir=tmp_path / "quant",
                               calibration_data=samples, feature_layer=3, method=method,
                               device="cuda:0", iters=2)
    payload = {}
    for path in (tmp_path / "quant").glob("*.safetensors"):
        with safe_open(path, framework="pt", device="cpu") as checkpoint:
            for name in checkpoint.keys():
                if name.endswith((".qweight", ".qzeros", ".g_idx", ".scales")):
                    payload[name] = checkpoint.get_tensor(name)
    quantized = {name[:-len(".qweight")] for name in payload if name.endswith(".qweight")}
    assert len(quantized) == 3 * 7
    assert all(name.startswith(tuple(f"model.language_model.layers.{i}." for i in range(3))) for name in quantized)
    assert result.provenance["reload_validated"]
    assert result.provenance["method"] == method
    assert json.loads((tmp_path / "quant/qwen3_vl_feature_quantization.json").read_text())["feature_layer"] == 3
    assert (tmp_path / "source/model.safetensors").is_file()
    from gptqmodel import GPTQModel
    from gptqmodel.nn_modules.qlinear.torch import TorchQuantLinear
    from gptqmodel.utils.backend import BACKEND

    # The checkpoint stores FP16 scales. Match that dtype explicitly instead
    # of comparing an automatic BF16 inference cast with serialized FP16.
    restored = GPTQModel.load(result.output_dir, device="cpu", backend=BACKEND.TORCH, dtype=torch.float16)
    modules = {name: module for name, module in restored.model.named_modules() if isinstance(module, TorchQuantLinear)}
    assert set(modules) == quantized
    for name, module in modules.items():
        for field in ("qweight", "qzeros", "g_idx", "scales"):
            torch.testing.assert_close(getattr(module, field), payload[f"{name}.{field}"], rtol=0, atol=0)
        assert torch.isfinite(module.scales).all() and (module.scales > 0).all()
        shifts = torch.arange(0, 32, module.bits)
        zeros = (module.qzeros.long().unsqueeze(-1) >> shifts) & module.maxq
        assert torch.all(zeros == 2 ** (module.bits - 1))
