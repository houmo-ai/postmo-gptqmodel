# Copyright 2026 HOUMO AI
# SPDX-License-Identifier: Apache-2.0
"""Native decoder boundary parity for multimodal GPTQ calibration."""

import pytest
import torch
from transformers import (Qwen2Tokenizer, Qwen2VLImageProcessor, Qwen3VLConfig,
                          Qwen3VLForConditionalGeneration, Qwen3VLProcessor, Qwen3VLVideoProcessor)
from gptqmodel.models.definitions.base_qwen3_vl import BaseQwen3VLGPTQ
from gptqmodel.models.definitions.qwen3_vl import Qwen3_VLQModel
from gptqmodel.looper.input_cache import InputCache


def model_config():
    return Qwen3VLConfig(
        text_config=dict(vocab_size=64, hidden_size=128, intermediate_size=256, num_hidden_layers=4,
                         num_attention_heads=2, num_key_value_heads=1, head_dim=64,
                         rope_parameters={"rope_type": "default", "rope_theta": 1000000.,
                                          "mrope_section": [8, 12, 12]}),
        vision_config=dict(depth=3, hidden_size=64, intermediate_size=128, num_heads=4,
                           out_hidden_size=128, patch_size=16, temporal_patch_size=2,
                           spatial_merge_size=2, deepstack_visual_indexes=[0, 1, 2]),
        image_token_id=60, video_token_id=61, vision_start_token_id=62, vision_end_token_id=63,
    )


def packed_input():
    return dict(input_ids=torch.tensor([[1, 62, 60, 63, 2]]),
                   attention_mask=torch.ones(1, 5, dtype=torch.long),
                   mm_token_type_ids=torch.tensor([[0, 0, 1, 0, 0]]),
                   pixel_values=torch.randn(4, 1536), image_grid_thw=torch.tensor([[1, 2, 2]]))


def input_adapter():
    """The real dataset adapter needs no weights for tensor validation."""
    quant = object.__new__(Qwen3_VLQModel)
    torch.nn.Module.__init__(quant)
    return quant


def model_and_input():
    torch.manual_seed(17)
    quant = input_adapter()
    quant.model = Qwen3VLForConditionalGeneration(model_config()).eval()
    return quant, packed_input()


@torch.no_grad()
def test_deepstack_replay_matches_native_inputs_and_is_idempotent():
    quant, example = model_and_input()
    extras = quant.build_layer_extra_inputs_cache(example, torch.device("cpu"))
    before, after, handles = [], [], []
    for layer in quant.model.model.language_model.layers:
        handles.append(layer.register_forward_pre_hook(
            lambda mod, args, kw: before.append((args[0] if args else kw["hidden_states"]).clone()),
            with_kwargs=True))
        handles.append(layer.register_forward_hook(lambda mod, args, out: after.append(out.clone())))
    try:
        quant.model(**example, use_cache=False)
    finally:
        for handle in handles:
            handle.remove()
    cache = InputCache([], [{}], [], [], layer_extra_inputs=[[entry] for entry in extras])
    for layer_index in (1, 2, 3):
        raw = after[layer_index - 1]
        saved = raw.clone()
        prepared = quant.prepare_layer_inputs_for_quantize([[raw]], cache, layer_index)
        torch.testing.assert_close(prepared[0][0], before[layer_index], rtol=0, atol=0)
        assert quant.prepare_layer_inputs_for_quantize(prepared, cache, layer_index) is prepared
        torch.testing.assert_close(raw, saved, rtol=0, atol=0)
        assert not torch.equal(raw, before[layer_index])
    assert not quant.model.model.language_model._forward_pre_hooks


@pytest.mark.parametrize("tensor_rank,with_labels", [(None, False), (1, False), (2, True)])
def test_text_dictionaries_keep_existing_normalization_and_batching(tensor_rank, with_labels):
    quant = input_adapter()
    samples = [{"input_ids": list(range(1, length + 1))} for length in (3, 5, 4)]
    if with_labels:
        for sample in samples:
            sample["labels"] = sample["input_ids"][:]
    if tensor_rank is not None:
        samples = [{name: torch.tensor(value) if tensor_rank == 1 else torch.tensor([value])
                    for name, value in sample.items()} for sample in samples]
    options = dict(batch_size=2, calibration_data_min_length=3, calibration_dataset_sort="desc")
    expected = BaseQwen3VLGPTQ.prepare_dataset(quant, samples, **options)
    actual = quant.prepare_dataset(samples, **options)
    assert len(actual) == len(expected) == 1
    assert actual[0]["input_ids"].shape == (2, 5)
    for name, value in expected[0].items():
        torch.testing.assert_close(actual[0][name], value, rtol=0, atol=0)


def test_packed_text_validation_preserves_short_recipe_inputs_without_loading_model():
    quant = input_adapter()
    example = {"input_ids": torch.tensor([[1, 2]]), "attention_mask": torch.ones(1, 2, dtype=torch.long)}
    prepared = quant.prepare_dataset([example])
    assert len(prepared) == 1
    assert all(prepared[0][name] is value for name, value in example.items())


def processor_samples():
    vocab = {f"t{i}": i for i in range(60) if i not in (1, 2)}
    special_tokens = {"<|image_pad|>": 60, "<|video_pad|>": 61,
                      "<|vision_start|>": 62, "<|vision_end|>": 63}
    vocab.update({"a": 1, "b": 2, **special_tokens})
    tokenizer = Qwen2Tokenizer(vocab=vocab, merges=[], unk_token="t0", pad_token="t0", eos_token="b",
                              additional_special_tokens=list(special_tokens))
    processor = Qwen3VLProcessor(
        tokenizer=tokenizer,
        image_processor=Qwen2VLImageProcessor(patch_size=16, temporal_patch_size=2, merge_size=2,
                                             min_pixels=1024, max_pixels=1024),
        video_processor=Qwen3VLVideoProcessor(patch_size=16, temporal_patch_size=2, merge_size=2,
                                             size={"shortest_edge": 1024, "longest_edge": 1024}),
    )
    pixels = torch.arange(3 * 32 * 32).reshape(3, 32, 32).remainder(256).to(torch.uint8)
    text = processor(text="ab", return_tensors="pt")
    image = processor(text="a<|vision_start|><|image_pad|><|vision_end|>b",
                      images=[pixels], return_tensors="pt")
    video = processor(text="a<|video_pad|>b", videos=[torch.stack([pixels, pixels])],
                      do_sample_frames=False,
                      video_metadata=[{"fps": 24., "total_num_frames": 2, "frames_indices": [0, 1]}],
                      return_tensors="pt")
    assert set(video) == {"input_ids", "attention_mask", "mm_token_type_ids",
                          "pixel_values_videos", "video_grid_thw"}
    return [text, image, video]


@torch.no_grad()
def test_native_processor_text_image_video_normalize_optional_none_and_capture():
    quant, _ = model_and_input()
    samples = processor_samples()
    optional_fields = {"pixel_values", "pixel_values_videos", "image_grid_thw", "video_grid_thw", "position_ids"}
    for sample in samples:
        for name in optional_fields - set(sample):
            sample[name] = None
    original_fields = [dict(sample) for sample in samples]
    prepared = quant.prepare_dataset(samples)
    for index, (sample, example) in enumerate(zip(samples, prepared)):
        assert set(example) == {name for name, value in sample.items() if value is not None}
        assert all(example[name] is value for name, value in sample.items() if value is not None)
        assert set(sample) == set(original_fields[index])
        assert all(sample[name] is value for name, value in original_fields[index].items())
        extras = quant.build_layer_extra_inputs_cache(example, "cpu")
        if index == 0:
            assert extras == [[], [], [], []]
        else:
            assert not extras[0] and all(len(extra) == 2 for extra in extras[1:])
            assert all(extra[0].sum() == 1 for extra in extras[1:])
        native_logits = quant.model(**sample, use_cache=False).logits
        prepared_logits = quant.model(**example, use_cache=False).logits
        assert torch.isfinite(prepared_logits).all()
        torch.testing.assert_close(prepared_logits, native_logits, rtol=0, atol=0)
    assert not quant._language_model()._forward_pre_hooks


@pytest.mark.parametrize("field,value", [("unknown_field", None), ("video_metadata", [{}]),
                                         ("position_ids", [1, 2]), ("input_ids", None)])
def test_native_presentation_rejects_unknown_or_nontensor_fields(field, value):
    quant, example = input_adapter(), packed_input()
    with pytest.raises(ValueError, match="unsupported|requires tensors"):
        quant.prepare_dataset([{**example, field: value}])


@pytest.mark.parametrize("failure", ["vision", "capture"])
def test_deepstack_capture_removes_hook_when_native_forward_fails(failure):
    quant, example = model_and_input()
    language = quant._language_model()
    layers = language.layers
    broken = dict(example)
    if failure == "vision":
        broken["pixel_values"] = broken["pixel_values"][:, :-1]
        expected_error, message = RuntimeError, "shape"
    else:
        language.layers = torch.nn.ModuleList(list(layers)[:3])
        expected_error, message = ValueError, "after the last DeepStack addition"
    try:
        with pytest.raises(expected_error, match=message):
            quant.build_layer_extra_inputs_cache(broken, "cpu")
        assert not language._forward_pre_hooks
    finally:
        language.layers = layers
    extras = quant.build_layer_extra_inputs_cache(example, "cpu")
    assert all(len(extra) == 2 for extra in extras[1:])
    assert not language._forward_pre_hooks


def test_deepstack_groups_preserve_all_injection_boundaries():
    from gptqmodel.recipes.qwen3_vl import deepstack_block_groups

    groups = deepstack_block_groups(50, 3)
    assert [len(group) for group in groups] == [1, 1, 1, 47]
    assert [name for group in groups for name in group] == [f"model.language_model.layers.{i}" for i in range(50)]


def test_invalid_recipe_request_does_not_publish(tmp_path):
    from gptqmodel.recipes import quantize_qwen3_vl

    # All failures precede model loading or calibration; a real HF config and
    # native tensor presentation are sufficient inputs for these contracts.
    source = tmp_path / "source"
    model_config().save_pretrained(source)
    common = dict(model_dir=source, output_dir=tmp_path / "out", calibration_data=[packed_input()], feature_layer=3)
    for override in ({"method": "unknown"}, {"feature_layer": 4}, {"bits": 3},
                     {"calibration_data": []}, {"method": "autoround", "iters": 0}):
        with pytest.raises(ValueError):
            quantize_qwen3_vl(**(common | override))
        assert not (tmp_path / "out").exists()
