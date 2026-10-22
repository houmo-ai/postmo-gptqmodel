# Copyright 2026 HOUMO AI
# SPDX-License-Identifier: Apache-2.0
"""Preserve Qwen3-VL's inter-layer visual additions during GPTQ replay."""

from collections.abc import Mapping

import torch


class _PreparedLayerInputs(list):
    def __init__(self, values, layer_index):
        super().__init__(values)
        self.layer_index = layer_index


class Qwen3VLCalibrationMixin:
    def prepare_dataset(self, calibration_dataset, batch_size=1, **kwargs):
        if calibration_dataset and all(
            isinstance(row, Mapping) and "input_ids" in row
            and set(row) <= {"input_ids", "attention_mask", "labels"}
            for row in calibration_dataset
        ) and any(
            "labels" in row or not isinstance(row["input_ids"], torch.Tensor)
            or row["input_ids"].ndim != 2
            for row in calibration_dataset
        ):
            # Keep the existing text normalization, filtering and batching path.
            return super().prepare_dataset(
                [dict(row) for row in calibration_dataset], batch_size=batch_size, **kwargs
            )
        # H3 supplies a pre-tokenized presentation, not a chat conversation.
        # Keep processor-produced multimodal tensors and mRoPE input tags intact.
        if calibration_dataset and all(isinstance(row, Mapping) and "input_ids" in row for row in calibration_dataset):
            permitted = {"input_ids", "attention_mask", "pixel_values", "pixel_values_videos",
                         "image_grid_thw", "video_grid_thw", "mm_token_type_ids", "position_ids"}
            result = []
            for row in calibration_dataset:
                if set(row) - permitted:
                    raise ValueError(f"unsupported Qwen3-VL calibration fields: {sorted(set(row) - permitted)}")
                if not isinstance(row["input_ids"], torch.Tensor) or any(
                    value is not None and not isinstance(value, torch.Tensor) for value in row.values()
                ):
                    raise ValueError("pre-tokenized Qwen3-VL calibration requires tensors or optional None fields")
                if row["input_ids"].ndim != 2 or row["input_ids"].shape[0] != 1:
                    raise ValueError("pre-tokenized Qwen3-VL calibration requires one packed request per sample")
                # Optional None kwargs are equivalent to the model defaults.
                # Normalize them before tensor-only cache/device handling.
                result.append({name: value for name, value in row.items() if value is not None})
            return result
        return super().prepare_dataset(calibration_dataset, batch_size=batch_size, **kwargs)

    @torch.no_grad()
    def build_layer_extra_inputs_cache(self, example, data_device):
        language = self._language_model()
        result = [[] for _ in language.layers]
        if not any(example.get(name) is not None for name in ("pixel_values", "pixel_values_videos")):
            return result

        class PrefixCaptured(Exception):
            pass

        def capture(module, args, kwargs):
            additions = kwargs.get("deepstack_visual_embeds")
            mask = kwargs.get("visual_pos_masks")
            if additions is None or mask is None:
                raise ValueError("visual calibration did not produce Qwen3-VL DeepStack inputs")
            if len(additions) >= len(result):
                raise ValueError("calibration needs a decoder layer after the last DeepStack addition")
            for index, addition in enumerate(additions):
                result[index + 1] = [mask.detach().to(data_device), addition.detach().to(data_device)]
            raise PrefixCaptured

        # Stop before the first decoder layer. This captures native visual placement
        # and avoids reconstructing image/video interleaving in the quantizer.
        handle = language.register_forward_pre_hook(capture, with_kwargs=True)
        try:
            try:
                self.model(**example, use_cache=False)
            except PrefixCaptured:
                pass
            else:
                raise RuntimeError("Qwen3-VL calibration never entered the language tower")
        finally:
            handle.remove()
        return result

    def prepare_layer_inputs_for_quantize(self, layer_inputs, input_cache, layer_index):
        if isinstance(layer_inputs, _PreparedLayerInputs) and layer_inputs.layer_index == layer_index:
            return layer_inputs
        extras = getattr(input_cache, "layer_extra_inputs", None)
        if not extras or layer_index >= len(extras):
            return layer_inputs
        batches = extras[layer_index]
        if len(batches) != len(layer_inputs):
            raise ValueError("DeepStack cache batch count differs from the decoder inputs")
        prepared = []
        for inputs, extra in zip(layer_inputs, batches):
            if not extra:
                prepared.append(inputs)
                continue
            mask, addition = extra
            hidden = inputs[0].clone()
            mask = mask.to(hidden.device)
            hidden[mask] += addition.to(device=hidden.device, dtype=hidden.dtype)
            prepared.append([hidden, *inputs[1:]])
        return _PreparedLayerInputs(prepared, layer_index)
