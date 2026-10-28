# Copyright 2026 HOUMO AI
# SPDX-License-Identifier: Apache-2.0
"""Qwen3-VL feature-conditioning weights for downstream deployment compilers."""

from __future__ import annotations

import json
import tempfile
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Qwen3VLQuantRecipeResult:
    output_dir: str
    method: str
    artifact_format: str
    provenance: dict


def deepstack_block_groups(feature_layer, deepstack_layers):
    """Cache native boundaries after each visual addition, then replay the tail."""
    names = [f"model.language_model.layers.{i}" for i in range(feature_layer)]
    split = min(feature_layer, deepstack_layers)
    groups = [[name] for name in names[:split]]
    if split < feature_layer:
        groups.append(names[split:])
    return groups


def quantize_qwen3_vl(
    *, model_dir, output_dir, calibration_data, method="gptq", feature_layer=50,
    bits=4, group_size=64, device="cuda:0", seed=42, iters=200,
    tokenizer_dir=None, processor_dir=None,
) -> Qwen3VLQuantRecipeResult:
    """Quantize the used language prefix; retain vision and unused weights.

    ``calibration_data`` contains native processor tensors for one complete
    presentation per dictionary. No chat template, truncation, or text-only
    replacement is applied. A8 and vision W8 are downstream compiler choices;
    this recipe produces canonical packed language weights, not deployment graphs.
    """
    if method not in ("gptq", "autoround"):
        raise ValueError("method must be gptq or autoround")
    if bits not in (4, 8) or group_size != 64:
        raise ValueError("Qwen3-VL deployment weights require W4/W8 with group_size=64")
    if type(feature_layer) is not int or feature_layer <= 0:
        raise ValueError("feature_layer must be positive")
    if method == "autoround" and (type(iters) is not int or iters <= 0):
        raise ValueError("AutoRound requires positive tuning iterations")
    source, target = Path(model_dir).expanduser().resolve(), Path(output_dir).expanduser().resolve()
    if target.exists() or target == source:
        raise ValueError("output_dir must not exist and must differ from model_dir")
    if not target.parent.is_dir():
        raise ValueError("output_dir parent must exist")
    cfg = json.loads((source / "config.json").read_text())
    if cfg.get("model_type") != "qwen3_vl":
        raise ValueError("recipe requires dense Qwen3-VL")
    if feature_layer >= cfg["text_config"]["num_hidden_layers"]:
        raise ValueError("feature_layer must precede final norm in the native hidden-state stack")

    if not isinstance(calibration_data, list) or not calibration_data or not all(
        isinstance(row, dict) and "input_ids" in row for row in calibration_data
    ):
        raise ValueError("calibration_data requires nonempty native processor tensor dictionaries")
    import torch
    from ..models.definitions.qwen3_vl import Qwen3_VLQModel

    validator = object.__new__(Qwen3_VLQModel)
    torch.nn.Module.__init__(validator)
    samples = validator.prepare_dataset(calibration_data, batch_size=1)
    if not samples:
        raise ValueError("calibration_data must contain complete native presentations")
    for row in samples:
        if not 0 < row["input_ids"].shape[1] <= cfg["text_config"]["max_position_embeddings"]:
            raise ValueError("calibration presentation exceeds the native context")
    torch.manual_seed(seed)
    # Source files remain read-only. A local view combines split HF resources.
    with tempfile.TemporaryDirectory(prefix="qwen3-vl-source-", dir=target.parent) as view_name:
        view = Path(view_name)
        for path in source.iterdir():
            if path.is_file():
                (view / path.name).symlink_to(path)
        for directory in (tokenizer_dir, processor_dir):
            if directory is None:
                continue
            for path in Path(directory).expanduser().resolve().iterdir():
                if path.is_file() and path.suffix in (".json", ".txt", ".jinja") and path.name != "config.json":
                    destination = view / path.name
                    if destination.is_symlink():
                        destination.unlink()
                    if not destination.exists():
                        destination.symlink_to(path)
        with tempfile.TemporaryDirectory(prefix="qwen3-vl-quant-", dir=target.parent) as staging:
            output = Path(staging) / "checkpoint"
            if method == "gptq":
                _gptq(view, output, samples, feature_layer, bits, group_size, device)
            else:
                _autoround(view, output, samples, cfg, feature_layer, bits, group_size, device, seed, iters)
            provenance = dict(method=method, feature_layer=feature_layer, bits=bits, group_size=group_size,
                              seed=seed, data_type="int_sym_xh" if method == "autoround" else "int_sym_gptq",
                              calibration_samples=len(samples),
                              calibration_lengths=[int(row["input_ids"].shape[1]) for row in samples],
                              input_format="native_pretokenized_multimodal", checkpoint_format="gptq_v2",
                              vision_weights="unquantized",
                              deployment_activation_bits=None, deepstack_preserved=True)
            if method == "autoround":
                provenance.update(iters=iters, block_groups=deepstack_block_groups(
                    feature_layer, len(cfg["vision_config"]["deepstack_visual_indexes"])))
            _validate_reload(output, samples[0], feature_layer, device)
            provenance["reload_validated"] = True
            (output / "qwen3_vl_feature_quantization.json").write_text(json.dumps(provenance, indent=2) + "\n")
            output.rename(target)
    return Qwen3VLQuantRecipeResult(str(target), method, "gptqmodel_hf", provenance)


def _gptq(source, output, samples, feature_layer, bits, group_size, device):
    from .. import GPTQModel, QuantizeConfig
    from ..quantization import FORMAT

    config = QuantizeConfig(bits=bits, group_size=group_size, sym=True, desc_act=False,
                            data_type="int_sym_gptq",
                            act_group_aware=False, format=FORMAT.GPTQ_V2, device=device,
                            offload_to_disk=False,
                            dynamic={f"-:model.language_model.layers.{i}\\..*": {} for i in range(
                                feature_layer, json.loads((source / "config.json").read_text())["text_config"]["num_hidden_layers"]
                            )})
    model = GPTQModel.load(str(source), config)
    model.quantize(samples, batch_size=1)
    model.save(str(output))


def _autoround(source, output, samples, cfg, feature_layer, bits, group_size, device, seed, iters):
    import torch
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
    from auto_round import AutoRound

    model = Qwen3VLForConditionalGeneration.from_pretrained(str(source), dtype=torch.float32,
                                                          local_files_only=True).eval()
    processor = AutoProcessor.from_pretrained(source, local_files_only=True)
    names = tuple(f"model.language_model.layers.{i}." for i in range(feature_layer))
    layer_config = {
        name: ({"bits": bits, "group_size": group_size, "sym": True, "data_type": "int_sym_xh"}
               if name.startswith(names) else {"bits": 16})
        for name, module in model.named_modules() if isinstance(module, torch.nn.Linear)
    }
    tuner = AutoRound(model=model, tokenizer=processor.tokenizer, processor=processor,
                      bits=bits, group_size=group_size, layer_config=layer_config,
                      dataset=samples, nsamples=len(samples), batch_size=1,
                      seqlen=min(row["input_ids"].shape[1] for row in samples),
                      iters=iters, seed=seed, device_map=device.removeprefix("cuda:"), low_gpu_mem_usage=True)
    tuner.quant_block_list = deepstack_block_groups(feature_layer, len(cfg["vision_config"]["deepstack_visual_indexes"]))
    tuner.quantize()
    tuner.save_quantized(str(output), format="auto_round:gptqmodel")
    processor.save_pretrained(output)


def _validate_reload(output, sample, feature_layer, device):
    import torch
    from .. import GPTQModel
    from ..quantization import FORMAT
    from ..utils.backend import BACKEND

    model = GPTQModel.load(str(output), device=device, backend=BACKEND.TORCH)
    if model.quantize_config.format != FORMAT.GPTQ_V2:
        raise ValueError("Qwen3-VL feature recipe must publish GPTQ v2 checkpoints")
    with torch.no_grad():
        result = model.model.model(**{name: value.to(device) for name, value in sample.items()},
                                   use_cache=False, output_hidden_states=True)
    features = result.hidden_states[feature_layer]
    if features.shape[:2] != sample["input_ids"].shape or not bool(torch.isfinite(features).all()):
        raise ValueError(f"saved Qwen3-VL feature checkpoint failed reload validation: "
                         f"shape={tuple(features.shape)}, finite={bool(torch.isfinite(features).all())}, "
                         f"bad_parameters={[name for name, value in model.model.named_parameters() if not bool(torch.isfinite(value).all())][:8]}")
