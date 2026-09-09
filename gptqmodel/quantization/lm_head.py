# Copyright 2026 HOUMO AI
# SPDX-License-Identifier: Apache-2.0
"""Head calibration configuration and checkpoint-preserving serialization."""
from __future__ import annotations

import copy
import json
import re
import shutil
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from .config import FORMAT, METHOD, QuantizeConfig


@dataclass(frozen=True)
class HeadQuantizationConfig:
    bits: int
    algorithm: str = "inherit"
    group_size: int = 64
    iters: int = 400
    data_type: str | None = None
    row_chunk_size: int = 2048
    batch_tokens: int = 128
    seed: int = 42
    source_model_dir: str | None = None

    def __post_init__(self):
        if self.bits not in (4, 5, 8):
            raise ValueError("head.bits must be 4, 5 or 8")
        if self.algorithm not in ("inherit", "gptq", "autoround"):
            raise ValueError("head.algorithm must be inherit, gptq or autoround")
        if self.group_size != 64:
            raise ValueError("head.group_size must be 64 for the Merak recipe")
        if self.iters < 1 or self.row_chunk_size < 1 or self.batch_tokens < 1:
            raise ValueError("head iters, row_chunk_size and batch_tokens must be positive")
        if self.data_type not in (None, "int_sym_xh", "int_sym_gptq", "int"):
            raise ValueError("unsupported head.data_type")


def read_checkpoint_config(source):
    root = Path(source)
    config = json.loads((root / "config.json").read_text())
    # Match the local loader's sidecar precedence while retaining producer
    # fields that some exporters only place in the embedded configuration.
    for filename in ("quantize_config.json", "quant_config.json"):
        path = root / filename
        if path.is_file():
            embedded = config.get("quantization_config") or {}
            sidecar = json.loads(path.read_text())
            config["quantization_config"] = {**embedded, **sidecar}
            break
    return config


def resolve_head_algorithm(source, requested="inherit", *, backbone_algorithm=None):
    if requested in ("gptq", "autoround"):
        return requested
    if requested != "inherit":
        raise ValueError(f"Unsupported head algorithm: {requested}")
    if backbone_algorithm in ("gptq", "autoround"):
        return backbone_algorithm
    root = Path(source)
    q = read_checkpoint_config(root).get("quantization_config", {})
    meta = q.get("meta") or {}
    # AutoRound frequently exports quant_method=gptq. Producer evidence takes
    # precedence over that storage/loader tag.
    autoround = (
        q.get("provider") in ("auto-round", "autoround", "auto_round")
        or q.get("quant_method") in ("autoround", "auto_round")
        or q.get("autoround_version") is not None
        or q.get("auto_round_enabled") is True
        or bool((meta.get("auto_round") or {}).get("auto_round_enabled"))
        or (root / "qwen35_autoround_recipe_provenance.json").is_file()
    )
    if autoround:
        return "autoround"
    if q.get("quant_method") == "gptq":
        return "gptq"
    raise ValueError("Cannot infer the checkpoint producer; set head.algorithm explicitly")


def merge_head_config(dynamic, name, head):
    rule = {"bits": head["bits"], "group_size": head["group_size"],
            "sym": True, "desc_act": False, "act_group_aware": False}
    result = {}
    for pattern, value in (dynamic or {}).items():
        expression = pattern[2:] if pattern.startswith(("+:", "-:")) else pattern
        if re.search(expression, name) and (pattern.startswith("-:") or value is False):
            raise ValueError(f"Dynamic exclusion {pattern!r} disables the requested head")
        result[pattern] = copy.deepcopy(value)
    key = "+:^" + re.escape(name) + "$"
    result.pop(key, None)
    return {key: rule, **result}


def _checkpoint_weight_map(root):
    index = root / "model.safetensors.index.json"
    if index.is_file():
        data = json.loads(index.read_text())
        return data["weight_map"], data
    path = root / "model.safetensors"
    with safe_open(path, framework="pt", device="cpu") as tensors:
        mapping = dict.fromkeys(tensors.keys(), path.name)
    return mapping, {"metadata": {}, "weight_map": mapping}


def inspect_checkpoint_head(source, head):
    """Check the serialized head without loading or copying the backbone."""
    from ..nn_modules.qlinear.torch import TorchQuantLinear

    root = Path(source).expanduser().resolve()
    config = read_checkpoint_config(root)
    q = config.get("quantization_config") or {}
    if q.get("quant_method") not in ("gptq", "auto_round", "autoround"):
        raise ValueError("Low-bit head completion requires a GPTQModel checkpoint; quantize native HF input normally")
    qcfg = QuantizeConfig.from_quant_config(q)
    if qcfg.format not in (FORMAT.GPTQ, FORMAT.GPTQ_V2) or qcfg.rotation:
        raise ValueError("Low-bit head requires an unrotated local GPTQ/GPTQ V2 checkpoint")
    mapping, _ = _checkpoint_weight_map(root)
    text_config = config.get("text_config", config)
    if "lm_head.qweight" not in mapping:
        if any(key.startswith("lm_head.") and key != "lm_head.weight" and key != "lm_head.bias" for key in mapping):
            raise ValueError("Incomplete or unsupported packed lm_head checkpoint")
        key = "lm_head.weight"
        if key not in mapping and (config.get("tie_word_embeddings") or text_config.get("tie_word_embeddings")):
            key = next((key for key in ("model.embed_tokens.weight", "model.language_model.embed_tokens.weight")
                        if key in mapping), key)
        if key not in mapping:
            raise ValueError("Checkpoint contains neither a floating head nor its tied embedding")
        with safe_open(root / mapping[key], framework="pt", device="cpu") as tensors:
            weight = tensors.get_tensor(key)
            if not weight.is_floating_point() or tuple(weight.shape) != (text_config["vocab_size"], text_config["hidden_size"]):
                raise ValueError("Floating head shape/dtype does not match the model configuration")
        return None
    if "lm_head.weight" in mapping or not qcfg.lm_head or qcfg.dynamic_get("lm_head") is False:
        raise ValueError("Packed lm_head tensors conflict with checkpoint quantization metadata")
    values = {key: qcfg.dynamic_get("lm_head", key, getattr(qcfg, key))
              for key in ("bits", "group_size", "sym", "desc_act")}
    module = TorchQuantLinear(**values, in_features=text_config["hidden_size"],
                              out_features=text_config["vocab_size"], pack_dtype=qcfg.pack_dtype,
                              register_buffers=False)
    for name in ("qweight", "qzeros", "scales", "g_idx"):
        key = "lm_head." + name
        if key not in mapping:
            raise ValueError(f"Packed head is missing {key}")
        with safe_open(root / mapping[key], framework="pt", device="cpu") as tensors:
            setattr(module, name, tensors.get_tensor(key))
    module.qzero_format(format=1 if qcfg.format == FORMAT.GPTQ else 2)
    owner = torch.nn.Module()
    owner.lm_head = module
    return head_skip_info(SimpleNamespace(model=owner, lm_head="lm_head", quantize_config=qcfg,
                                          model_local_path=str(root)), head)


def untie_float_head(model):
    """Separate the output projection while preserving input embedding values."""
    head = model.model.get_submodule(model.lm_head)
    if not isinstance(head, torch.nn.Linear) or hasattr(head, "qweight"):
        return
    embedding = model.model.get_input_embeddings()
    config = model.model.config
    text_config = getattr(config, "text_config", config)
    shared = embedding is not None and head.weight is embedding.weight
    if embedding is not None and not head.weight.is_meta and not embedding.weight.is_meta:
        shared |= head.weight.data_ptr() == embedding.weight.data_ptr()
    if not (shared or getattr(config, "tie_word_embeddings", False)
            or getattr(text_config, "tie_word_embeddings", False)):
        return
    if head.weight.is_meta:
        raise ValueError("Tied head must be materialized before calibration")
    head.weight = torch.nn.Parameter(head.weight.detach().clone(), requires_grad=False)
    config.tie_word_embeddings = False
    text_config.tie_word_embeddings = False
    # Transformers 5 keeps a resolved tie map as well as the class declaration.
    for name in ("_tied_weights_keys", "all_tied_weights_keys"):
        mapping = getattr(model.model, name, None)
        if isinstance(mapping, dict):
            setattr(model.model, name, {key: value for key, value in mapping.items()
                                       if not key.startswith(model.lm_head + ".")})
        elif isinstance(mapping, (list, tuple)):
            setattr(model.model, name, [key for key in mapping if not key.startswith(model.lm_head + ".")])
    model._lm_head_untied = True


def restore_float_head(model, source):
    root = Path(source).expanduser().resolve()
    source_config = read_checkpoint_config(root)
    current_config = read_checkpoint_config(model.model_local_path)
    for key in ("hidden_size", "vocab_size"):
        left = source_config.get("text_config", source_config).get(key)
        right = current_config.get("text_config", current_config).get(key)
        if left != right:
            raise ValueError(f"Float head source has different {key}: {left} != {right}")
    mapping, _ = _checkpoint_weight_map(root)
    key = model.lm_head + ".weight"
    source_text_config = source_config.get("text_config", source_config)
    if key not in mapping and (source_config.get("tie_word_embeddings") or source_text_config.get("tie_word_embeddings")):
        key = next((key for key in ("model.embed_tokens.weight", "model.language_model.embed_tokens.weight")
                    if key in mapping), key)
    if key not in mapping:
        raise ValueError("Float head source must contain lm_head.weight")
    with safe_open(root / mapping[key], framework="pt", device="cpu") as tensors:
        weight = tensors.get_tensor(key)
    if not weight.is_floating_point() or weight.ndim != 2:
        raise ValueError("Head source weight must be a floating-point matrix")
    text_config = current_config.get("text_config", current_config)
    if tuple(weight.shape) != (text_config["vocab_size"], text_config["hidden_size"]):
        raise ValueError("Head source weight shape does not match the model configuration")
    old = model.model.get_submodule(model.lm_head)
    if old.bias is not None:
        raise ValueError("Restoring a biased head requires a matching bias and is not supported")
    head = torch.nn.Linear(weight.shape[1], weight.shape[0], bias=False, device="meta")
    head.weight = torch.nn.Parameter(weight.clone(), requires_grad=False)
    model.model.set_submodule(model.lm_head, head)


def head_skip_info(model, head):
    """Reuse a compatible packed head; an explicit float source requests calibration.

    Algorithm/iteration defaults apply only when calibration runs. Skipping
    preserves the existing head's producer metadata instead of relabeling it.
    """
    from ..nn_modules.qlinear import BaseQuantLinear

    if head is None:
        raise ValueError("requantize() requires an explicit head configuration")
    requested = head if isinstance(head, HeadQuantizationConfig) else HeadQuantizationConfig(**head)
    if requested.source_model_dir:
        return None
    module = model.model.get_submodule(model.lm_head)
    if not isinstance(module, BaseQuantLinear):
        return None
    actual = {
        "bits": int(getattr(module, "logical_bits", int(module.maxq).bit_length())),
        "storage_bits": int(module.bits), "group_size": int(module.group_size),
        "sym": bool(module.sym), "desc_act": bool(module.desc_act),
        "pack_dtype": str(module.pack_dtype),
    }
    expected = {"bits": requested.bits, "storage_bits": 8 if requested.bits == 5 else requested.bits,
                "group_size": requested.group_size, "sym": True, "desc_act": False,
                "pack_dtype": "torch.int32"}
    errors = [f"{key}={actual[key]!r} (requested {value!r})"
              for key, value in expected.items() if actual[key] != value]
    qcfg = model.quantize_config
    if qcfg.format not in (FORMAT.GPTQ, FORMAT.GPTQ_V2):
        errors.append(f"unsupported format={qcfg.format}")
    if qcfg.rotation or qcfg.adapter or module.adapter:
        errors.append("rotation/adapters are unsupported")
    if int(module.maxq) != 2**requested.bits - 1 or module.qzero_format() not in (1, 2):
        errors.append("incompatible integer range or qzeros format")
    tensors = {name: getattr(module, name, None) for name in ("qweight", "qzeros", "scales", "g_idx")}
    if any(not torch.is_tensor(value) or value.is_meta for value in tensors.values()):
        errors.append("packed head tensors are missing or unmaterialized")
    else:
        k, n = module.in_features, module.out_features
        factor = 32 // expected["storage_bits"]
        shapes = {"qweight": (k // factor, n), "qzeros": (k // requested.group_size, n // factor),
                  "scales": (k // requested.group_size, n), "g_idx": (k,)}
        if k % requested.group_size or n % factor or any(
                tuple(tensors[name].shape) != shape for name, shape in shapes.items()):
            errors.append("packed tensor shapes do not match the requested layout")
        if tensors["qweight"].dtype != torch.int32 or tensors["qzeros"].dtype != torch.int32:
            errors.append("packed tensors must use int32 words")
        if not torch.is_floating_point(tensors["scales"]):
            errors.append("scales must be floating point")
        if tensors["g_idx"].dtype not in (torch.int32, torch.int64):
            errors.append("g_idx must contain integer group indices")
        if not torch.equal(tensors["g_idx"], torch.arange(k, device=tensors["g_idx"].device) // requested.group_size):
            errors.append("g_idx must contain consecutive groups")
    metadata = getattr(model, "_lm_head_quantization", None)
    if metadata is None:
        source_q = read_checkpoint_config(model.model_local_path).get("quantization_config", {})
        metadata = (source_q.get("meta") or {}).get("lm_head_quantization") or {}
        data_type = metadata.get("data_type") or (source_q.get("extra_config") or {}).get(
            model.lm_head, {}).get("data_type") or source_q.get("data_type")
    else:
        data_type = metadata.get("data_type")
    if requested.data_type is not None and requested.data_type != data_type:
        errors.append(f"data_type={data_type!r} (requested {requested.data_type!r})")
    if errors:
        raise ValueError("Head is already packed and cannot be skipped: " + "; ".join(errors)
                         + ". Provide head.source_model_dir with the original floating head to recalibrate")
    return {"action": "skipped", "reason": "existing head matches the requested quantization layout",
            "requested": asdict(requested), "actual": {**actual, "data_type": data_type,
                                                        "algorithm": metadata.get("algorithm")},
            "source_checkpoint": str(Path(model.model_local_path).resolve())}


def save_skipped_head_checkpoint(source, save_dir, operation):
    """Copy a reused checkpoint verbatim, retaining its original quantization metadata."""
    source = Path(source).resolve()
    destination = Path(save_dir).expanduser().resolve()
    if destination == source or source in destination.parents:
        raise ValueError("Head output must be a new directory outside the source checkpoint")
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"Head output directory is not empty: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=destination.name + ".tmp-", dir=destination.parent) as tmp:
        stage = Path(tmp)
        shutil.copytree(source, stage, dirs_exist_ok=True)
        (stage / "lm_head_operation.json").write_text(json.dumps(operation, indent=2) + "\n")
        if destination.exists():
            destination.rmdir()
        stage.rename(destination)


def validate_head_quantization(model):
    qcfg = model.quantize_config
    if not qcfg.lm_head:
        raise ValueError("Head calibration requires quantize_config.lm_head=True")
    if qcfg.quant_method not in (METHOD.GPTQ, METHOD.AUTO_ROUND):
        raise ValueError("Head calibration supports GPTQ and AutoRound")
    if qcfg.format not in (FORMAT.GPTQ, FORMAT.GPTQ_V2):
        raise ValueError("Head calibration requires the local GPTQ/GPTQ V2 format")
    if qcfg.rotation or qcfg.mix_precision_search or qcfg.gptaq or qcfg.adapter:
        raise ValueError("Head-only calibration cannot apply rotation, mix search, GPTAQ or adapters")
    head = model.model.get_submodule(model.lm_head)
    if not isinstance(head, torch.nn.Linear) or hasattr(head, "qweight"):
        raise ValueError("Head is already packed; provide head.source_model_dir with the original floating head")
    if not head.weight.is_floating_point() or head.weight.is_meta:
        raise ValueError("Head calibration needs a materialized floating-point Linear")
    embedding = model.model.get_input_embeddings()
    model_config = model.model.config
    text_config = getattr(model_config, "text_config", model_config)
    if (getattr(model_config, "tie_word_embeddings", False)
            or getattr(text_config, "tie_word_embeddings", False)
            or (embedding is not None and head.weight.data_ptr() == embedding.weight.data_ptr())):
        raise ValueError("Head calibration requires untied embeddings")


def prepare_head_quantization(model, head):
    if head is None:
        raise ValueError("requantize() requires an explicit head configuration")
    config = head if isinstance(head, HeadQuantizationConfig) else HeadQuantizationConfig(**head)
    source_qcfg = read_checkpoint_config(model.model_local_path).get("quantization_config", {})
    if getattr(model.quantize_config, "rotation", None):
        raise ValueError("Head-only calibration of rotated checkpoints is not supported")
    if model.quantize_config.pack_dtype != torch.int32:
        raise ValueError("Head calibration requires pack_dtype=torch.int32 with the local block packer")
    algorithm = resolve_head_algorithm(model.model_local_path, config.algorithm)
    data_type = config.data_type or source_qcfg.get("data_type") or "int_sym_xh"
    if data_type not in ("int_sym_xh", "int_sym_gptq", "int"):
        raise ValueError(f"Unsupported inherited head.data_type: {data_type!r}")
    qcfg = QuantizeConfig(
        bits=config.bits, group_size=config.group_size, sym=True, desc_act=False,
        act_group_aware=False, lm_head=True, format=model.quantize_config.format,
        pack_dtype=model.quantize_config.pack_dtype,
        quant_method=METHOD.AUTO_ROUND if algorithm == "autoround" else METHOD.GPTQ,
        data_type=data_type, iters=config.iters, device=model.quantize_config.device,
        offload_to_disk=False, wait_for_submodule_finalizers=True,
    )
    provenance = {**asdict(config), "algorithm": algorithm, "data_type": data_type,
                  "storage_bits": 8 if config.bits == 5 else config.bits,
                  "source_checkpoint": str(Path(model.model_local_path).resolve())}
    qcfg.dynamic = merge_head_config(None, model.lm_head, provenance)
    qcfg.device = torch.device(model.quantize_config.device or "cpu")
    qcfg.head_tuning = provenance
    # Fail on a conflicting source exclusion before restoring/mutating its head.
    merge_head_config(model.quantize_config.dynamic, model.lm_head, provenance)
    if config.source_model_dir:
        restore_float_head(model, config.source_model_dir)
    untie_float_head(model)
    return qcfg, provenance


def save_head_checkpoint(model, save_dir, *, safetensors_metadata=None):
    """Copy source tensors verbatim and publish a new head as a separate shard."""
    source = Path(model._lm_head_replacement_source).resolve()
    destination = Path(save_dir).expanduser().resolve()
    if destination == source or source in destination.parents:
        raise ValueError("Head output must be a new directory outside the source checkpoint")
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"Head output directory is not empty: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    mapping, index = _checkpoint_weight_map(source)
    prefix = model.lm_head + "."
    old_keys = {name for name in mapping if name.startswith(prefix)}
    if not old_keys and not getattr(model, "_lm_head_untied", False):
        raise ValueError("Source checkpoint contains no head tensors")
    head = model.model.get_submodule(model.lm_head)
    payload = {prefix + key: value.detach().cpu().contiguous()
               for key, value in head.state_dict().items()}
    required = {prefix + key for key in ("qweight", "qzeros", "scales", "g_idx")}
    if not required <= payload.keys():
        raise ValueError("Calibrated head is missing packed tensors")
    with tempfile.TemporaryDirectory(prefix=destination.name + ".tmp-", dir=destination.parent) as tmp:
        stage = Path(tmp)
        affected = {mapping[key] for key in old_keys}
        total_size = 0
        for path in source.iterdir():
            if path.is_file() and path.name not in affected:
                shutil.copy2(path, stage / path.name)
            elif path.is_dir():
                shutil.copytree(path, stage / path.name)
        for filename in set(mapping.values()):
            with safe_open(source / filename, framework="pt", device="cpu") as tensors:
                retained = {key: tensors.get_tensor(key) for key in tensors.keys() if key not in old_keys}
                total_size += sum(t.numel() * t.element_size() for t in retained.values())
                if filename in affected and retained:
                    output_filename = "model-backbone.safetensors" if filename == "model.safetensors" else filename
                    save_file(retained, stage / output_filename, metadata=tensors.metadata())
        if (stage / "model.safetensors").exists():
            (stage / "model.safetensors").rename(stage / "model-backbone.safetensors")
        filename = "model-lm-head.safetensors"
        suffix = 1
        while (stage / filename).exists():
            filename = f"model-lm-head-{suffix}.safetensors"
            suffix += 1
        save_file(payload, stage / filename, metadata=safetensors_metadata or {"format": "pt"})
        new_mapping = {key: value for key, value in mapping.items() if key not in old_keys}
        # HF loaders prefer an unsharded model.safetensors over the index. A
        # formerly single-shard checkpoint must become unambiguously sharded.
        new_mapping = {key: "model-backbone.safetensors" if value == "model.safetensors" else value
                       for key, value in new_mapping.items()}
        new_mapping.update(dict.fromkeys(payload, filename))
        total_size += sum(t.numel() * t.element_size() for t in payload.values())
        index["weight_map"] = new_mapping
        index.setdefault("metadata", {})["total_size"] = total_size
        (stage / "model.safetensors.index.json").write_text(json.dumps(index, indent=2) + "\n")
        config = read_checkpoint_config(source)
        if getattr(model, "_lm_head_untied", False):
            config["tie_word_embeddings"] = False
            if "text_config" in config:
                config["text_config"]["tie_word_embeddings"] = False
        q = config["quantization_config"]
        q["lm_head"] = True
        q["dynamic"] = merge_head_config(q.get("dynamic"), model.lm_head, model._lm_head_quantization)
        q.setdefault("meta", {})["lm_head_quantization"] = model._lm_head_quantization
        if "extra_config" in q:
            q["extra_config"][model.lm_head] = {
                "bits": model._lm_head_quantization["bits"], "group_size": 64,
                "data_type": model._lm_head_quantization["data_type"], "sym": True,
            }
        (stage / "config.json").write_text(json.dumps(config, indent=2) + "\n")
        for name in ("quantize_config.json", "quant_config.json", "quantization_config.json"):
            if name == "quantize_config.json" or (stage / name).exists():
                (stage / name).write_text(json.dumps(q, indent=2) + "\n")
        (stage / "lm_head_quantization.json").write_text(json.dumps(
            {**model._lm_head_quantization, "quant_log": model.quant_log}, indent=2, default=str) + "\n")
        (stage / "lm_head_operation.json").write_text(json.dumps(
            model._lm_head_last_operation, indent=2, default=str) + "\n")
        if destination.exists():
            destination.rmdir()
        stage.rename(destination)
