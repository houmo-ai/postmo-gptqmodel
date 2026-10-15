# Copyright 2026 HOUMO AI
# SPDX-License-Identifier: Apache-2.0
"""Preserve native DSpark weights independently of the quantized target model.

Transformers' target-only V4 implementation does not instantiate the DSpark
network. Keeping its checkpoint in a separate directory also prevents native
FP4/FP8 draft tensors from being mislabeled as packed GPTQ target weights.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import struct
import tempfile
from pathlib import Path

DSPARK_DIRECTORY = "dspark"
DSPARK_MANIFEST = "dspark_meta_info.json"


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _header(path: Path) -> dict:
    with path.open("rb") as stream:
        prefix = stream.read(8)
        if len(prefix) != 8:
            raise ValueError(f"Invalid safetensors header: {path}")
        length = struct.unpack("<Q", prefix)[0]
        if not 2 <= length <= 100_000_000:
            raise ValueError(f"Invalid safetensors header length: {path}")
        return json.loads(stream.read(length))


def _local_file(root: Path, filename: str) -> Path:
    path = root / filename
    if Path(filename).is_absolute() or ".." in Path(filename).parts:
        raise ValueError(f"Unsafe DSpark shard path: {filename}")
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _weights_digest(inventory: dict) -> str:
    digest = hashlib.sha256()
    mapping = inventory["weight_map"]
    for filename, header in sorted(inventory["headers"].items()):
        path = _local_file(inventory["root"], filename)
        with path.open("rb") as stream:
            base = 8 + struct.unpack("<Q", stream.read(8))[0]
            for key in sorted(key for key, shard in mapping.items() if shard == filename):
                entry = header[key]
                digest.update(json.dumps([key, entry["dtype"], entry["shape"]]).encode())
                start, end = entry["data_offsets"]
                stream.seek(base + start)
                remaining = end - start
                while remaining:
                    data = stream.read(min(remaining, 8 * 1024 * 1024))
                    if not data:
                        raise ValueError(f"Truncated DSpark tensor {key} in {path}")
                    digest.update(data)
                    remaining -= len(data)
    return digest.hexdigest()


def inspect_deepseek_v4_dspark(source: str | Path) -> dict | None:
    """Validate a native checkpoint or an already preserved draft component."""
    root = Path(source)
    if (root / DSPARK_DIRECTORY / DSPARK_MANIFEST).is_file():
        root = root / DSPARK_DIRECTORY
    config = _read_json(root / "config.json")
    if not config.get("dspark_target_layer_ids"):
        return None
    index_path = root / "model.safetensors.index.json"
    if not index_path.is_file():
        return None
    mapping = _read_json(index_path).get("weight_map", {})
    draft = {key: value for key, value in mapping.items() if key.startswith("mtp.")}
    if not draft:
        return None
    if not any(".markov_head." in key for key in draft) and not all(
            field in config for field in ("dspark_block_size", "dspark_markov_rank", "dspark_noise_token_id")):
        # A legacy MTP layer is not a DSpark component.
        return None
    layers = int(config.get("n_mtp_layers") or 3)
    expected = set(range(layers))
    actual = {int(key.split(".")[1]) for key in draft}
    if actual != expected:
        raise ValueError(f"Incomplete DSpark layers: expected {expected}, got {actual}")
    required = {
        "mtp.0.main_proj.weight", "mtp.0.main_norm.weight",
        f"mtp.{layers - 1}.markov_head.markov_w1.weight",
        f"mtp.{layers - 1}.markov_head.markov_w2.weight",
    }
    for layer in expected:
        required.add(f"mtp.{layer}.attn.wkv.weight")
    missing = required - draft.keys()
    if missing:
        raise ValueError(f"Incomplete DSpark weights: {sorted(missing)}")
    if config.get("model_type") != "deepseek_v4":
        raise ValueError("DSpark requires a DeepSeek-V4 configuration")
    for field in ("dspark_block_size", "dspark_markov_rank", "hidden_size", "vocab_size"):
        if not isinstance(config.get(field), int) or config[field] <= 0:
            raise ValueError(f"DSpark {field} must be a positive integer")
    if not isinstance(config.get("dspark_noise_token_id"), int):
        raise ValueError("DSpark noise token id is missing")
    headers = {}
    total_size = 0
    for key, filename in draft.items():
        if filename not in headers:
            headers[filename] = _header(_local_file(root, filename))
        entry = headers[filename].get(key)
        if not entry:
            raise ValueError(f"DSpark index references missing tensor {key}")
        start, end = entry["data_offsets"]
        total_size += end - start
    digest = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    return {"root": root, "config": config, "weight_map": draft,
            "headers": headers, "total_size": total_size,
            "source_config_sha256": digest, "num_draft_layers": layers}


def preserve_deepseek_v4_dspark(source: str | Path, destination: str | Path) -> dict | None:
    """Atomically add a portable native draft component without rewriting target shards.

    This is also the repair API for previously quantized checkpoints. Shared
    embedding/head are owned by the target; no second copy is serialized here.
    """
    inventory = inspect_deepseek_v4_dspark(source)
    if inventory is None:
        return None
    destination = Path(destination)
    output = destination / DSPARK_DIRECTORY
    manifest = {
        "format_version": 1,
        "weight_format": "deepseek_v4_native_fp4_fp8",
        "source_config_sha256": inventory["source_config_sha256"],
        "weights_sha256": _weights_digest(inventory),
        "num_draft_layers": inventory["num_draft_layers"],
        "tensor_count": len(inventory["weight_map"]),
        "total_size": inventory["total_size"],
        "target_layer_ids": inventory["config"]["dspark_target_layer_ids"],
        "shared_embedding": "model.embed_tokens.weight",
        "shared_lm_head": "lm_head",
    }
    if output.exists():
        existing = inspect_deepseek_v4_dspark(output)
        if (existing is not None and (output / DSPARK_MANIFEST).is_file()
                and _read_json(output / DSPARK_MANIFEST) == manifest
                and _weights_digest(existing) == manifest["weights_sha256"]):
            return manifest
        raise FileExistsError(f"Refusing to replace a different DSpark component: {output}")
    destination.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".dspark-", dir=destination))
    root = inventory["root"]
    mapping = inventory["weight_map"]
    try:
        for filename, header in inventory["headers"].items():
            keys = [key for key, value in mapping.items() if value == filename]
            target_file = staging / filename
            target_file.parent.mkdir(parents=True, exist_ok=True)
            if set(keys) == set(header) - {"__metadata__"}:
                # Official 0731 uses three draft-only shards. Byte copying
                # retains E8M0 scales even on Torch versions lacking that dtype.
                shutil.copy2(root / filename, target_file)
            else:
                from safetensors import safe_open
                from safetensors.torch import save_file

                with safe_open(root / filename, framework="pt", device="cpu") as tensors:
                    values = {key: tensors.get_tensor(key) for key in keys}
                save_file(values, target_file, metadata={"format": "pt"})
                del values
        index = {"metadata": {"total_size": inventory["total_size"]}, "weight_map": mapping}
        for filename, value in (("config.json", inventory["config"]),
                                ("model.safetensors.index.json", index),
                                (DSPARK_MANIFEST, manifest)):
            (staging / filename).write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
        staged = inspect_deepseek_v4_dspark(staging)
        if _weights_digest(staged) != manifest["weights_sha256"]:
            raise ValueError("DSpark weights changed while being preserved")
        staging.rename(output)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return manifest


def prepare_deepseek_v4_dspark_bf16(source: str | Path, output: str | Path,
                                    *, device: str = "cpu") -> dict:
    """Dequantize only the three draft blocks for a target-independent graph export.

    The resulting HF decoder checkpoint has three sliding-only learned-router
    blocks. Context projection and Markov/confidence tensors are separate; the
    consumer supplies the target's embedding and fixed shared LM head.
    """
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file

    from .model_dequant import convert_fp8_shard

    inventory = inspect_deepseek_v4_dspark(source)
    if inventory is None:
        raise ValueError("The checkpoint has no preserved DSpark weights")
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite DSpark preparation: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}-", dir=output.parent))
    root, mapping = inventory["root"], inventory["weight_map"]
    weight_map, heads = {}, {}
    total_size = 0

    class DraftReader:
        def __init__(self, reader, filename):
            self.reader, self.filename = reader, filename

        def keys(self):
            return [key for key, filename in mapping.items() if filename == self.filename]

        def get_tensor(self, key):
            entry = inventory["headers"][self.filename][key]
            if entry["dtype"] == "F8_E8M0" and not hasattr(torch, "float8_e8m0fnu"):
                # Decode direct power-of-two scales without requiring a newer
                # Torch than the export SDK. The raw preserved shard is intact.
                with (root / self.filename).open("rb") as stream:
                    base = 8 + struct.unpack("<Q", stream.read(8))[0]
                    start, end = entry["data_offsets"]
                    stream.seek(base + start)
                    raw = bytearray(stream.read(end - start))
                exponent = torch.frombuffer(raw, dtype=torch.uint8).reshape(entry["shape"]).to(device)
                value = torch.pow(2.0, exponent.float() - 127)
                return torch.where(exponent == 255, torch.nan, value)
            return self.reader.get_tensor(key).to(device)

    try:
        for number, filename in enumerate(sorted(inventory["headers"]), start=1):
            with safe_open(root / filename, framework="pt", device="cpu") as reader:
                converted = convert_fp8_shard(DraftReader(reader, filename), torch.bfloat16,
                                             block_shape=(128, 128), model_type="deepseek_v4")
            block = {}
            for key, value in converted.items():
                _, stage, rest = key.split(".", 2)
                if rest.startswith(("main_proj.", "main_norm.", "markov_head.", "confidence_head.")):
                    heads[rest] = value
                elif rest.startswith(("hc_head_", "norm.")):
                    block[rest] = value
                else:
                    block[f"layers.{stage}.{rest}"] = value
            saved_name = f"model-dspark-{number:05}.safetensors"
            save_file(block, staging / saved_name, metadata={"format": "pt"})
            weight_map.update(dict.fromkeys(block, saved_name))
            total_size += sum(value.numel() * value.element_size() for value in block.values())
            del converted, block
        save_file(heads, staging / "dspark-heads.safetensors", metadata={"format": "pt"})
        config = dict(inventory["config"])
        config.pop("quantization_config", None)
        config.pop("expert_dtype", None)
        config.update(num_hidden_layers=inventory["num_draft_layers"], num_hash_layers=0,
                      num_nextn_predict_layers=0, torch_dtype="bfloat16",
                      layer_types=["sliding_attention"] * inventory["num_draft_layers"],
                      mlp_layer_types=["moe"] * inventory["num_draft_layers"],
                      compress_ratios=[0] * inventory["num_draft_layers"])
        manifest = {"format_version": 1, "source_weights_sha256": _weights_digest(inventory),
                    "target_layer_ids": config["dspark_target_layer_ids"],
                    "num_draft_layers": inventory["num_draft_layers"],
                    "heads": "dspark-heads.safetensors",
                    "shared_embedding": "target", "shared_lm_head": "target"}
        for filename, value in (("config.json", config),
                                ("model.safetensors.index.json", {"metadata": {"total_size": total_size},
                                                                  "weight_map": weight_map}),
                                ("dspark_bf16_preparation.json", manifest)):
            (staging / filename).write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
        staging.rename(output)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return manifest


__all__ = ["inspect_deepseek_v4_dspark", "preserve_deepseek_v4_dspark",
           "prepare_deepseek_v4_dspark_bf16"]
