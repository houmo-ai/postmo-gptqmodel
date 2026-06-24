# SPDX-License-Identifier: Apache-2.0
"""
Restore tensors that are stripped during model loading/saving.

When ``transformers`` loads a model, it only instantiates sub-modules declared
in the model class.  Auxiliary parameters — notably **MTP (Multi-Token Prediction)**
layers — are stored in the original checkpoint but are silently omitted when the
model is saved via ``save_pretrained``.

This module provides :func:`copy_missing_tensors_from_source`, which detects and
copies such tensors from the original checkpoint into a sidecar file in the
saved output directory.
Detection works by comparing parent layer names present in the source checkpoint
against those present in the saved output: any source tensor whose parent layer
has no tensors at all in the saved output (and is itself absent) is considered
"missing".
"""

from __future__ import annotations

import json
import os
from typing import Set

from safetensors import safe_open
from safetensors.torch import save_file

from ..utils.logger import setup_logger

log = setup_logger()


def _get_tensor_files(model_dir: str):
    """Return (index_path, shard_to_path_dict) for a model directory."""
    index_path = os.path.join(model_dir, "model.safetensors.index.json")
    if os.path.exists(index_path):
        with open(index_path) as f:
            idx = json.load(f)
        wmap = idx.get("weight_map", {})
        shard_to_path = {shard: os.path.join(model_dir, shard) for shard in set(wmap.values())}
        return index_path, shard_to_path
    single = os.path.join(model_dir, "model.safetensors")
    if os.path.exists(single):
        return None, {single: single}
    return None, {}


def _load_tensor_names(shard_to_path: dict) -> Set[str]:
    """Load all tensor names from a set of shard files."""
    names = set()
    for shard_path in shard_to_path.values():
        with safe_open(shard_path, framework="pt", device="cpu") as f:
            for key in f.keys():
                if isinstance(key, str):
                    names.add(key)
    return names


def _first_numeric_prefix(tensor_name: str):
    """Return path up to and including the first numeric segment.

    e.g. ``model.layers.0.mlp.gate.qweight`` → ``model.layers.0``
         ``model.embed_tokens.weight``         → ``None``
    """
    if not isinstance(tensor_name, str):
        return None
    parts = tensor_name.split(".")
    for i, p in enumerate(parts):
        if p.isdigit():
            return ".".join(parts[: i + 1])
    return None


def copy_missing_tensors_from_source(
    source_dir: str,
    target_dir: str,
) -> int:
    """Copy tensors from source checkpoint that are absent from the saved output.

    Some parameters (e.g., MTP layers) are not loaded by ``transformers`` and
    therefore not written by ``save_pretrained``.  This function copies those
    tensors from the original checkpoint into a dedicated sidecar in the
    quantized output dir, then patches ``model.safetensors.index.json``.  It
    intentionally avoids rewriting existing model shards, because those shards
    may contain mmap-backed quantized buffers and must remain immutable after
    the main save path completes.

    Detection works by comparing **parent layer names** between source and target.
    A source tensor is considered "missing" when:
      1. its name is not present in the saved output, AND
      2. its parent layer name is not among the parent layers of any saved tensor
         (i.e. the entire layer was skipped by save_pretrained), AND
      3. its "block prefix" (path up to first numeric segment) is not present
         in the saved output.

    Args:
        source_dir: Path to the original (pre-quantization) checkpoint directory.
        target_dir: Directory to which the quantized model was saved.  Must
            contain ``model.safetensors`` or ``model.safetensors.index.json``.

    Returns:
        Number of tensors copied.
    """
    # ── Load source tensor names ──────────────────────────────────────────────
    _, source_shards = _get_tensor_files(source_dir)
    if not source_shards:
        log.debug("copy_missing_tensors: no safetensors found in source %s", source_dir)
        return 0
    source_names = _load_tensor_names(source_shards)
    if not source_names:
        return 0

    # Build source tensor_name → shard path mapping
    source_name_to_shard = {}
    if os.path.exists(os.path.join(source_dir, "model.safetensors.index.json")):
        with open(os.path.join(source_dir, "model.safetensors.index.json")) as f:
            wmap = json.load(f).get("weight_map", {})
        source_name_to_shard = {name: os.path.join(source_dir, shard) for name, shard in wmap.items()}
    else:
        single = os.path.join(source_dir, "model.safetensors")
        with safe_open(single, framework="pt", device="cpu") as f:
            for key in f.keys():
                source_name_to_shard[key] = single

    # ── Load target tensor names ─────────────────────────────────────────────
    _, target_shards = _get_tensor_files(target_dir)
    if not target_shards:
        log.debug("copy_missing_tensors: no safetensors found in target %s", target_dir)
        return 0
    target_names = _load_tensor_names(target_shards)

    # Build target parent layers and block prefixes
    target_parent_layers = {name.rsplit(".", 1)[0] for name in target_names if isinstance(name, str)}
    # Also add shortcut forms (e.g. "language_model.layers.0" from "model.language_model.layers.0")
    # Iterate over a list copy to avoid "Set changed size during iteration"
    for name in list(target_parent_layers):
        if isinstance(name, str) and "." in name:
            target_parent_layers.add(name.split(".", 1)[1])

    target_block_prefixes = set()
    for name in target_names:
        bp = _first_numeric_prefix(name)
        if bp is not None:
            target_block_prefixes.add(bp)
            # Also add shortcut form
            if "." in bp:
                target_block_prefixes.add(bp.split(".", 1)[1])

    # ── Identify missing tensors ─────────────────────────────────────────────
    to_copy = {}
    for name in source_names:
        if not isinstance(name, str):
            continue
        if name in target_names:
            continue
        parent = name.rsplit(".", 1)[0]
        if parent in target_parent_layers:
            continue
        bp = _first_numeric_prefix(name)
        if bp is not None and bp in target_block_prefixes:
            continue
        to_copy[name] = source_name_to_shard[name]

    if not to_copy:
        log.debug("copy_missing_tensors: no missing tensors found")
        return 0

    log.info("copy_missing_tensors: found %d missing tensors in %s", len(to_copy), target_dir)

    # ── Sidecar copy ────────────────────────────────────────────────────────
    # Never rewrite an existing safetensors shard here.  Qwen3.5 no-rotation
    # MTP restore used to append the passthrough tensors to an already-written
    # model shard, which can corrupt mmap-backed qweight/qzeros/scales buffers
    # when the same file is read and overwritten in-place.  Match the rotated
    # MTP path instead: write a dedicated sidecar and patch the index.
    sidecar_name = "missing_tensors_passthrough.safetensors"
    sidecar_path = os.path.join(target_dir, sidecar_name)
    sidecar_tensors = {}
    for name, src_shard_path in to_copy.items():
        if not isinstance(name, str):
            continue
        with safe_open(src_shard_path, framework="pt", device="cpu") as f:
            sidecar_tensors[name] = f.get_tensor(name)
    save_file(
        sidecar_tensors,
        sidecar_path,
        metadata={"format": "pt", "produced_by": "gptqmodel.utils.missing_tensors"},
    )

    target_index_path = os.path.join(target_dir, "model.safetensors.index.json")
    if os.path.exists(target_index_path):
        with open(target_index_path) as f:
            idx = json.load(f)
    else:
        idx = {"metadata": {}, "weight_map": {}}
        single_path = os.path.join(target_dir, "model.safetensors")
        if os.path.exists(single_path):
            with safe_open(single_path, framework="pt", device="cpu") as f:
                idx["weight_map"].update({key: "model.safetensors" for key in f.keys()})

    wmap = idx.setdefault("weight_map", {})
    for name in to_copy:
        if isinstance(name, str):
            wmap[name] = sidecar_name

    meta = idx.setdefault("metadata", {})
    try:
        shard_names = set(wmap.values())
        meta["total_size"] = sum(
            os.path.getsize(os.path.join(target_dir, shard_name))
            for shard_name in shard_names
            if os.path.exists(os.path.join(target_dir, shard_name))
        )
    except OSError:
        pass

    with open(target_index_path, "w") as f:
        json.dump(idx, f, indent=2, sort_keys=True)
        f.write("\n")

    log.info("copy_missing_tensors: copied %d tensors to sidecar %s", len(to_copy), sidecar_name)
    return len(to_copy)
