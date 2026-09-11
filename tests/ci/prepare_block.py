#!/usr/bin/env python3
"""Cut representative Qwen3.5/Gemma4 text layers from local floating checkpoints.

Preserve real dimensions/experts and auxiliary weights. Gemma shared-KV layers
keep their actual source donor layers, including packed per-layer embeddings.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
import shutil
import tempfile
from pathlib import Path


def layer_plan(config):
    result = copy.deepcopy(config)
    text = result.get("text_config", result)
    family = result.get("model_type", "")
    if family not in ("qwen3_5", "qwen3_5_moe", "gemma4"):
        raise ValueError(f"unsupported block family: {family}")
    if result.get("quantization_config") or text.get("quantization_config"):
        raise ValueError("block input must be a floating checkpoint, not packed/FP8 weights")
    types = text.get("layer_types", [])
    count = text.get("num_hidden_layers")
    if not types or len(types) != count:
        raise ValueError("explicit layer_types must match num_hidden_layers")
    expected = {"sliding_attention", "full_attention"} if family == "gemma4" else {"linear_attention", "full_attention"}
    if set(types) != expected:
        raise ValueError(f"fixture must retain both attention kinds: {sorted(expected)}")
    shared = text.get("num_kv_shared_layers", 0)
    if not isinstance(shared, int) or not 0 <= shared < count:
        raise ValueError("invalid num_kv_shared_layers")
    boundary = count - shared
    selected = set()
    # For Gemma shared KV, a borrower reads the last earlier non-shared layer
    # of the same attention kind. Keep those exact donors and one borrower.
    for kind in dict.fromkeys(types):
        donors = [i for i in range(boundary) if types[i] == kind]
        if not donors:
            raise ValueError(f"missing non-shared KV donor for {kind}")
        selected.add(donors[-1] if shared else donors[0])
        borrowers = [i for i in range(boundary, count) if types[i] == kind]
        if borrowers:
            selected.add(borrowers[0])
    # A sparse architecture may also have ordinary dense MLP layers. Keep a
    # representative of each attention/MLP combination rather than dropping it.
    dense_layers = set(text.get("mlp_only_layers", []))
    step = text.get("decoder_sparse_step", 1)
    if step != 1:
        raise ValueError("sparse-step remapping requires a family-specific cutter")
    if dense_layers:
        for kind in dict.fromkeys(types):
            for dense in (False, True):
                candidates = [i for i in range(boundary) if types[i] == kind and (i in dense_layers) == dense]
                if candidates:
                    selected.add(candidates[0])
    indices = sorted(selected)
    text["num_hidden_layers"] = len(indices)
    text["layer_types"] = [types[i] for i in indices]
    if "per_layer_config" in text:
        if not isinstance(text["per_layer_config"], list) or len(text["per_layer_config"]) != count:
            raise ValueError("per_layer_config must have one entry per source layer")
        text["per_layer_config"] = [text["per_layer_config"][i] for i in indices]
    if "mlp_only_layers" in text:
        text["mlp_only_layers"] = [i for i, old in enumerate(indices) if old in dense_layers]
    if "num_kv_shared_layers" in text:
        text["num_kv_shared_layers"] = sum(i >= boundary for i in indices)
    if "full_attention_interval" in text and text["layer_types"] == ["linear_attention", "full_attention"]:
        text["full_attention_interval"] = 2
    # This fixture owns text-backbone quantization. MTP has its own real test.
    for key in ("mtp_num_hidden_layers", "num_nextn_predict_layers"):
        if key in text:
            text[key] = 0
    return result, indices


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def validate_shapes(checkpoint):
    """Match the real HF module tree without allocating model weights/GPU."""
    import torch
    from safetensors import safe_open
    from transformers import AutoConfig, AutoModelForImageTextToText

    config = AutoConfig.from_pretrained(checkpoint, local_files_only=True)
    with torch.device("meta"):
        model = AutoModelForImageTextToText.from_config(config, dtype=torch.bfloat16)
    expected = {key: list(value.shape) for key, value in model.state_dict().items()}
    actual = {}
    for path in checkpoint.glob("*.safetensors"):
        with safe_open(path, framework="pt", device="cpu") as reader:
            for key in reader.keys():
                if key in actual:
                    raise ValueError(f"duplicate block tensor: {key}")
                actual[key] = reader.get_slice(key).get_shape()
    tied = getattr(model, "all_tied_weights_keys", {})
    missing = [key for key in expected.keys() - actual.keys() if tied.get(key) not in actual]
    extra = sorted(actual.keys() - expected.keys())
    mismatched = {key: [actual[key], expected[key]] for key in actual.keys() & expected.keys() if actual[key] != expected[key]}
    if missing or extra or mismatched:
        raise ValueError(f"block/HF mismatch: missing={sorted(missing)}, extra={extra}, shapes={mismatched}")
    return {"checkpoint": str(checkpoint), "tensor_count": len(actual), "status": "shapes passed",
            "scope": "HF module tree and tensor shapes; no forward or quantization"}


def prepare(source, output):
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file

    source, output = source.resolve(), output.resolve()
    if output.exists() or output == source or source in output.parents:
        raise ValueError("use a new output directory outside the source checkpoint")
    config_path = source / "config.json"
    original = json.loads(config_path.read_text())
    config, indices = layer_plan(original)
    files = sorted(source.glob("*.safetensors"))
    if not files:
        raise ValueError("source has no safetensors shards")
    key_files = {}
    for path in files:
        with safe_open(path, framework="pt", device="cpu") as reader:
            for key in reader.keys():
                if key.endswith((".qweight", ".qzeros", ".weight_scale_inv")):
                    raise ValueError("packed/FP8 weights cannot serve as a floating quantization fixture")
                if key in key_files:
                    raise ValueError(f"duplicate source tensor: {key}")
                key_files[key] = path
    for index in source.glob("*.safetensors.index.json"):
        expected = json.loads(index.read_text())["weight_map"]
        missing = set(expected) - key_files.keys()
        misplaced = [key for key in expected.keys() & key_files.keys() if key_files[key].name != expected[key]]
        if missing or misplaced:
            raise ValueError(f"incomplete source checkpoint: missing={sorted(missing)}, misplaced={misplaced}")
    prefixes = ("model.language_model.layers.", "model.layers.")
    prefix = next((p for p in prefixes if any(key.startswith(p) for key in key_files)), None)
    if prefix is None:
        raise ValueError("cannot locate the text layer prefix")
    mapping = {old: new for new, old in enumerate(indices)}
    text = config.get("text_config", config)
    per_layer_width = text.get("hidden_size_per_layer_input", 0)
    tensors, loaded_layers = {}, set()
    for path in files:
        with safe_open(path, framework="pt", device="cpu") as reader:
            for key in reader.keys():
                if key.startswith("mtp.") or ".mtp." in key:
                    continue
                name = key
                if key.startswith(prefix):
                    match = re.fullmatch(re.escape(prefix) + r"(\d+)(\..+)", key)
                    if not match:
                        raise ValueError(f"invalid text tensor key: {key}")
                    old = int(match[1])
                    if old not in mapping:
                        continue
                    # Some original Gemma checkpoints retain redundant K/V
                    # tensors for shared layers. HF reads these layers' KV
                    # from their donors and does not instantiate these modules.
                    source_text = original.get("text_config", original)
                    shared_start = source_text["num_hidden_layers"] - source_text.get("num_kv_shared_layers", 0)
                    if original["model_type"] == "gemma4" and old >= shared_start and match[2].startswith(
                        (".self_attn.k_proj.", ".self_attn.v_proj.", ".self_attn.k_norm.")
                    ):
                        continue
                    name = f"{prefix}{mapping[old]}{match[2]}"
                    loaded_layers.add(old)
                if per_layer_width and key.endswith("embed_tokens_per_layer.weight"):
                    view = reader.get_slice(key)
                    if view.get_shape()[1] != original.get("text_config", original)["num_hidden_layers"] * per_layer_width:
                        raise ValueError("invalid packed per-layer embedding width")
                    tensors[name] = torch.cat([view[:, i * per_layer_width:(i + 1) * per_layer_width] for i in indices], dim=1)
                elif per_layer_width and key.endswith("per_layer_model_projection.weight"):
                    view = reader.get_slice(key)
                    if view.get_shape()[0] != original.get("text_config", original)["num_hidden_layers"] * per_layer_width:
                        raise ValueError("invalid packed per-layer projection height")
                    tensors[name] = torch.cat([view[i * per_layer_width:(i + 1) * per_layer_width] for i in indices], dim=0)
                else:
                    tensors[name] = reader.get_tensor(key)
    if loaded_layers != set(indices):
        raise ValueError(f"missing selected layers: {set(indices) - loaded_layers}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent, prefix="prepare-block-") as temporary:
        destination = Path(temporary) / "checkpoint"
        destination.mkdir()
        for path in source.iterdir():
            if path.is_file() and path.name != "config.json" and not path.name.endswith((".safetensors", ".index.json", ".bin", ".pt")):
                shutil.copy2(path, destination / path.name)
        (destination / "config.json").write_text(json.dumps(config, indent=2) + "\n")
        weights = destination / "model.safetensors"
        save_file(tensors, str(weights), metadata={"format": "pt"})
        report = {"version": 1, "source": str(source), "source_config_sha256": _sha256(config_path),
                  "source_shards": [{"name": p.name, "size": p.stat().st_size} for p in files],
                  "source_layers": indices, "layer_types": text["layer_types"],
                  "num_kv_shared_layers": text.get("num_kv_shared_layers", 0),
                  "omitted": ["MTP", "unused Gemma shared-layer K/V tensors"],
                  "scope": "text backbone; original widths/experts and auxiliary weights retained; MTP excluded",
                  "weights_sha256": _sha256(weights), "weights_bytes": weights.stat().st_size}
        (destination / "ci_block_provenance.json").write_text(json.dumps(report, indent=2) + "\n")
        destination.rename(output)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--plan", action="store_true")
    mode.add_argument("--check-shapes", action="store_true", help="Validate an existing block against the installed HF model on meta device")
    args = parser.parse_args()
    if args.plan:
        config, indices = layer_plan(json.loads((args.source / "config.json").read_text()))
        text = config.get("text_config", config)
        print(json.dumps({"source_layers": indices, "text_config": text}, indent=2))
    elif args.check_shapes:
        print(json.dumps(validate_shapes(args.source.resolve()), indent=2))
    elif args.output is None:
        parser.error("--output is required unless --plan or --check-shapes is used")
    else:
        print(json.dumps(prepare(args.source, args.output), indent=2))


if __name__ == "__main__":
    main()
