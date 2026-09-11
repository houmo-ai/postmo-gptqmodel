#!/usr/bin/env python3
"""Cut a real Ling3.0 block retaining KDA/dense, KDA/MoE and MLA/MoE."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
import shutil
import tempfile
from pathlib import Path


def layer_plan(original, keep_mtp=False):
    config = copy.deepcopy(original)
    if config.get("model_type") != "bailing_hybrid" or config.get("quantization_config"):
        raise ValueError("requires an unquantized Ling bailing_hybrid checkpoint")
    if config.get("ling_mla_layout", "hf_unabsorbed") != "hf_unabsorbed":
        raise ValueError("start from unabsorbed MLA weights")
    count, group = config["num_hidden_layers"], config["layer_group_size"]
    dense = config["first_k_dense_replace"]
    if not 0 < dense < count or not 1 < group <= count or not config.get("num_experts"):
        raise ValueError("source must contain KDA/dense, KDA/MoE and MLA/MoE")

    def kind(index):
        mla = (index + 1) % group == 0 or index >= count // group * group
        return ("mla" if mla else "kda", "dense" if index < dense else "moe")

    indices = []
    for required in (("kda", "dense"), ("kda", "moe"), ("mla", "moe")):
        candidates = [i for i in range(count) if kind(i) == required]
        if not candidates:
            raise ValueError(f"missing Ling layer kind: {required}")
        indices.append(candidates[0])
    if indices != sorted(indices):
        raise ValueError("source layer order requires a different Ling cutter")
    mtp = config.get("num_nextn_predict_layers", 0)
    if keep_mtp:
        if mtp != 1:
            raise ValueError("--keep-mtp requires exactly one original MTP layer")
        indices.append(count)
    config.update(num_hidden_layers=3, layer_group_size=3, first_k_dense_replace=1,
                  num_nextn_predict_layers=int(keep_mtp))
    for name in ("expert_swiglu_limit_list", "share_expert_swiglu_limit_list"):
        if name in config:
            values = original[name]
            if len(values) not in (count, count + mtp):
                raise ValueError(f"invalid per-layer list: {name}")
            config[name] = [values[i] for i in indices if i < len(values)]
    return config, indices


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while data := stream.read(8 * 1024 * 1024):
            digest.update(data)
    return digest.hexdigest()


def prepare(source, output, keep_mtp=False):
    from safetensors import safe_open
    from safetensors.torch import save_file

    source, output = source.resolve(), output.resolve()
    if output.exists() or source == output or source in output.parents:
        raise ValueError("use a new output directory outside the source checkpoint")
    config, indices = layer_plan(json.loads((source / "config.json").read_text()), keep_mtp)
    shards = sorted(source.glob("*.safetensors"))
    inventory = {}
    for shard in shards:
        with safe_open(shard, framework="pt", device="cpu") as reader:
            for key in reader.keys():
                if key in inventory or key.endswith((".qweight", ".qzeros", ".weight_scale_inv")):
                    raise ValueError(f"duplicate or quantized source tensor: {key}")
                inventory[key] = shard.name
    index = source / "model.safetensors.index.json"
    if index.exists() and json.loads(index.read_text())["weight_map"] != inventory:
        raise ValueError("source tensor inventory does not match checkpoint index")
    mapping = {old: new for new, old in enumerate(indices)}
    tensors, layers = {}, set()
    for shard in shards:
        with safe_open(shard, framework="pt", device="cpu") as reader:
            for key in reader.keys():
                name = key
                if key.startswith("model.layers."):
                    match = re.fullmatch(r"model\.layers\.(\d+)(\..+)", key)
                    if not match:
                        raise ValueError(f"invalid Ling layer key: {key}")
                    old = int(match[1])
                    if old not in mapping:
                        continue
                    layers.add(old)
                    name = f"model.layers.{mapping[old]}{match[2]}"
                tensors[name] = reader.get_tensor(key)
    if layers != set(indices):
        raise ValueError(f"missing selected layers: {set(indices) - layers}")
    code = [source / name for name in ("configuration_bailing_moe_v3.py", "modeling_bailing_moe_v3.py")]
    if not all(path.is_file() for path in code):
        raise ValueError("Ling block requires checkpoint-side config and model code")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent, prefix="prepare-ling-") as temporary:
        destination = Path(temporary) / "checkpoint"
        destination.mkdir()
        for path in source.iterdir():
            if path.is_file() and path.name != "config.json" and not path.name.endswith((".safetensors", ".index.json", ".bin", ".pt")):
                shutil.copy2(path, destination / path.name)
        (destination / "config.json").write_text(json.dumps(config, indent=2) + "\n")
        weights = destination / "model.safetensors"
        save_file(tensors, str(weights), metadata={"format": "pt"})
        report = {"version": 1, "source": str(source), "source_config_sha256": _sha256(source / "config.json"),
                  "source_code_sha256": {path.name: _sha256(path) for path in code}, "source_layers": indices,
                  "structures": ["KDA+dense", "KDA+MoE", "MLA+MoE"] + (["MTP"] if keep_mtp else []),
                  "weights_sha256": _sha256(weights), "weights_bytes": weights.stat().st_size,
                  "scope": "original dimensions, experts and router; representative text layer mechanisms"}
        (destination / "ci_block_provenance.json").write_text(json.dumps(report, indent=2) + "\n")
        destination.rename(output)
    return report


def validate_block(checkpoint, forward=False):
    import torch
    from safetensors import safe_open
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from gptqmodel.models.definitions.ling_3_flash import (
        load_ling_3_flash_config,
        patch_ling_3_flash_remote_code,
        refresh_ling_rotary_buffers,
    )

    config = load_ling_3_flash_config(str(checkpoint))
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(config, trust_remote_code=True, attn_implementation="eager")
    patch_ling_3_flash_remote_code()
    expected = {key: list(value.shape) for key, value in model.state_dict().items()}
    with safe_open(checkpoint / "model.safetensors", framework="pt", device="cpu") as reader:
        actual = {key: reader.get_slice(key).get_shape() for key in reader.keys()}
    mismatched = [key for key in actual.keys() & expected.keys() if actual[key] != expected[key]]
    if actual.keys() != expected.keys() or mismatched:
        raise ValueError(f"Ling block/HF mismatch: missing={sorted(expected.keys() - actual.keys())}, "
                         f"extra={sorted(actual.keys() - expected.keys())}, shapes={mismatched}")
    report = {"checkpoint": str(checkpoint), "tensor_count": len(actual), "shapes": "passed",
              "structures": [{"attention": type(layer.attention).__name__, "mlp": type(layer.mlp).__name__}
                             for layer in model.model.layers]}
    del model
    if forward:
        if not torch.cuda.is_available():
            raise ValueError("Ling KDA forward requires a CUDA device")
        model = AutoModelForCausalLM.from_pretrained(
            str(checkpoint), config=config, trust_remote_code=True, dtype=torch.bfloat16,
            device_map="cuda:0", attn_implementation="eager", local_files_only=True,
        ).eval()
        patch_ling_3_flash_remote_code()
        refresh_ling_rotary_buffers(model)
        tokenizer = AutoTokenizer.from_pretrained(checkpoint, config=config, trust_remote_code=True, local_files_only=True)
        tokens = tokenizer("Hello, how are you?", return_tensors="pt")["input_ids"].to("cuda:0")
        with torch.inference_mode():
            logits = model(input_ids=tokens, use_cache=False).logits
        if logits.shape[:2] != tokens.shape or not torch.isfinite(logits).all():
            raise ValueError("Ling block forward produced invalid logits")
        report.update(forward="BF16 passed", logits_shape=list(logits.shape))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--keep-mtp", action="store_true")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--plan", action="store_true")
    mode.add_argument("--check-shapes", action="store_true")
    mode.add_argument("--check-forward", action="store_true", help="Check shapes and run BF16 text forward on the allocated CUDA device")
    args = parser.parse_args()
    if args.plan:
        config, indices = layer_plan(json.loads((args.source / "config.json").read_text()), args.keep_mtp)
        print(json.dumps({"source_layers": indices, "config": config}, indent=2))
    elif args.check_shapes or args.check_forward:
        print(json.dumps(validate_block(args.source.resolve(), args.check_forward), indent=2))
    elif args.output is None:
        parser.error("--output is required unless planning or validating an existing block")
    else:
        print(json.dumps(prepare(args.source, args.output, args.keep_mtp), indent=2))


if __name__ == "__main__":
    main()
