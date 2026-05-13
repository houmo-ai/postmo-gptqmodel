"""Convert VLM quantized models to text-only format for xhquant_llm compatibility.

Our auto-round quantized models are in VLM format (Qwen3_5ForConditionalGeneration)
with weight keys like `model.language_model.layers.X...`. xhquant_llm's GPTQModel
loader expects text-only format (Qwen3_5ForCausalLM) with keys like `model.layers.X...`.

This script:
1. Renames weight keys: strips `language_model.` prefix
2. Removes visual model weights (model.visual.*)
3. Dequantizes in_proj_a/in_proj_b weights (GPTQModel marks them non-quantizable)
4. Updates config.json: architectures -> Qwen3_5ForCausalLM, model_type -> qwen3_5_text
5. Generates quantize_config.json for GPTQModel compatibility
6. Ensures quantization_config has `meta.quantizer` for dequantize routing
7. Copies tokenizer with chat_template (from --base-model if source lacks it)
"""

import argparse
import json
import os
import shutil
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file


# Modules that GPTQModel's Qwen3_5QModel marks as non-quantizable (:! flag).
# If auto-round quantized these, we must dequantize them back to float.
GPTQMODEL_NON_QUANT_MODULES = {"in_proj_a", "in_proj_b"}

GPTQ_QUANT_SUFFIXES = (".qweight", ".qzeros", ".scales", ".g_idx")


def _dequantize_gptq_4bit(qweight: torch.Tensor, qzeros: torch.Tensor,
                           scales: torch.Tensor, g_idx: torch.Tensor,
                           bits: int = 4) -> torch.Tensor:
    """Dequantize a GPTQ 4-bit packed weight tensor back to float.

    Standard GPTQ layout: int32 packs 8 × 4-bit values.
    Formula: weight_float = scales[g_idx] * (unpacked_qweight - unpacked_qzeros[g_idx])
    """
    pack_factor = 32 // bits
    maxq = (1 << bits) - 1
    # bit-shift offsets: [0, 4, 8, 12, 16, 20, 24, 28] for 4-bit
    shifts = torch.arange(0, 32, bits, dtype=torch.int32)

    # Unpack qzeros: [num_groups, packed_out] -> [num_groups, out_features]
    qzeros_expanded = qzeros.unsqueeze(2).expand(-1, -1, pack_factor)
    zeros = (qzeros_expanded >> shifts.view(1, 1, -1)) & maxq
    zeros = zeros.reshape(scales.shape).to(scales.dtype)

    # Unpack qweight: [packed_in, out_features] -> [in_features, out_features]
    qweight_expanded = qweight.unsqueeze(1).expand(-1, pack_factor, -1)
    weight = (qweight_expanded >> shifts.view(1, -1, 1)) & maxq
    weight = weight.reshape(-1, qweight.shape[1]).to(scales.dtype)

    # Dequantize using group index
    g_idx_long = g_idx.long()
    dequant = scales[g_idx_long] * (weight - zeros[g_idx_long])
    return dequant


def _is_quant_module_key(key: str) -> bool:
    """Check if a weight key belongs to a quantized module (has qweight/qzeros/scales/g_idx)."""
    return any(key.endswith(s) for s in GPTQ_QUANT_SUFFIXES)


def _get_module_name(key: str) -> str:
    """Extract module name from a weight key (strip the .qweight/.scales/etc suffix)."""
    for s in GPTQ_QUANT_SUFFIXES:
        if key.endswith(s):
            return key[: -len(s)]
    return key


def _should_dequantize(key: str) -> bool:
    """Check if a quantized weight key belongs to a module that needs dequantization."""
    module = _get_module_name(key)
    leaf = module.rsplit(".", 1)[-1] if "." in module else module
    return leaf in GPTQMODEL_NON_QUANT_MODULES


def _ensure_quantization_meta(qcfg: dict) -> dict:
    """Ensure quantization_config has `meta.quantizer` for xhquant_llm routing.

    xhquant_llm's dequantize_hf_model() checks meta.quantizer to route to
    _dequantize_gptqmodel(). Without it, models loaded via GPTQModel.load()
    fall through to _dequantize_gptq_hf_model() which requires hf_quantizer
    attribute (absent on GPTQModel-loaded models).
    """
    if "meta" not in qcfg:
        qcfg["meta"] = {}
    meta = qcfg["meta"]
    if "quantizer" not in meta:
        meta["quantizer"] = []
    quantizers = meta["quantizer"]
    if not any("gptqmodel" in q for q in quantizers):
        quantizers.append("auto-round-via-gptqmodel")
    if "uri" not in meta:
        meta["uri"] = "https://github.com/intel/auto-round"
    return qcfg


def _update_quantization_config(qcfg: dict) -> dict:
    """Fix quantization_config for text-only model:
    - Update block_name_to_quantize path prefix
    - Clean up dynamic entries (remove visual-only, rename language_model prefix)
    """
    # Fix block_name_to_quantize
    if qcfg.get("block_name_to_quantize") == "model.language_model.layers":
        qcfg["block_name_to_quantize"] = "model.layers"

    # Clean up dynamic entries
    if "dynamic" in qcfg:
        new_dynamic = {}
        for pattern, value in qcfg["dynamic"].items():
            # Remove visual-only entries
            if "visual" in pattern or "merger" in pattern:
                continue
            # Rename language_model prefix
            new_pattern = pattern.replace("model\\.language_model\\.", "model\\.")
            new_dynamic[new_pattern] = value
        qcfg["dynamic"] = new_dynamic

    return qcfg


def convert_model(src_dir: str, dst_dir: str, keep_visual: bool = False,
                  base_model_dir: str | None = None):
    src_path = Path(src_dir)
    dst_path = Path(dst_dir)
    dst_path.mkdir(parents=True, exist_ok=True)

    # 1. Process config.json
    with open(src_path / "config.json") as f:
        config = json.load(f)

    # Extract text_config if present and merge into top-level
    text_config = config.get("text_config", {})
    new_config = dict(text_config)
    new_config["architectures"] = ["Qwen3_5ForCausalLM"]
    new_config["model_type"] = "qwen3_5_text"
    # Keep quantization_config from original
    if "quantization_config" in config:
        new_config["quantization_config"] = config["quantization_config"]
    # Keep important top-level fields
    for key in ["torch_dtype", "transformers_version"]:
        if key in config and key not in new_config:
            new_config[key] = config[key]

    # Ensure meta.quantizer is present for dequantize routing
    if "quantization_config" in new_config:
        _ensure_quantization_meta(new_config["quantization_config"])
        _update_quantization_config(new_config["quantization_config"])
        print(f"  config.json: ensured meta.quantizer and updated quantization_config paths")

    with open(dst_path / "config.json", "w") as f:
        json.dump(new_config, f, indent=2, ensure_ascii=False)
    print(f"  config.json: updated architectures to Qwen3_5ForCausalLM")

    # 2. Process safetensors: rename keys, dequantize incompatible modules
    st_files = sorted(src_path.glob("*.safetensors"))
    print(f"  Found {len(st_files)} safetensor files")

    # Read quantization config for dequantization parameters
    bits = 4
    group_size = 64
    if "quantization_config" in config:
        bits = config["quantization_config"].get("bits", 4)
        group_size = config["quantization_config"].get("group_size", 64)
    # Also try separate quantization_config.json
    for qcfg_name in ["quantize_config.json", "quantization_config.json"]:
        qcfg_file = src_path / qcfg_name
        if qcfg_file.exists():
            with open(qcfg_file) as f:
                _qcfg = json.load(f)
            bits = _qcfg.get("bits", bits)
            group_size = _qcfg.get("group_size", group_size)
            break

    # First pass: collect all tensors and identify modules needing dequantization
    all_tensors = {}  # new_key -> tensor
    for st_file in st_files:
        with safe_open(str(st_file), framework="pt", device="cpu") as sf:
            for key in sf.keys():
                all_tensors[key] = sf.get_tensor(key)

    # Collect quantized tensors for modules that need dequantization
    dequant_modules = {}  # module_name -> {suffix: tensor}
    for key, tensor in all_tensors.items():
        if _is_quant_module_key(key) and _should_dequantize(key):
            module = _get_module_name(key)
            suffix = key[len(module):]  # e.g. ".qweight"
            dequant_modules.setdefault(module, {})[suffix] = tensor

    # Perform dequantization
    dequant_results = {}  # module_name -> float weight tensor
    for module, tensors in dequant_modules.items():
        if all(s in tensors for s in GPTQ_QUANT_SUFFIXES):
            dequant_results[module] = _dequantize_gptq_4bit(
                tensors[".qweight"], tensors[".qzeros"],
                tensors[".scales"], tensors[".g_idx"], bits=bits)
            print(f"    Dequantized: {module} -> shape {dequant_results[module].shape}")

    # Second pass: build output tensors with renaming, dequantization, and filtering
    all_renamed = {}
    total_kept = 0
    total_skipped = 0
    total_dequantized = 0
    shard_idx = 1
    MAX_SHARD_SIZE = 5_000_000_000  # ~5GB per shard
    current_shard_size = 0
    current_shard = {}

    for key, tensor in all_tensors.items():
        # Skip visual model weights
        if not keep_visual and ("visual" in key or "merger" in key):
            total_skipped += 1
            continue

        # Rename: model.language_model.X -> model.X
        new_key = key
        if "model.language_model." in key:
            new_key = key.replace("model.language_model.", "model.")

        # Handle dequantized modules
        module = _get_module_name(key)
        if module in dequant_results:
            if key.endswith(".qweight"):
                # Replace qweight with dequantized float weight
                # GPTQ stores as [in_features, out_features], nn.Linear expects [out_features, in_features]
                float_weight = dequant_results[module].t().contiguous().to(torch.bfloat16)
                new_key = _get_module_name(new_key) + ".weight"
                current_shard[new_key] = float_weight
                current_shard_size += float_weight.numel() * float_weight.element_size()
                total_dequantized += 1
                total_kept += 1
            else:
                # Skip other quant tensors (qzeros, scales, g_idx) for dequantized modules
                continue
        else:
            current_shard[new_key] = tensor
            current_shard_size += tensor.numel() * tensor.element_size()
            total_kept += 1

        # Write shard if large enough
        if current_shard_size >= MAX_SHARD_SIZE:
            shard_name = f"model-{shard_idx:05d}-of-PLACEHOLDER.safetensors"
            for k in current_shard:
                all_renamed[k] = shard_idx
            save_file(current_shard, str(dst_path / shard_name))
            print(f"    Wrote shard {shard_idx}: {len(current_shard)} tensors, {current_shard_size/1e9:.1f}GB")
            shard_idx += 1
            current_shard = {}
            current_shard_size = 0

    # Write remaining shard
    if current_shard:
        shard_name = f"model-{shard_idx:05d}-of-PLACEHOLDER.safetensors"
        for k in current_shard:
            all_renamed[k] = shard_idx
        save_file(current_shard, str(dst_path / shard_name))
        print(f"    Wrote shard {shard_idx}: {len(current_shard)} tensors, {current_shard_size/1e9:.1f}GB")

    total_shards = shard_idx
    print(f"  Renamed {total_kept} tensors, skipped {total_skipped} visual tensors, "
          f"dequantized {total_dequantized} modules")

    # Rename shard files with correct total count
    weight_map = {}
    for i in range(1, total_shards + 1):
        old_name = f"model-{i:05d}-of-PLACEHOLDER.safetensors"
        new_name = f"model-{i:05d}-of-{total_shards:05d}.safetensors"
        if (dst_path / old_name).exists():
            (dst_path / old_name).rename(dst_path / new_name)

    # Build weight map for model.safetensors.index.json
    for key, shard_num in all_renamed.items():
        shard_file = f"model-{shard_num:05d}-of-{total_shards:05d}.safetensors"
        weight_map[key] = shard_file

    index = {"metadata": {"total_size": sum(
        os.path.getsize(dst_path / f"model-{i:05d}-of-{total_shards:05d}.safetensors")
        for i in range(1, total_shards + 1)
    )}, "weight_map": weight_map}
    with open(dst_path / "model.safetensors.index.json", "w") as f:
        json.dump(index, f, indent=2)
    print(f"  model.safetensors.index.json: {len(weight_map)} entries")

    # 3. Copy tokenizer and other files
    for fname in [
        "tokenizer.json", "tokenizer_config.json",
        "special_tokens_map.json", "generation_config.json",
        "vocab.json", "merges.txt",
    ]:
        src_file = src_path / fname
        if src_file.exists():
            shutil.copy2(src_file, dst_path / fname)
            print(f"  Copied {fname}")

    # 4. Generate quantize_config.json for GPTQModel compatibility
    # GPTQModel looks for quantize_config.json (not quantization_config.json)
    qcfg_data = None
    for qcfg_name in ["quantize_config.json", "quantization_config.json"]:
        qcfg_file = src_path / qcfg_name
        if qcfg_file.exists():
            with open(qcfg_file) as f:
                qcfg_data = json.load(f)
            break
    if qcfg_data is None and "quantization_config" in config:
        qcfg_data = dict(config["quantization_config"])

    if qcfg_data:
        _update_quantization_config(qcfg_data)
        _ensure_quantization_meta(qcfg_data)
        with open(dst_path / "quantize_config.json", "w") as f:
            json.dump(qcfg_data, f, indent=2, ensure_ascii=False)
        print(f"  quantize_config.json: generated with updated paths")

    # 5. Ensure chat_template is present in tokenizer_config.json
    tok_cfg_path = dst_path / "tokenizer_config.json"
    if tok_cfg_path.exists():
        with open(tok_cfg_path) as f:
            tok_cfg = json.load(f)
        if "chat_template" not in tok_cfg:
            base_dir = Path(base_model_dir) if base_model_dir else None
            chat_template = None
            if base_dir and (base_dir / "tokenizer_config.json").exists():
                with open(base_dir / "tokenizer_config.json") as f:
                    base_tok = json.load(f)
                chat_template = base_tok.get("chat_template")
            if chat_template:
                tok_cfg["chat_template"] = chat_template
                with open(tok_cfg_path, "w") as f:
                    json.dump(tok_cfg, f, indent=2, ensure_ascii=False)
                print(f"  tokenizer_config.json: copied chat_template from base model")
            else:
                print(f"  WARNING: tokenizer_config.json missing chat_template. "
                      f"Use --base-model to specify FP16 base model path.")

    print(f"  Done! Output: {dst_path}")


def main():
    parser = argparse.ArgumentParser(description="Convert VLM quantized model to text-only format")
    parser.add_argument("--src", required=True, help="Source VLM model directory")
    parser.add_argument("--dst", required=True, help="Destination text-only model directory")
    parser.add_argument("--keep-visual", action="store_true", help="Keep visual weights")
    parser.add_argument("--base-model", default=None,
                        help="FP16 base model directory (for copying chat_template if missing)")
    args = parser.parse_args()

    print(f"Converting {args.src} -> {args.dst}")
    convert_model(args.src, args.dst, args.keep_visual, args.base_model)


if __name__ == "__main__":
    main()
