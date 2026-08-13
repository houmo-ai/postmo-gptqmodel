#!/usr/bin/env python3
"""Repair a DeepSeek-V4 mixed W8/W4 GPTQ checkpoint packed without clamp.

The affected Python packers rounded the source weight to an integer and then
bit-packed it without saturating to the module's bit range.  AutoRound's fake
quantization *does* saturate, so a clipped W4 value of 16 was silently stored
as 0.  This utility reuses the learned scales and group assignments, recomputes
the integers from the BF16 calibration checkpoint, clamps them, and writes a
new artifact without modifying the input checkpoint.

It also repairs checkpoints written by the former mixed-precision qzero bug,
where every module used the global W8 GPTQ-v1 offset, and restores the 406
official DeepSeek-V4 FP32 control tensors from the native checkpoint.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import re
import shutil
import time
from contextlib import ExitStack
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file


_MASK32 = 0xFFFF_FFFF
_ATTENTION_SOURCE_NAMES = {
    "q_a_proj": "wq_a",
    "q_b_proj": "wq_b",
    "kv_proj": "wkv",
    "o_b_proj": "wo_b",
}
_MLP_SOURCE_NAMES = {
    "gate_proj": "w1",
    "up_proj": "w3",
    "down_proj": "w2",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quant-model", type=Path, required=True)
    parser.add_argument("--bf16-model", type=Path, required=True)
    parser.add_argument("--native-model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--source-qzero-offset",
        choices=("global", "module"),
        default="global",
        help=(
            "Offset used when the input artifact was converted from GPTQ-v2 "
            "to v1. The affected mixed W8/W4 artifact used 'global'."
        ),
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Keep already completed output shards after an interrupted run.",
    )
    return parser.parse_args()


def load_index(model_dir: Path) -> dict:
    path = model_dir / "model.safetensors.index.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def packed_one_offset(bits: int) -> int:
    if bits not in (2, 4, 8):
        raise ValueError(f"This repair supports 2/4/8-bit modules, got {bits}")
    return sum(1 << (shift * bits) for shift in range(32 // bits))


def unsigned32(value: torch.Tensor) -> torch.Tensor:
    return value.to(torch.int64) & _MASK32


def signed32(value: torch.Tensor) -> torch.Tensor:
    value = value & _MASK32
    return torch.where(value >= 2**31, value - 2**32, value).to(torch.int32)


def unpack_qzeros_v2(
    qzeros_v1: torch.Tensor,
    *,
    bits: int,
    source_offset_bits: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return logical zeros and the recovered GPTQ-v2 packed words."""

    packed_v2 = (
        unsigned32(qzeros_v1) + packed_one_offset(source_offset_bits)
    ) & _MASK32
    mask = (1 << bits) - 1
    logical = torch.stack(
        [
            ((packed_v2 >> (slot * bits)) & mask).to(torch.int32)
            for slot in range(32 // bits)
        ],
        dim=-1,
    ).reshape(qzeros_v1.shape[0], -1)
    return logical, packed_v2


def convert_qzeros_v2_to_standard_v1(
    packed_v2: torch.Tensor,
    *,
    bits: int,
) -> torch.Tensor:
    return signed32(packed_v2 - packed_one_offset(bits))


def module_bits(module_name: str) -> int:
    return 4 if ".mlp.experts." in module_name else 8


def quant_module_source_key(module_name: str) -> str:
    attention = re.fullmatch(
        r"model\.layers\.(\d+)\.self_attn\."
        r"(q_a_proj|q_b_proj|kv_proj|o_b_proj)",
        module_name,
    )
    if attention:
        layer, projection = attention.groups()
        return f"layers.{layer}.attn.{_ATTENTION_SOURCE_NAMES[projection]}.weight"

    expert = re.fullmatch(
        r"model\.layers\.(\d+)\.mlp\.experts\.(\d+)\."
        r"(gate_proj|up_proj|down_proj)",
        module_name,
    )
    if expert:
        layer, expert_index, projection = expert.groups()
        return (
            f"layers.{layer}.ffn.experts.{expert_index}."
            f"{_MLP_SOURCE_NAMES[projection]}.weight"
        )

    shared = re.fullmatch(
        r"model\.layers\.(\d+)\.mlp\.shared_experts\."
        r"(gate_proj|up_proj|down_proj)",
        module_name,
    )
    if shared:
        layer, projection = shared.groups()
        return (
            f"layers.{layer}.ffn.shared_experts."
            f"{_MLP_SOURCE_NAMES[projection]}.weight"
        )

    raise KeyError(f"Unsupported quantized DeepSeek-V4 module: {module_name}")


def native_fp32_source_key(runtime_key: str) -> str | None:
    match = re.fullmatch(
        r"model\.layers\.(\d+)\.(attn_hc|ffn_hc)\.(fn|base|scale)",
        runtime_key,
    )
    if match:
        layer, site, parameter = match.groups()
        site = "attn" if site == "attn_hc" else "ffn"
        return f"layers.{layer}.hc_{site}_{parameter}"

    match = re.fullmatch(
        r"model\.layers\.(\d+)\.self_attn\.sinks",
        runtime_key,
    )
    if match:
        return f"layers.{match.group(1)}.attn.attn_sink"

    match = re.fullmatch(
        r"model\.layers\.(\d+)\.self_attn\.compressor\.position_bias",
        runtime_key,
    )
    if match:
        return f"layers.{match.group(1)}.attn.compressor.ape"

    match = re.fullmatch(
        r"model\.layers\.(\d+)\.self_attn\.compressor\.indexer\.position_bias",
        runtime_key,
    )
    if match:
        return f"layers.{match.group(1)}.attn.indexer.compressor.ape"

    match = re.fullmatch(
        r"model\.layers\.(\d+)\.mlp\.gate\.e_score_correction_bias",
        runtime_key,
    )
    if match:
        return f"layers.{match.group(1)}.ffn.gate.bias"

    return {
        "model.hc_head.hc_fn": "hc_head_fn",
        "model.hc_head.hc_base": "hc_head_base",
        "model.hc_head.hc_scale": "hc_head_scale",
    }.get(runtime_key)


@torch.inference_mode()
def repack_weight(
    source_weight: torch.Tensor,
    scales: torch.Tensor,
    zeros: torch.Tensor,
    g_idx: torch.Tensor,
    *,
    bits: int,
    expected_shape: tuple[int, ...],
    device: torch.device,
) -> tuple[torch.Tensor, int, int]:
    out_features, in_features = source_weight.shape
    pack_factor = 32 // bits
    if in_features % pack_factor:
        raise ValueError(
            f"Input dimension {in_features} is not divisible by pack factor {pack_factor}"
        )
    if tuple(scales.shape) != (int(g_idx.max()) + 1, out_features):
        raise ValueError(
            f"Scale shape {tuple(scales.shape)} does not match source "
            f"{tuple(source_weight.shape)} and g_idx max {int(g_idx.max())}"
        )

    weight_device = source_weight.to(device=device, dtype=torch.float32)
    scales_device = scales.to(device=device, dtype=torch.float32)
    zeros_device = zeros.to(device=device, dtype=torch.float32)
    groups_device = g_idx.to(device=device, dtype=torch.int64)
    if bool((groups_device < 0).any()):
        groups_device = groups_device.remainder(scales.shape[0])

    scale_rows = scales_device.index_select(0, groups_device)
    zero_rows = zeros_device.index_select(0, groups_device)
    rounded = torch.round(weight_device.transpose(0, 1) / scale_rows + zero_rows)
    maximum = (1 << bits) - 1
    clipped_low = int((rounded < 0).sum().item())
    clipped_high = int((rounded > maximum).sum().item())
    integers = rounded.clamp_(0, maximum).to(torch.int32)

    packed = torch.zeros(
        (in_features // pack_factor, out_features),
        dtype=torch.int32,
        device=device,
    )
    for slot in range(pack_factor):
        packed.bitwise_or_(integers[slot::pack_factor] << (slot * bits))
    packed_cpu = packed.cpu()
    if tuple(packed_cpu.shape) != expected_shape:
        raise ValueError(
            f"Packed shape {tuple(packed_cpu.shape)} != checkpoint shape {expected_shape}"
        )
    return packed_cpu, clipped_low, clipped_high


def copy_auxiliary_files(source: Path, output: Path) -> None:
    for entry in source.iterdir():
        if entry.name == "model.safetensors.index.json" or entry.suffix == ".safetensors":
            continue
        destination = output / entry.name
        if entry.is_dir():
            shutil.copytree(entry, destination, dirs_exist_ok=True)
        else:
            shutil.copy2(entry, destination)


def normalize_gptq_storage_contract(output: Path) -> bool:
    """Advertise AutoRound artifacts through their actual GPTQ disk format.

    Legacy GPTQModel writers used ``quant_method=auto_round`` even though the
    serialized tensors are the standard GPTQ qweight/qzeros/scales/g_idx
    contract.  Runtime dispatchers such as vLLM key off this field.  Keep the
    AutoRound algorithm provenance under ``meta.auto_round`` and normalize only
    the storage discriminator in both copied config files.
    """

    changed = False
    for filename in ("config.json", "quantize_config.json"):
        path = output / filename
        if not path.is_file():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        quantization_config = (
            payload.get("quantization_config", {})
            if filename == "config.json"
            else payload
        )
        if quantization_config.get("quant_method") != "auto_round":
            continue
        quantization_config["quant_method"] = "gptq"
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        changed = True
    return changed


def completed_shard_is_valid(path: Path, expected_keys: set[str]) -> bool:
    if not path.is_file():
        return False
    try:
        with safe_open(path, framework="pt", device="cpu") as reader:
            return set(reader.keys()) == expected_keys
    except Exception:
        return False


def validate_completed_output_shards(
    output: Path,
    shard_keys: dict[str, set[str]],
) -> None:
    invalid_shards = [
        shard_name
        for shard_name, expected_keys in shard_keys.items()
        if not completed_shard_is_valid(output / shard_name, expected_keys)
    ]
    if invalid_shards:
        preview = ", ".join(invalid_shards[:5])
        suffix = "" if len(invalid_shards) <= 5 else ", ..."
        raise RuntimeError(
            f"Output validation failed for {len(invalid_shards)} shard(s): "
            f"{preview}{suffix}"
        )


def main() -> None:
    args = parse_args()
    quant_model = args.quant_model.expanduser().resolve()
    bf16_model = args.bf16_model.expanduser().resolve()
    native_model = args.native_model.expanduser().resolve()
    output = args.output.expanduser().resolve()
    for model_dir in (quant_model, bf16_model, native_model):
        if not model_dir.is_dir():
            raise FileNotFoundError(model_dir)
    if output.exists() and not args.resume:
        raise FileExistsError(
            f"Refusing to overwrite {output}; pass --resume only for this repair's partial output"
        )
    output.mkdir(parents=True, exist_ok=True)

    quant_index = load_index(quant_model)
    bf16_index = load_index(bf16_model)
    native_index = load_index(native_model)
    quant_weight_map = quant_index["weight_map"]
    bf16_weight_map = bf16_index["weight_map"]
    native_weight_map = native_index["weight_map"]
    config = json.loads((quant_model / "config.json").read_text(encoding="utf-8"))
    global_bits = int(config["quantization_config"]["bits"])

    quant_modules = sorted(
        key[: -len(".qweight")]
        for key in quant_weight_map
        if key.endswith(".qweight")
    )
    if len(quant_modules) != 33_325:
        raise ValueError(
            f"Expected 33,325 DeepSeek-V4 quantized modules, found {len(quant_modules)}"
        )
    for module_name in quant_modules:
        source_key = quant_module_source_key(module_name)
        if source_key not in bf16_weight_map:
            raise KeyError(f"Missing BF16 source {source_key} for {module_name}")

    fp32_pairs = {
        runtime_key: source_key
        for runtime_key in quant_weight_map
        if (source_key := native_fp32_source_key(runtime_key)) is not None
    }
    missing_fp32 = {
        runtime_key: source_key
        for runtime_key, source_key in fp32_pairs.items()
        if source_key not in native_weight_map
    }
    if missing_fp32:
        raise KeyError(f"Missing native FP32 tensors: {missing_fp32}")
    if len(fp32_pairs) != 406:
        raise ValueError(
            f"Expected 406 official FP32 control tensors, found {len(fp32_pairs)}"
        )

    shard_names = list(dict.fromkeys(quant_weight_map.values()))
    shard_keys = {
        shard: {key for key, filename in quant_weight_map.items() if filename == shard}
        for shard in shard_names
    }
    copy_auxiliary_files(quant_model, output)
    storage_contract_normalized = normalize_gptq_storage_contract(output)

    started = time.perf_counter()
    repaired_modules = 0
    restored_fp32 = 0
    clipped_low_total = 0
    clipped_high_total = 0
    processed_shards = 0
    reused_shards = 0
    source_offset_bits_by_module = {
        module_name: (
            global_bits if args.source_qzero_offset == "global" else module_bits(module_name)
        )
        for module_name in quant_modules
    }
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)

    for shard_index, shard_name in enumerate(shard_names, start=1):
        destination = output / shard_name
        expected_keys = shard_keys[shard_name]
        if args.resume and completed_shard_is_valid(destination, expected_keys):
            reused_shards += 1
            print(
                f"[{shard_index:02d}/{len(shard_names):02d}] resume: {shard_name}",
                flush=True,
            )
            continue

        shard_started = time.perf_counter()
        with ExitStack() as stack:
            quant_reader = stack.enter_context(
                safe_open(
                    quant_model / shard_name,
                    framework="pt",
                    device="cpu",
                )
            )
            metadata = quant_reader.metadata()
            tensors = {
                key: quant_reader.get_tensor(key)
                for key in quant_reader.keys()
            }

            quant_readers = {shard_name: quant_reader}
            bf16_readers = {}
            native_readers = {}

            def quant_tensor(key: str) -> torch.Tensor:
                filename = quant_weight_map[key]
                if filename not in quant_readers:
                    quant_readers[filename] = stack.enter_context(
                        safe_open(
                            quant_model / filename,
                            framework="pt",
                            device="cpu",
                        )
                    )
                return quant_readers[filename].get_tensor(key)

            def bf16_tensor(key: str) -> torch.Tensor:
                filename = bf16_weight_map[key]
                if filename not in bf16_readers:
                    bf16_readers[filename] = stack.enter_context(
                        safe_open(
                            bf16_model / filename,
                            framework="pt",
                            device="cpu",
                        )
                    )
                return bf16_readers[filename].get_tensor(key)

            def native_tensor(key: str) -> torch.Tensor:
                filename = native_weight_map[key]
                if filename not in native_readers:
                    native_readers[filename] = stack.enter_context(
                        safe_open(
                            native_model / filename,
                            framework="pt",
                            device="cpu",
                        )
                    )
                return native_readers[filename].get_tensor(key)

            qweight_keys = sorted(
                key for key in tensors if key.endswith(".qweight")
            )
            shard_repaired = 0
            shard_low = 0
            shard_high = 0
            for qweight_key in qweight_keys:
                module_name = qweight_key[: -len(".qweight")]
                bits = module_bits(module_name)
                source_key = quant_module_source_key(module_name)
                scales = quant_tensor(module_name + ".scales")
                g_idx = quant_tensor(module_name + ".g_idx")
                qzeros_key = module_name + ".qzeros"
                logical_zeros, packed_v2 = unpack_qzeros_v2(
                    quant_tensor(qzeros_key),
                    bits=bits,
                    source_offset_bits=source_offset_bits_by_module[module_name],
                )
                expected_zero = 1 << (bits - 1)
                if not bool((logical_zeros == expected_zero).all()):
                    minimum = int(logical_zeros.min())
                    maximum = int(logical_zeros.max())
                    raise ValueError(
                        f"{module_name}: symmetric zero must be {expected_zero}, "
                        f"observed [{minimum}, {maximum}]; wrong source qzero offset?"
                    )

                packed_weight, clipped_low, clipped_high = repack_weight(
                    bf16_tensor(source_key),
                    scales,
                    logical_zeros,
                    g_idx,
                    bits=bits,
                    expected_shape=tuple(tensors[qweight_key].shape),
                    device=device,
                )
                tensors[qweight_key] = packed_weight
                shard_repaired += 1
                shard_low += clipped_low
                shard_high += clipped_high

            # qweight and its companion tensors are allowed to live in
            # different safetensor shards.  Rewrite every qzero in its own
            # physical shard instead of assuming co-location with qweight.
            for qzeros_key in sorted(
                key for key in tensors if key.endswith(".qzeros")
            ):
                module_name = qzeros_key[: -len(".qzeros")]
                bits = module_bits(module_name)
                logical_zeros, packed_v2 = unpack_qzeros_v2(
                    tensors[qzeros_key],
                    bits=bits,
                    source_offset_bits=source_offset_bits_by_module[module_name],
                )
                expected_zero = 1 << (bits - 1)
                if not bool((logical_zeros == expected_zero).all()):
                    minimum = int(logical_zeros.min())
                    maximum = int(logical_zeros.max())
                    raise ValueError(
                        f"{module_name}: symmetric zero must be {expected_zero}, "
                        f"observed [{minimum}, {maximum}]; wrong source qzero offset?"
                    )
                tensors[qzeros_key] = convert_qzeros_v2_to_standard_v1(
                    packed_v2,
                    bits=bits,
                )

            shard_restored = 0
            for runtime_key in sorted(expected_keys & fp32_pairs.keys()):
                source = native_tensor(fp32_pairs[runtime_key])
                if source.dtype != torch.float32:
                    raise TypeError(
                        f"Native control tensor {fp32_pairs[runtime_key]} is {source.dtype}, not FP32"
                    )
                if tuple(source.shape) != tuple(tensors[runtime_key].shape):
                    raise ValueError(
                        f"Shape mismatch for {runtime_key}: {tuple(tensors[runtime_key].shape)} "
                        f"vs native {tuple(source.shape)}"
                    )
                tensors[runtime_key] = source.contiguous()
                shard_restored += 1

            temporary = output / f".{shard_name}.tmp"
            save_file(tensors, temporary, metadata=metadata)
            os.replace(temporary, destination)

        repaired_modules += shard_repaired
        restored_fp32 += shard_restored
        clipped_low_total += shard_low
        clipped_high_total += shard_high
        processed_shards += 1
        elapsed = time.perf_counter() - shard_started
        total_elapsed = time.perf_counter() - started
        print(
            f"[{shard_index:02d}/{len(shard_names):02d}] {shard_name}: "
            f"modules={shard_repaired}, fp32={shard_restored}, "
            f"clipped=({shard_low},{shard_high}), "
            f"shard_s={elapsed:.1f}, total_s={total_elapsed:.1f}",
            flush=True,
        )
        del tensors
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if not args.resume:
        if repaired_modules != len(quant_modules):
            raise RuntimeError(
                f"Repacked {repaired_modules} modules, expected {len(quant_modules)}"
            )
        if restored_fp32 != len(fp32_pairs):
            raise RuntimeError(
                f"Restored {restored_fp32} FP32 tensors, expected {len(fp32_pairs)}"
            )

    validate_completed_output_shards(output, shard_keys)

    shutil.copy2(
        quant_model / "model.safetensors.index.json",
        output / "model.safetensors.index.json",
    )
    report = {
        "source_quant_model": str(quant_model),
        "source_bf16_model": str(bf16_model),
        "source_native_model": str(native_model),
        "quantized_modules": len(quant_modules),
        "official_fp32_control_tensors": len(fp32_pairs),
        "source_qzero_offset": args.source_qzero_offset,
        "global_bits": global_bits,
        "resume": args.resume,
        "completed_output_shards": len(shard_names),
        "processed_shards_this_run": processed_shards,
        "reused_shards_this_run": reused_shards,
        "repaired_modules_this_run": repaired_modules,
        "restored_fp32_control_tensors_this_run": restored_fp32,
        "storage_contract": {
            "calibration_method": "auto_round",
            "config_quant_method": "gptq",
            "legacy_config_normalized": storage_contract_normalized,
        },
        "clipped_low_this_run": clipped_low_total,
        "clipped_high_this_run": clipped_high_total,
        "elapsed_seconds_this_run": time.perf_counter() - started,
    }
    (output / "deepseek_v4_pack_repair.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
