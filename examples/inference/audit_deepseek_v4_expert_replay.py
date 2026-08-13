#!/usr/bin/env python3
"""Replay one DeepSeek-V4 routed expert across native and GPTQ paths.

The runtime dump is produced by ``deepseek_v4_quant_demo.py``.  This script
then separates three possible error sources:

* native FP4 -> GPTQ INT4 weight reconstruction error;
* serialized GPTQ -> Triton dequant/matmul error; and
* TritonV2's output-dtype contract when its input is FP32 under autocast.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors import safe_open


FP4_TABLE = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
     0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
    dtype=torch.float32,
)
PROJECTIONS = {
    "gate_proj": "w1",
    "up_proj": "w3",
    "down_proj": "w2",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-model", type=Path, required=True)
    parser.add_argument("--packed-model", type=Path, required=True)
    parser.add_argument("--runtime-dump", type=Path, required=True)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--expert", type=int, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--swiglu-limit", type=float, default=10.0)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_index(model: Path) -> dict[str, str]:
    return json.loads((model / "model.safetensors.index.json").read_text())["weight_map"]


def get_tensors(model: Path, names: list[str]) -> dict[str, torch.Tensor]:
    index = load_index(model)
    by_shard: dict[str, list[str]] = {}
    for name in names:
        by_shard.setdefault(index[name], []).append(name)
    result = {}
    for shard, shard_names in by_shard.items():
        with safe_open(model / shard, framework="pt", device="cpu") as reader:
            result.update({name: reader.get_tensor(name) for name in shard_names})
    return result


def unpack_rows(packed: torch.Tensor, bits: int) -> torch.Tensor:
    shifts = torch.arange(0, 32, bits, dtype=torch.int64)
    values = (packed.to(torch.int64).unsqueeze(1) >> shifts.view(1, -1, 1))
    return (values & ((1 << bits) - 1)).reshape(packed.shape[0] * len(shifts), packed.shape[1])


def unpack_cols(packed: torch.Tensor, bits: int) -> torch.Tensor:
    shifts = torch.arange(0, 32, bits, dtype=torch.int64)
    values = packed.to(torch.int64).unsqueeze(-1) >> shifts
    return (values & ((1 << bits) - 1)).reshape(packed.shape[0], packed.shape[1] * len(shifts))


def dequant_packed(buffers: dict[str, torch.Tensor], prefix: str) -> torch.Tensor:
    qweight = buffers[prefix + ".qweight"]
    qzeros = buffers[prefix + ".qzeros"]
    scales = buffers[prefix + ".scales"].float()
    g_idx = buffers[prefix + ".g_idx"].long()
    # Serialized GPTQ-v1 stores zero - 1.  GPTQModel restores one to every
    # packed field before feeding its V2 kernels.
    zeros = (unpack_cols(qzeros, 4) + 1) & 0xF
    values = unpack_rows(qweight, 4)
    return ((values.float() - zeros[g_idx].float()) * scales[g_idx]).t().contiguous()


def runtime_qzeros(qzeros: torch.Tensor) -> torch.Tensor:
    offset = torch.tensor(0x11111111, dtype=torch.int32)
    return qzeros + offset


def dequant_native(packed: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    packed_u8 = packed.view(torch.uint8)
    codes = torch.stack((packed_u8 & 0x0F, (packed_u8 >> 4) & 0x0F), dim=-1)
    codes = codes.reshape(packed.shape[0], packed.shape[1] * 2).long()
    return FP4_TABLE[codes] * scales.float().repeat_interleave(32, dim=1)


def summary(value: torch.Tensor) -> dict:
    original_dtype = str(value.dtype)
    value = value.detach().float().cpu()
    return {
        "shape": list(value.shape),
        "dtype": original_dtype,
        "mean": value.mean().item(),
        "rms": value.square().mean().sqrt().item(),
        "abs_max": value.abs().max().item(),
    }


def comparison(actual: torch.Tensor, reference: torch.Tensor) -> dict:
    actual = actual.detach().float().cpu()
    reference = reference.detach().float().cpu()
    error = actual - reference
    signal_sq = reference.square().sum().item()
    error_sq = error.square().sum().item()
    relative_rmse = math.sqrt(error_sq / signal_sq) if signal_sq else 0.0
    cosine = F.cosine_similarity(actual.flatten(), reference.flatten(), dim=0).item()
    return {
        "relative_rmse": relative_rmse,
        "cosine": cosine,
        "max_abs_error": error.abs().max().item(),
        "mean_error": error.mean().item(),
    }


def apply_gate(gate: torch.Tensor, up: torch.Tensor, limit: float) -> torch.Tensor:
    return F.silu(gate.clamp(max=limit)) * up.clamp(min=-limit, max=limit)


def main() -> None:
    args = parse_args()
    packed_prefix = f"model.layers.{args.layer}.mlp.experts.{args.expert}"
    native_prefix = f"layers.{args.layer}.ffn.experts.{args.expert}"
    packed_names = [
        f"{packed_prefix}.{projection}.{suffix}"
        for projection in PROJECTIONS
        for suffix in ("qweight", "qzeros", "scales", "g_idx")
    ]
    native_names = [
        f"{native_prefix}.{projection}.{suffix}"
        for projection in PROJECTIONS.values()
        for suffix in ("weight", "scale")
    ]
    packed_buffers = get_tensors(args.packed_model, packed_names)
    native_buffers = get_tensors(args.native_model, native_names)

    packed_weights = {}
    native_weights = {}
    weight_report = {}
    for projection, native_projection in PROJECTIONS.items():
        qprefix = f"{packed_prefix}.{projection}"
        nprefix = f"{native_prefix}.{native_projection}"
        packed_weights[projection] = dequant_packed(packed_buffers, qprefix)
        native_weights[projection] = dequant_native(
            native_buffers[nprefix + ".weight"],
            native_buffers[nprefix + ".scale"],
        )
        weight_report[projection] = {
            "packed": summary(packed_weights[projection]),
            "native": summary(native_weights[projection]),
            "packed_vs_native": comparison(packed_weights[projection], native_weights[projection]),
        }

    dump = torch.load(args.runtime_dump, map_location="cpu", weights_only=False)
    runtime_input = dump["inputs"][0]
    captured_gate = dump["output"]
    device = torch.device(args.device)
    x_fp32 = runtime_input.to(device=device, dtype=torch.float32)

    def dense_expert(weights: dict[str, torch.Tensor], dtype: torch.dtype) -> dict[str, torch.Tensor]:
        gpu_weights = {name: weight.to(device=device, dtype=dtype) for name, weight in weights.items()}
        with torch.autocast(device_type="cuda", dtype=dtype):
            gate = F.linear(x_fp32, gpu_weights["gate_proj"])
            up = F.linear(x_fp32, gpu_weights["up_proj"])
            gated = apply_gate(gate, up, args.swiglu_limit)
            down = F.linear(gated, gpu_weights["down_proj"])
        return {"gate": gate, "up": up, "gated": gated, "down": down}

    # C: materialized GPTQ reconstruction through ordinary FP16 Linear.
    dense_packed = dense_expert(packed_weights, torch.float16)
    # Native checkpoint replay uses its exact FP4 dequantization rounded to BF16,
    # matching the preparation path used as AutoRound/GPTQ input.
    dense_native = dense_expert(native_weights, torch.bfloat16)

    from gptqmodel.nn_modules.triton_utils.dequant import QuantLinearFunction

    def triton_linear(projection: str, x: torch.Tensor, preserve_input_dtype: bool) -> torch.Tensor:
        prefix = f"{packed_prefix}.{projection}"
        qweight = packed_buffers[prefix + ".qweight"].to(device)
        qzeros = runtime_qzeros(packed_buffers[prefix + ".qzeros"]).to(device)
        scales = packed_buffers[prefix + ".scales"].to(device)
        g_idx = packed_buffers[prefix + ".g_idx"].to(device)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            output = QuantLinearFunction.apply(x, qweight, scales, qzeros, g_idx, 4, 32, 15)
        return output.to(x.dtype) if preserve_input_dtype else output

    def triton_expert(preserve_input_dtype: bool, first_input: torch.Tensor) -> dict[str, torch.Tensor]:
        gate = triton_linear("gate_proj", first_input, preserve_input_dtype)
        up = triton_linear("up_proj", first_input, preserve_input_dtype)
        gated = apply_gate(gate, up, args.swiglu_limit)
        down = triton_linear("down_proj", gated, preserve_input_dtype)
        return {"gate": gate, "up": up, "gated": gated, "down": down}

    # A: current TritonV2 behavior (raw FP16 GEMM result promoted to FP32).
    triton_current = triton_expert(True, x_fp32)
    # B: explicit FP16 input avoids the promotion even with today's wrapper.
    triton_half_input = triton_expert(True, x_fp32.half())
    # D: proposed contract -- return the actual matmul dtype, not x.dtype.
    triton_actual_dtype = triton_expert(False, x_fp32)

    paths = {
        "runtime_captured_gate": captured_gate,
        "native_bf16": dense_native,
        "packed_dense_fp16": dense_packed,
        "triton_current_fp32_input": triton_current,
        "triton_current_fp16_input": triton_half_input,
        "triton_return_actual_dtype": triton_actual_dtype,
    }
    activation_report = {}
    for path, values in paths.items():
        if isinstance(values, torch.Tensor):
            activation_report[path] = {"gate": summary(values)}
            continue
        activation_report[path] = {name: summary(value) for name, value in values.items()}

    comparisons = {
        "captured_gate_vs_triton_current": comparison(captured_gate, triton_current["gate"]),
        "triton_current_vs_return_actual_gate": comparison(triton_current["gate"], triton_actual_dtype["gate"]),
        "triton_current_vs_return_actual_gated": comparison(triton_current["gated"], triton_actual_dtype["gated"]),
        "triton_current_vs_return_actual_down": comparison(triton_current["down"], triton_actual_dtype["down"]),
        "packed_dense_vs_triton_actual_down": comparison(dense_packed["down"], triton_actual_dtype["down"]),
        "packed_dense_vs_native_down": comparison(dense_packed["down"], dense_native["down"]),
        "triton_actual_vs_native_down": comparison(triton_actual_dtype["down"], dense_native["down"]),
    }
    report = {
        "layer": args.layer,
        "expert": args.expert,
        "runtime_module": dump.get("module"),
        "runtime_input": summary(runtime_input),
        "weight_report": weight_report,
        "activation_report": activation_report,
        "comparisons": comparisons,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
