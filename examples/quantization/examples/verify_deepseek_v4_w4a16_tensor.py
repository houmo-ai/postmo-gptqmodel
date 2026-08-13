#!/usr/bin/env python3
"""Compare one baicai1145 W4A16 expert tensor with native DeepSeek-V4 FP4.

This verifier intentionally needs only one converted safetensor shard, not the
complete 166-GB checkpoint.  It dequantizes the native E2M1 tensor, unpacks the
compressed-tensors I32 INT4 codes, measures the released reconstruction error,
and independently reproduces the published six-rate group-32 conversion.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
from safetensors import safe_open


FP4_TABLE = (
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
    0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
)
PUBLISHED_RATES = (6.0 / 7.0, 0.9, 0.95, 1.0, 1.05, 1.1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-model", required=True)
    parser.add_argument("--converted-shard", required=True)
    parser.add_argument(
        "--tensor",
        required=True,
        help="Native weight key, for example layers.29.ffn.experts.126.w1.weight",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def unpack_native(weight: torch.Tensor, scale: torch.Tensor, device: torch.device) -> torch.Tensor:
    packed = weight.to(device).view(torch.uint8)
    codes = torch.stack((packed & 0x0F, (packed >> 4) & 0x0F), dim=-1)
    codes = codes.reshape(weight.shape[0], weight.shape[1] * 2).long()
    table = torch.tensor(FP4_TABLE, dtype=torch.float32, device=device)
    return table[codes] * scale.to(device).float().repeat_interleave(32, dim=1)


def unpack_i32_int4(packed: torch.Tensor, shape: tuple[int, int], device: torch.device) -> torch.Tensor:
    value = packed.to(device)
    shifts = torch.arange(0, 32, 4, dtype=torch.int32, device=device)
    return (((value[..., None] >> shifts) & 0x0F) - 8).to(torch.int8).reshape(shape)


def reproduce(reference: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    groups = reference.reshape(reference.shape[0], -1, 32)
    amax = groups.abs().amax(dim=-1, keepdim=True)
    rates = torch.tensor(PUBLISHED_RATES, dtype=torch.float32, device=reference.device)
    scales = amax.unsqueeze(0) * rates[:, None, None, None] / 7.0
    codes = torch.round(groups.unsqueeze(0) / scales).clamp_(-8, 7)
    errors = (groups.unsqueeze(0) - codes * scales).square().sum(dim=-1)
    best = errors.argmin(dim=0)
    index = best[None, ..., None]
    selected_scale = scales.gather(0, index).squeeze(0)
    selected_code = torch.round(groups / selected_scale).clamp_(-8, 7)
    return selected_code.to(torch.int8).reshape_as(reference), selected_scale.squeeze(-1).to(torch.bfloat16)


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    native_root = Path(args.native_model)
    with (native_root / "model.safetensors.index.json").open(encoding="utf-8") as handle:
        native_index = json.load(handle)["weight_map"]
    native_scale_key = args.tensor[:-len("weight")] + "scale"
    with safe_open(native_root / native_index[args.tensor], framework="pt", device="cpu") as handle:
        native_weight = handle.get_tensor(args.tensor)
        native_scale = handle.get_tensor(native_scale_key)
    reference = unpack_native(native_weight, native_scale, device)

    converted_prefix = args.tensor[:-len("weight")]
    with safe_open(args.converted_shard, framework="pt", device="cpu") as handle:
        packed = handle.get_tensor(converted_prefix + "weight_packed")
        released_scale = handle.get_tensor(converted_prefix + "weight_scale")
        released_shape = tuple(int(x) for x in handle.get_tensor(converted_prefix + "weight_shape"))
    released_code = unpack_i32_int4(packed, released_shape, device)
    released_scale = released_scale.to(device)
    reproduced_code, reproduced_scale = reproduce(reference)

    released = released_code.float() * released_scale.float().repeat_interleave(32, dim=1)
    reproduced = reproduced_code.float() * reproduced_scale.float().repeat_interleave(32, dim=1)
    signal_sq = reference.square().sum().item()
    released_error_sq = (reference - released).square().sum().item()
    reproduced_error_sq = (reference - reproduced).square().sum().item()
    payload = {
        "tensor": args.tensor,
        "shape": list(reference.shape),
        "released_code_matches": int((released_code == reproduced_code).sum().item()),
        "released_code_elements": reference.numel(),
        "released_code_match_fraction": (released_code == reproduced_code).float().mean().item(),
        "released_scale_matches": int((released_scale == reproduced_scale).sum().item()),
        "released_scale_elements": released_scale.numel(),
        "released_scale_match_fraction": (released_scale == reproduced_scale).float().mean().item(),
        "released_relative_mse": released_error_sq / signal_sq,
        "released_snr_db": -10.0 * math.log10(released_error_sq / signal_sq),
        "reproduced_relative_mse": reproduced_error_sq / signal_sq,
        "reproduced_snr_db": -10.0 * math.log10(reproduced_error_sq / signal_sq),
        "published_rates": list(PUBLISHED_RATES),
    }
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
