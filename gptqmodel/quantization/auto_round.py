# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from .config import AutoRoundVersion, QuantizeConfig


def _effective_group_size(num_cols: int, group_size: int) -> int:
    if group_size == -1 or group_size <= 0 or group_size > num_cols:
        return num_cols
    return group_size


def _group_count(num_cols: int, group_size: int) -> int:
    effective = _effective_group_size(num_cols, group_size)
    return (num_cols + effective - 1) // effective


def _reshape_pad_tensor_by_group_size(
    tensor: torch.Tensor,
    group_size: int,
) -> Tuple[torch.Tensor, Tuple[int, ...], int, int]:
    original_shape = tuple(tensor.shape)
    if tensor.ndim != 2:
        raise ValueError("Auto-Round helpers currently expect 2D tensors.")

    rows, cols = tensor.shape
    effective = _effective_group_size(cols, group_size)
    if effective == cols:
        return tensor.reshape(-1, effective), original_shape, 0, effective

    remainder = cols % effective
    pad_len = (effective - remainder) % effective
    if pad_len:
        tensor = torch.nn.functional.pad(tensor, (0, pad_len))
    return tensor.reshape(-1, effective), original_shape, pad_len, effective


def _revert_tensor_by_pad(
    tensor: torch.Tensor,
    original_shape: Tuple[int, ...],
    pad_len: int,
) -> torch.Tensor:
    if pad_len:
        tensor = tensor.reshape(original_shape[0], -1)[:, :-pad_len]
    else:
        tensor = tensor.reshape(original_shape)
    return tensor.reshape(original_shape)


def _round_ste(tensor: torch.Tensor) -> torch.Tensor:
    rounded = torch.round(tensor)
    return tensor + (rounded - tensor).detach()


def _safe_reciprocal(tensor: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return torch.reciprocal(torch.where(torch.abs(tensor) < eps, torch.full_like(tensor, eps), tensor))


def _normalize_group_param(
    value,
    *,
    num_groups: int,
    rows: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        tensor = value.to(device=device, dtype=dtype)
    else:
        tensor = torch.tensor(value, device=device, dtype=dtype)

    if tensor.numel() == 1:
        return tensor.reshape(1).expand(num_groups)
    if tensor.numel() == num_groups:
        return tensor.reshape(num_groups)
    if tensor.ndim == 2 and tensor.shape == (rows, num_groups // rows):
        return tensor.reshape(num_groups)
    if tensor.ndim == 2 and tensor.shape == (num_groups, 1):
        return tensor.reshape(num_groups)
    raise ValueError(f"Unexpected Auto-Round group parameter shape: {tuple(tensor.shape)}")


def _normalize_round_values(
    value,
    *,
    rows: int,
    cols: int,
    group_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        tensor = value.to(device=device, dtype=dtype)
    else:
        tensor = torch.tensor(value, device=device, dtype=dtype)

    effective = _effective_group_size(cols, group_size)
    num_groups = rows * _group_count(cols, group_size)

    if tensor.numel() == 1:
        return tensor.reshape(1, 1).expand(num_groups, effective)

    if tensor.ndim == 2 and tensor.shape == (rows, cols):
        grouped, _, _, _ = _reshape_pad_tensor_by_group_size(tensor, group_size)
        return grouped

    if tensor.ndim == 2 and tensor.shape == (num_groups, effective):
        return tensor

    raise ValueError(f"Unexpected Auto-Round value shape: {tuple(tensor.shape)}")


def search_scales(data: torch.Tensor, bits: int, qw=None) -> torch.Tensor:
    nmax = 1 << (bits - 1)
    imax = torch.abs(data).argmax(dim=-1, keepdim=True)
    group_max = torch.take_along_dim(data, imax, dim=-1)

    iscales = -nmax * _safe_reciprocal(group_max)
    scales = _safe_reciprocal(iscales)

    q = torch.round(iscales * data).clamp(-nmax, nmax - 1)
    loss = ((scales * q - data).to(torch.float32)) ** 2
    if isinstance(qw, torch.Tensor):
        loss = loss * qw.to(loss.device, loss.dtype)
    elif qw is not None:
        loss = loss * float(qw)
    best_loss = torch.sum(loss, dim=-1)

    for step in range(-90, 91):
        if step == 0:
            continue
        iscales_tmp = -(nmax - 0.01 * step) * _safe_reciprocal(group_max)
        tmp_q = torch.round(iscales_tmp * data).clamp(-nmax, nmax - 1)
        tmp_scales = _safe_reciprocal(iscales_tmp)
        tmp_loss = ((tmp_scales * tmp_q - data).to(torch.float32)) ** 2
        if isinstance(qw, torch.Tensor):
            tmp_loss = tmp_loss * qw.to(tmp_loss.device, tmp_loss.dtype)
        elif qw is not None:
            tmp_loss = tmp_loss * float(qw)
        tmp_loss = torch.sum(tmp_loss, dim=-1)
        replace = tmp_loss < best_loss
        if replace.any():
            scales = torch.where(replace.unsqueeze(-1), tmp_scales, scales)
            best_loss = torch.where(replace, tmp_loss, best_loss)

    return scales.reshape(-1).to(torch.float32)


def quant_tensor_sym_gptq(
    tensor: torch.Tensor,
    bits: int = 4,
    group_size: int = -1,
    v=0,
    min_scale=1.0,
    max_scale=1.0,
    scale_dtype=torch.float32,
    tensor_min=None,
    tensor_max=None,
    q_scale_thresh: float = 1e-5,
    init_scale=None,
    **_: Dict,
):
    grouped, original_shape, pad_len, effective = _reshape_pad_tensor_by_group_size(tensor, group_size)
    rows = tensor.shape[0]
    num_groups = grouped.shape[0]
    maxq = 2**bits - 1

    if tensor_min is None or tensor_max is None:
        wmin_tmp = torch.clamp(grouped.min(dim=-1).values, max=0)
        wmax_tmp = torch.clamp(grouped.max(dim=-1).values, min=0)
    else:
        wmin_tmp = tensor_min.to(grouped.device, grouped.dtype).reshape(num_groups)
        wmax_tmp = tensor_max.to(grouped.device, grouped.dtype).reshape(num_groups)

    if init_scale is not None:
        scale = _normalize_group_param(
            init_scale,
            num_groups=num_groups,
            rows=rows,
            device=grouped.device,
            dtype=grouped.dtype,
        )
        scale = torch.abs(scale)
    else:
        _ = min_scale  # symmetric GPTQ-compatible path only uses max range.
        wmax_abs = torch.maximum((-wmin_tmp), wmax_tmp)
        scale = (2 * wmax_abs) / maxq

    max_scale_tensor = _normalize_group_param(
        max_scale,
        num_groups=num_groups,
        rows=rows,
        device=grouped.device,
        dtype=grouped.dtype,
    )
    scale = (scale * max_scale_tensor).to(scale_dtype)
    scale = torch.where(scale < 0, torch.clamp(scale, max=-q_scale_thresh), torch.clamp(scale, min=q_scale_thresh))
    scale = scale.unsqueeze(-1)

    zp = torch.full_like(scale, maxq / 2)
    value = _normalize_round_values(
        v,
        rows=rows,
        cols=tensor.shape[1],
        group_size=effective,
        device=grouped.device,
        dtype=grouped.dtype,
    )
    int_w = _round_ste(grouped / scale + value)
    q = torch.clamp(int_w + zp, 0, maxq)
    qdq = (scale * (q - zp)).to(grouped.dtype)
    qdq = _revert_tensor_by_pad(qdq, original_shape=original_shape, pad_len=pad_len)
    return qdq, scale.to(torch.float32), zp.to(torch.float32)


def quant_tensor_asym(
    tensor: torch.Tensor,
    bits: int = 4,
    group_size: int = -1,
    v=0,
    min_scale=1.0,
    max_scale=1.0,
    scale_dtype=torch.float32,
    tensor_min=None,
    tensor_max=None,
    q_scale_thresh: float = 1e-5,
    **_: Dict,
):
    grouped, original_shape, pad_len, effective = _reshape_pad_tensor_by_group_size(tensor, group_size)
    rows = tensor.shape[0]
    num_groups = grouped.shape[0]
    maxq = 2**bits - 1

    if tensor_min is None or tensor_max is None:
        wmin_tmp = torch.clamp(grouped.min(dim=-1).values, max=0)
        wmax_tmp = torch.clamp(grouped.max(dim=-1).values, min=0)
    else:
        wmin_tmp = tensor_min.to(grouped.device, grouped.dtype).reshape(num_groups)
        wmax_tmp = tensor_max.to(grouped.device, grouped.dtype).reshape(num_groups)

    min_scale_tensor = _normalize_group_param(
        min_scale,
        num_groups=num_groups,
        rows=rows,
        device=grouped.device,
        dtype=grouped.dtype,
    )
    max_scale_tensor = _normalize_group_param(
        max_scale,
        num_groups=num_groups,
        rows=rows,
        device=grouped.device,
        dtype=grouped.dtype,
    )
    wmin = wmin_tmp * min_scale_tensor
    wmax = wmax_tmp * max_scale_tensor

    scale = ((wmax - wmin) / maxq).to(scale_dtype)
    scale = torch.clamp(scale, min=q_scale_thresh).unsqueeze(-1)
    zp = _round_ste((-wmin / scale.reshape(-1))).unsqueeze(-1)

    value = _normalize_round_values(
        v,
        rows=rows,
        cols=tensor.shape[1],
        group_size=effective,
        device=grouped.device,
        dtype=grouped.dtype,
    )
    int_w = _round_ste(grouped / scale + value)
    q = torch.clamp(int_w + zp, 0, maxq)
    qdq = (scale * (q - zp)).to(grouped.dtype)
    qdq = _revert_tensor_by_pad(qdq, original_shape=original_shape, pad_len=pad_len)
    return qdq, scale.to(torch.float32), zp.to(torch.float32)


def build_input_hessian(inputs: Sequence[torch.Tensor]) -> torch.Tensor:
    if not inputs:
        raise ValueError("Auto-Round Hessian initialization requires at least one captured input tensor.")

    total = None
    for tensor in inputs:
        if tensor is None:
            continue
        if tensor.ndim == 1:
            flat = tensor.reshape(1, -1)
        else:
            flat = tensor.reshape(-1, tensor.shape[-1])
        contrib = (flat.to(torch.float32) ** 2).sum(dim=0)
        total = contrib if total is None else total + contrib

    if total is None:
        raise ValueError("Auto-Round Hessian initialization did not find any valid inputs.")
    return total


def build_round_g_idx(in_features: int, group_size: int, device: Optional[torch.device] = None) -> torch.Tensor:
    effective = _effective_group_size(in_features, group_size)
    g_idx = torch.tensor([index // effective for index in range(in_features)], dtype=torch.int32)
    if device is not None:
        g_idx = g_idx.to(device=device)
    return g_idx


def build_qparams_for_pack(
    *,
    qdq_weight: torch.Tensor,
    scale: torch.Tensor,
    zero: torch.Tensor,
    rows: int,
    in_features: int,
    group_size: int,
):
    groups = _group_count(in_features, group_size)
    q_scales = scale.reshape(rows, groups).to(torch.float32)
    q_zeros = torch.round(zero.reshape(rows, groups)).to(torch.float32)
    q_g_idx = build_round_g_idx(in_features=in_features, group_size=group_size)
    return q_scales, q_zeros, q_g_idx


def build_qparams_from_best_params(
    *,
    weight: torch.Tensor,
    bits: int,
    group_size: int,
    sym: bool,
    value,
    min_scale,
    max_scale,
    init_scale=None,
):
    rows, cols = weight.shape
    grouped_weight, _, _, effective = _reshape_pad_tensor_by_group_size(weight, group_size)
    num_groups = grouped_weight.shape[0]
    _ = effective

    value_tensor = _normalize_round_values(
        value,
        rows=rows,
        cols=cols,
        group_size=group_size,
        device=weight.device,
        dtype=weight.dtype,
    )
    min_scale_tensor = _normalize_group_param(
        min_scale,
        num_groups=num_groups,
        rows=rows,
        device=weight.device,
        dtype=weight.dtype,
    )
    max_scale_tensor = _normalize_group_param(
        max_scale,
        num_groups=num_groups,
        rows=rows,
        device=weight.device,
        dtype=weight.dtype,
    )

    if sym:
        qdq_weight, scale, zero = quant_tensor_sym_gptq(
            weight,
            bits=bits,
            group_size=group_size,
            v=value_tensor,
            min_scale=min_scale_tensor,
            max_scale=max_scale_tensor,
            init_scale=init_scale,
        )
    else:
        qdq_weight, scale, zero = quant_tensor_asym(
            weight,
            bits=bits,
            group_size=group_size,
            v=value_tensor,
            min_scale=min_scale_tensor,
            max_scale=max_scale_tensor,
        )
    return qdq_weight, scale, zero


@dataclass
class AutoRoundState:
    qcfg: QuantizeConfig
    value: nn.Parameter
    max_scale: nn.Parameter
    min_scale: Optional[nn.Parameter]
    init_scale: Optional[torch.Tensor]
    g_idx: torch.Tensor


def auto_round_scale_limit(qcfg: QuantizeConfig) -> float:
    version = getattr(qcfg, "auto_round_version", None)
    if version == AutoRoundVersion.V2:
        return 2.0
    return 1.0


def clamp_auto_round_state_(state: AutoRoundState) -> None:
    upper = auto_round_scale_limit(state.qcfg)
    with torch.no_grad():
        state.max_scale.clamp_(0.0, upper)
        if state.min_scale is not None:
            state.min_scale.clamp_(0.0, upper)


def _top_abs_mask(tensor: torch.Tensor, ratio: float = 0.001) -> torch.Tensor:
    flat = tensor.reshape(-1)
    if flat.numel() == 0:
        return torch.zeros_like(tensor, dtype=torch.bool)
    k = max(1, int(flat.numel() * ratio))
    if k >= flat.numel():
        return torch.ones_like(tensor, dtype=torch.bool)
    _, indices = torch.topk(torch.abs(flat), k)
    mask = torch.zeros_like(flat, dtype=torch.bool)
    mask[indices] = True
    return mask.reshape_as(tensor)


def auto_round_loss(
    pred: torch.Tensor,
    ref: torch.Tensor,
    *,
    qcfg: QuantizeConfig,
) -> torch.Tensor:
    if pred.dim() == 0 or ref.dim() == 0:
        raise ValueError("Auto-Round loss expects tensor outputs, not scalars.")

    if pred.shape[-1] != ref.shape[-1]:
        raise ValueError(
            "Auto-Round loss expects matching feature dimensions, "
            f"got {tuple(pred.shape)} and {tuple(ref.shape)}."
        )

    if pred.dim() == 1:
        pred = pred.unsqueeze(0)
    else:
        pred = pred.reshape(-1, pred.shape[-1])

    if ref.dim() == 1:
        ref = ref.unsqueeze(0)
    else:
        ref = ref.reshape(-1, ref.shape[-1])

    if pred.shape[0] != ref.shape[0]:
        # Calibration hooks can capture padded token blocks or squeeze the batch
        # dimension. Compare the common token prefix after flattening to
        # [tokens, hidden] so AutoRound can keep optimizing instead of failing
        # on padding-only tails.
        trim = min(pred.shape[0], ref.shape[0])
        pred = pred.narrow(0, 0, trim)
        ref = ref.narrow(0, 0, trim)

    pred_f = pred.to(torch.float32)
    ref_f = ref.to(torch.float32)
    diff = pred_f - ref_f

    if getattr(qcfg, "auto_round_version", None) == AutoRoundVersion.V2:
        mask = _top_abs_mask(diff)
        return torch.mean((torch.abs(diff) * mask.to(diff.dtype)) ** 2)

    return torch.mean(diff ** 2)


def init_auto_round_state(
    weight: torch.Tensor,
    qcfg: QuantizeConfig,
    hessian_diag: Optional[torch.Tensor] = None,
) -> AutoRoundState:
    if weight.ndim != 2:
        raise ValueError("Auto-Round currently expects 2D linear weights.")

    rows, cols = weight.shape
    groups = _group_count(cols, qcfg.group_size)
    grouped_weight, _, _, effective = _reshape_pad_tensor_by_group_size(weight, qcfg.group_size)

    init_scale = None
    if qcfg.sym:
        if qcfg.hessian_mse and hessian_diag is not None:
            if hessian_diag.numel() != cols:
                raise ValueError(
                    f"Hessian diagonal size mismatch: expected {cols}, got {hessian_diag.numel()}."
                )
            hessian_view = hessian_diag.to(device=weight.device, dtype=torch.float32).reshape(1, cols)
            hessian_grouped, _, _, _ = _reshape_pad_tensor_by_group_size(hessian_view, effective)
            hessian_grouped = hessian_grouped.repeat(rows, 1)
            init_scale = torch.abs(search_scales(grouped_weight.detach(), bits=qcfg.bits, qw=hessian_grouped))
        else:
            init_scale = torch.abs(search_scales(grouped_weight.detach(), bits=qcfg.bits))

    min_scale = None if qcfg.sym else nn.Parameter(torch.ones((rows, groups), device=weight.device, dtype=torch.float32))
    state = AutoRoundState(
        qcfg=qcfg,
        value=nn.Parameter(torch.zeros_like(weight, dtype=torch.float32)),
        max_scale=nn.Parameter(torch.ones((rows, groups), device=weight.device, dtype=torch.float32)),
        min_scale=min_scale,
        init_scale=None if init_scale is None else init_scale.reshape(rows, groups).detach().clone(),
        g_idx=build_round_g_idx(cols, qcfg.group_size, device=weight.device),
    )
    clamp_auto_round_state_(state)
    return state


def auto_round_fake_quantize(weight: torch.Tensor, state: AutoRoundState):
    min_scale = state.min_scale if state.min_scale is not None else torch.ones_like(state.max_scale)
    qdq_weight, scale, zero = build_qparams_from_best_params(
        weight=weight,
        bits=state.qcfg.bits,
        group_size=state.qcfg.group_size,
        sym=state.qcfg.sym,
        value=state.value,
        min_scale=min_scale,
        max_scale=state.max_scale,
        init_scale=state.init_scale,
    )
    q_scales, q_zeros, q_g_idx = build_qparams_for_pack(
        qdq_weight=qdq_weight,
        scale=scale,
        zero=zero,
        rows=weight.shape[0],
        in_features=weight.shape[1],
        group_size=state.qcfg.group_size,
    )
    return qdq_weight, q_scales, q_zeros, q_g_idx


def snapshot_auto_round_state(state: AutoRoundState) -> Dict[str, torch.Tensor]:
    snapshot = {
        "value": state.value.detach().clone(),
        "max_scale": state.max_scale.detach().clone(),
    }
    if state.min_scale is not None:
        snapshot["min_scale"] = state.min_scale.detach().clone()
    return snapshot


def restore_auto_round_state(state: AutoRoundState, snapshot: Dict[str, torch.Tensor]) -> None:
    with torch.no_grad():
        state.value.copy_(snapshot["value"])
        state.max_scale.copy_(snapshot["max_scale"])
        if state.min_scale is not None and "min_scale" in snapshot:
            state.min_scale.copy_(snapshot["min_scale"])
