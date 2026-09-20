# Copyright 2026 HOUMO AI
# SPDX-License-Identifier: Apache-2.0
"""G64 reconstruction-Hessian initialization and positive log-scale tuning.

The score uses the input Gram matrix directly, not GPTQ's inverse-Hessian
Cholesky factor. Groups are independent during initialization; block output
reconstruction subsequently optimizes all rounding and scale parameters.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from functools import partial
import math

import torch

from auto_round.compressors.utils import block_forward
from auto_round.data_type.utils import round_ste
from auto_round.logger import logger
from auto_round.utils import check_to_quantized, set_module
from auto_round.wrapper import WrapperLinear


def resolve_xh_data_type(data_type, optimization_mode="standard"):
    """Resolve the public solver name to the existing packed weight format.

    The named preset enables the complete Hessian/log-scale solver. It is not
    a new integer encoding. Keep the explicit mode API for backend experiments.
    """
    if optimization_mode not in {"standard", "xh_hessian"}:
        raise ValueError(f"Unknown optimization_mode: {optimization_mode!r}")
    if data_type == "int_sym_xh_hessian":
        return "int_sym_xh", "xh_hessian"
    return data_type, optimization_mode


@dataclass(frozen=True)
class XHHessianConfig:
    candidates: int = 33
    search_min: float = 0.5
    search_max: float = 1.5
    scale_min: float = 0.5
    scale_max: float = 2.0
    row_chunk: int = 128
    token_chunk: int = 2048
    validation_samples: int = 16
    validation_interval: int = 25

    def __post_init__(self):
        for name in ("candidates", "row_chunk", "token_chunk", "validation_samples", "validation_interval"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"xh_hessian.{name} must be a positive integer")
        if self.candidates < 2:
            raise ValueError("xh_hessian.candidates must be >= 2")
        for low, high in ((self.search_min, self.search_max), (self.scale_min, self.scale_max)):
            if not (math.isfinite(low) and math.isfinite(high) and 0 < low < 1 < high):
                raise ValueError("xh_hessian scale/search bounds must satisfy 0 < min < 1 < max")


class GroupHessian:
    """Token-weighted, uncentered G64 second moments, shared by output rows."""

    def __init__(self, in_features: int, token_chunk: int = 2048):
        if in_features <= 0 or in_features % 64:
            raise ValueError("xh_hessian requires Linear input features divisible by 64")
        self.in_features = in_features
        self.token_chunk = token_chunk
        self.total = None
        self.count = 0

    @torch.no_grad()
    def add(self, inputs, mask=None):
        x = inputs.detach().reshape(-1, inputs.shape[-1])
        if x.shape[-1] != self.in_features:
            raise ValueError("Hessian input feature mismatch")
        if mask is not None:
            mask = mask.reshape(-1).to(device=x.device, dtype=torch.bool)
            if mask.numel() != x.shape[0]:
                raise ValueError("Hessian token mask does not match Linear input")
            x = x[mask]
        if not x.numel():
            return
        if self.total is None:
            self.total = torch.zeros(self.in_features // 64, 64, 64, device=x.device, dtype=torch.float32)
        # Disable autocast: these statistics must accumulate in FP32.
        with torch.autocast(device_type=x.device.type, enabled=False):
            for part in x.split(self.token_chunk):
                grouped = part.float().reshape(-1, self.in_features // 64, 64).transpose(0, 1)
                self.total.add_(torch.bmm(grouped.transpose(1, 2), grouped))
        self.count += x.shape[0]

    def finish(self):
        if not self.count:
            raise ValueError("No training tokens reached an xh_hessian Linear")
        hessian = self.total / self.count
        if not torch.isfinite(hessian).all():
            raise ValueError("Nonfinite Hessian statistics")
        return hessian


@torch.no_grad()
def search_positive_scales(weight, hessian, config, scale_dtype=torch.float16, minimum=1e-5):
    """Search e^T H_gg e with deployment scale precision and signed [-8, 7]."""
    if weight.ndim != 2 or weight.shape[1] % 64:
        raise ValueError("Expected a G64-compatible Linear weight")
    groups = weight.shape[1] // 64
    if hessian.shape != (groups, 64, 64):
        raise ValueError("Hessian shape mismatch")
    if not torch.isfinite(weight).all() or not torch.isfinite(hessian).all():
        raise ValueError("Nonfinite scale-search input")
    hessian = hessian.to(device=weight.device, dtype=torch.float32)
    scales = torch.empty(weight.shape[0], groups, device=weight.device, dtype=torch.float32)
    initial_score = final_score = 0.0
    factors = torch.linspace(config.search_min, config.search_max, config.candidates).tolist()
    with torch.autocast(device_type=weight.device.type, enabled=False):
        for start in range(0, weight.shape[0], config.row_chunk):
            w = weight[start:start + config.row_chunk].float().reshape(-1, groups, 64).transpose(0, 1)
            negative = (-w.amin(-1, keepdim=True)).clamp_min(0)
            positive = w.amax(-1, keepdim=True).clamp_min(0)
            xh = torch.maximum(negative / 8, positive / 7).clamp_min(minimum)
            gptq = (torch.maximum(negative, positive) / 7.5).clamp_min(minimum)

            def evaluate(candidate):
                s = candidate.clamp_min(minimum).to(scale_dtype).float()
                if not torch.isfinite(s).all() or not (s > 0).all():
                    raise ValueError("Scale cannot be represented in the deployment dtype")
                q = (w / s).round().clamp(-8, 7)
                error = s * q - w
                score = (torch.bmm(error, hessian) * error).sum(-1)
                return s, score

            best, best_score = evaluate(xh)
            initial_score += best_score.double().sum().item()
            # Always retain the original XH grid. No diagonal-only prefilter.
            for candidate in [gptq, *(xh * factor for factor in factors)]:
                s, score = evaluate(candidate)
                better = score < best_score
                best = torch.where(better.unsqueeze(-1), s, best)
                best_score = torch.minimum(score, best_score)
            final_score += best_score.double().sum().item()
            scales[start:start + w.shape[1]] = best.squeeze(-1).transpose(0, 1)
    return scales, {"xh_initial_score": initial_score, "searched_score": final_score}


class XHHessianLinear(WrapperLinear):
    """Reuse AutoRound's forward/packing ABI with a dedicated scale parameter."""

    def __init__(self, layer, *, xh_config, report, **kwargs):
        if (type(layer) is not torch.nn.Linear or layer.bits != 4 or layer.group_size != 64
                or layer.data_type != "int_sym_xh" or not layer.sym or layer.act_bits < 16):
            raise ValueError("xh_hessian v1 supports int_sym_xh W4A16/G64 Linear only")
        if not hasattr(layer, "_xh_hessian"):
            raise ValueError("Missing Hessian: refusing to silently fall back to min/max initialization")
        self.xh_config = xh_config
        self.xh_report = report
        super().__init__(layer, **kwargs)

    def _init_tuning_params_and_quant_func(self):
        super()._init_tuning_params_and_quant_func()
        layer = self.orig_layer
        stats = layer._xh_hessian
        init_scale, score = search_positive_scales(
            layer.weight, stats.finish(), self.xh_config, layer.scale_dtype, self.q_scale_thresh
        )
        self.register_buffer("init_scale", init_scale.reshape(-1, 1).to(self.device))
        self.xh_report.append({"module": getattr(layer, "global_name", ""), "tokens": stats.count, **score})
        del layer._xh_hessian
        for name in ("min_scale", "max_scale"):
            self.params.pop(name, None)
            delattr(self, name)
            setattr(self, name, torch.ones((), device=self.device))
        self._init_params("log_scale", torch.float32, self.init_scale.shape[0], 0.0, True)

    def _qdq_weight(self, value, min_scale, max_scale):
        with torch.no_grad():
            self.log_scale.clamp_(math.log(self.xh_config.scale_min), math.log(self.xh_config.scale_max))
        weight = self.orig_layer.weight.to(self.device)
        s = (self.init_scale * self.log_scale.exp().unsqueeze(-1)).clamp_min(self.q_scale_thresh)
        # The rounded scale participates in forward AND in the eventual pack.
        s = s.to(self.orig_layer.scale_dtype)
        w = weight.float().reshape(-1, 64)
        q = round_ste(w / s.float() + value).clamp(-8, 7)
        result = (q * s.float()).reshape_as(weight).to(weight.dtype)
        return result, s, torch.full_like(s, 8)

    def unwrapper(self, best_params):
        if best_params and "log_scale" in best_params:
            with torch.no_grad():
                self.log_scale.copy_(best_params["log_scale"].to(self.device))
        return super().unwrapper(best_params)


def wrap_block(block, enable_minmax_tuning, enable_norm_bias_tuning, *, owner, device="cpu", **kwargs):
    quantized, skipped = [], []
    for name, layer in list(block.named_modules()):
        if not isinstance(layer, torch.nn.Linear):
            continue
        if not check_to_quantized(layer):
            skipped.append(name)
            continue
        wrapper = XHHessianLinear(
            layer, xh_config=owner.xh_hessian_config, report=owner.xh_hessian_report,
            enable_minmax_tuning=enable_minmax_tuning, enable_norm_bias_tuning=False,
            device=device, **kwargs,
        )
        set_module(block, name, wrapper)
        quantized.append(name)
    if not quantized:
        raise ValueError("xh_hessian found no supported quantized Linear in this block")
    return quantized, skipped


def register_hooks(block, *, owner):
    handles = []
    for layer in block.modules():
        if check_to_quantized(layer) and type(layer) is not torch.nn.Linear:
            raise ValueError("xh_hessian v1 supports quantized torch.nn.Linear only")
        if isinstance(layer, torch.nn.Linear) and check_to_quantized(layer):
            if hasattr(layer, "_xh_hessian"):
                raise RuntimeError("Stale Hessian statistics on Linear")
            stats = GroupHessian(layer.in_features, owner.xh_hessian_config.token_chunk)
            layer._xh_hessian = stats

            def hook(module, inputs, output, stats=stats):
                stats.add(inputs[0], owner._xh_hessian_mask)

            handles.append(layer.register_forward_hook(hook))
    return handles


def batch_mask(owner, indices, nsamples, inputs):
    """Exclude padding and the held-out tail from Hessian calibration."""
    count = owner.xh_hessian_config.validation_samples
    if nsamples <= count:
        raise ValueError("xh_hessian needs more calibration samples than validation_samples")
    if owner.attention_mask:
        valid = torch.cat([owner.attention_mask[int(i)] for i in indices], dim=0).bool()
    else:
        if not isinstance(inputs, torch.Tensor) or inputs.ndim != 3 or owner.batch_dim != 0:
            raise ValueError("xh_hessian v1 requires batch-first [B,T,C] block inputs")
        valid = torch.ones(inputs.shape[:2], dtype=torch.bool)
    train = (indices < nsamples - count).to(valid.device)
    return valid & train[:, None]


@torch.no_grad()
def validation_loss(owner, block, input_ids, input_others, output, device, loss_device):
    nsamples = len(input_ids)
    start = nsamples - owner.xh_hessian_config.validation_samples
    total, elements = 0.0, 0
    for first in range(start, nsamples, owner.batch_size):
        indices = torch.arange(first, min(first + owner.batch_size, nsamples))
        expected = owner._get_current_output(output, indices).to(loss_device)
        actual = owner._get_current_q_output(block, input_ids, input_others, indices, device, loss_device)
        error = (actual.float() - expected.float()).square()
        if owner.attention_mask:
            valid = torch.cat([owner.attention_mask[int(i)] for i in indices], dim=0).to(error.device).bool()
            error = error[valid]
        total += error.double().sum().item()
        elements += error.numel()
    if not elements or not math.isfinite(total):
        raise ValueError("Invalid xh_hessian validation loss")
    return total / elements


def configure(owner, options):
    config = XHHessianConfig(**(options or {}))
    if (owner.bits != 4 or owner.group_size != 64 or owner.data_type != "int_sym_xh" or not owner.sym
            or owner.act_bits < 16 or owner.iters <= 0 or owner.is_moe_model or owner.is_auto_scheme):
        raise ValueError("xh_hessian v1 requires dense int_sym_xh W4A16/G64 with iters > 0")
    if owner.enable_alg_ext or owner.enable_norm_bias_tuning or not owner.enable_minmax_tuning:
        raise ValueError("xh_hessian requires scale tuning and excludes alg_ext/norm-bias tuning")
    if owner.scale_dtype != torch.float16 or getattr(owner, "quant_nontext_module", False):
        raise ValueError("xh_hessian v1 requires FP16 scale storage and text-only quantization")
    if owner.not_use_best_mse or len(owner.device_list) != 1 or owner.quant_lm_head:
        raise ValueError("xh_hessian requires best-checkpoint selection, one device and an unquantized head")
    if owner.nsamples <= config.validation_samples:
        raise ValueError("xh_hessian validation_samples must be smaller than nsamples")
    owner.xh_hessian_config = config
    owner.xh_hessian_report = []
    owner._xh_hessian_mask = None
    owner.block_forward = block_forward
    owner.wrapper_block = partial(wrap_block, owner=owner)
    owner._register_act_max_hook = partial(register_hooks, owner=owner)
    logger.info("xh_hessian enabled: %s", asdict(config))
