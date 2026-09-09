# Copyright 2026 HOUMO AI
# SPDX-License-Identifier: Apache-2.0
"""Bounded-memory head tuning using the vendored AutoRound implementation."""
from __future__ import annotations

import importlib
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from .auto_round_processor import AutoRoundProcessor
from ..models.writer import (
    PROCESS_LOG_NAME, PROCESS_LOG_LAYER, PROCESS_LOG_MODULE, PROCESS_LOG_TIME,
    QUANT_LOG_LOSS, QUANT_LOG_NSAMPLES, QUANT_LOG_DAMP,
)
from ..quantization.auto_round import build_round_g_idx
from ..utils.logger import setup_logger


log = setup_logger()


def _autoround_components():
    try:
        wrapper = importlib.import_module("auto_round.wrapper")
    except ModuleNotFoundError as exc:
        if exc.name != "auto_round":
            raise
        vendor = Path(__file__).resolve().parents[2] / "third_party" / "auto-round"
        if not (vendor / "auto_round").is_dir():
            raise ImportError("AutoRound head tuning requires the GPTQModel AutoRound dependency") from exc
        sys.path.insert(0, str(vendor))
        wrapper = importlib.import_module("auto_round.wrapper")
    optimizer = importlib.import_module("auto_round.sign_sgd")
    return wrapper.WrapperLinear, optimizer.SignSGD


def tune_head_chunk(weight, inputs, qcfg, options):
    """Tune independent output rows; monitor a fixed calibration token sample.

    WrapperLinear owns the quantization formula (including int_sym_xh), learned
    rounding and min/max parameters. SignSGD is AutoRound's own optimizer.
    Only token minibatches and one vocabulary chunk are resident on the GPU.
    """
    WrapperLinear, SignSGD = _autoround_components()
    with torch.inference_mode(False), torch.enable_grad():
        weight = weight.detach().clone()
        layer = torch.nn.Linear(weight.shape[1], weight.shape[0], bias=False,
                                device="meta", dtype=weight.dtype)
        layer.weight = torch.nn.Parameter(weight, requires_grad=False)
        for name, value in {"bits": qcfg.bits, "group_size": qcfg.group_size,
                            "sym": True, "data_type": qcfg.data_type,
                            "act_bits": 16, "scale_dtype": torch.float16}.items():
            setattr(layer, name, value)
        wrapper = WrapperLinear(layer, device=weight.device, enable_torch_compile=False,
                                enable_minmax_tuning=True)
        params = [{"params": [value], "lr": qcfg.minmax_lr if "scale" in name else qcfg.lr}
                  for name, value in wrapper.params.items()]
        optimizer = SignSGD(params, lr=qcfg.lr, weight_decay=0)
        scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=1.0, end_factor=0.0, total_iters=qcfg.iters)
        generator = torch.Generator().manual_seed(options["seed"])
        monitor_ids = torch.linspace(0, len(inputs) - 1, min(512, len(inputs))).long()
        monitor = inputs[monitor_ids].to(weight.device, weight.dtype).clone()
        with torch.no_grad():
            reference = layer(monitor).float()

        def evaluate():
            with torch.no_grad():
                return float(F.mse_loss(wrapper(monitor).float(), reference))

        def snapshot():
            return {name: value.detach().clone() for name, value in wrapper.params.items()}

        best_loss = evaluate()
        if not math.isfinite(best_loss):
            raise ValueError("AutoRound head initial loss is not finite")
        initial_loss = best_loss
        best_params = snapshot()
        best_step = 0
        for step in range(qcfg.iters):
            ids = torch.randint(len(inputs), (min(options["batch_tokens"], len(inputs)),), generator=generator)
            current = inputs[ids].to(weight.device, weight.dtype).clone()
            optimizer.zero_grad(set_to_none=True)
            with torch.no_grad():
                target = layer(current).float()
            loss = F.mse_loss(wrapper(current).float(), target)
            if not torch.isfinite(loss):
                raise ValueError(f"AutoRound head loss became non-finite at iteration {step}")
            loss.backward()
            optimizer.step()
            scheduler.step()
            if (step + 1) % 10 == 0 or step + 1 == qcfg.iters:
                value = evaluate()
                if math.isfinite(value) and value < best_loss:
                    best_loss, best_step, best_params = value, step + 1, snapshot()
        with torch.no_grad():
            layer = wrapper.unwrapper(best_params)
        return (layer.weight.detach(), layer.scale, layer.zp,
                {"initial_loss": initial_loss, "loss": best_loss, "best_step": best_step})


class LMHeadAutoRoundProcessor(AutoRoundProcessor):
    """Use the common looper's final-norm inputs without retaining logits."""

    def pre_process_fwd_hook(self, name):
        def capture(module, inputs, output):
            task = self.tasks[name]
            task.inputs.append(inputs[0].detach().reshape(-1, inputs[0].shape[-1]).cpu().clone())
            task.fwd_counter += 1
        return capture

    def process(self, module, device=None, **kwargs):
        task = self.tasks[module.name]
        if not task.inputs:
            raise ValueError(f"No valid calibration tokens reached {module.full_name}")
        options = self.qcfg.head_tuning
        inputs = torch.cat(task.inputs)
        task.free()
        nsamples = len(inputs)
        if nsamples == 0:
            raise ValueError("No non-padding tokens reached the head")
        started = time.perf_counter()
        scales, zeros, losses = [], [], []
        rows = module.weight.shape[0]
        for start in range(0, rows, options["row_chunk_size"]):
            end = min(start + options["row_chunk_size"], rows)
            qdq, scale, zero, metrics = tune_head_chunk(
                module.weight[start:end], inputs, task.qcfg, options)
            module.weight.data[start:end].copy_(qdq)
            scales.append(scale.reshape(end - start, -1).cpu())
            if not torch.is_tensor(zero):
                zero = torch.full_like(scales[-1], float(zero))
            zeros.append(zero.reshape(end - start, -1).cpu())
            losses.append(metrics["loss"] * (end - start))
            log.info("AutoRound head rows %d/%d: %.7g -> %.7g (step %d)",
                     end, rows, metrics["initial_loss"], metrics["loss"], metrics["best_step"])
            del qdq
        module.stream_state_payload_to_cpu({
            "q_scales": torch.cat(scales).float(), "q_zeros": torch.cat(zeros).float(),
            "q_g_idx": build_round_g_idx(module.weight.shape[1], task.qcfg.group_size),
        })
        loss = sum(losses) / rows
        duration = time.perf_counter() - started
        stat = {PROCESS_LOG_NAME: "auto_round", PROCESS_LOG_LAYER: module.layer_index,
                PROCESS_LOG_MODULE: module.name, QUANT_LOG_LOSS: f"{loss:.10f}",
                QUANT_LOG_NSAMPLES: str(nsamples), QUANT_LOG_DAMP: "0.00000",
                PROCESS_LOG_TIME: f"{duration:.3f}", "backend": "auto_round.wrapper.WrapperLinear"}
        with self.lock:
            self.avg_losses.append(loss)
            self.durations.append(duration)
            self.module_names.append(module.full_name)
            self.log.append(stat)
        self.log_new_row(stat)
