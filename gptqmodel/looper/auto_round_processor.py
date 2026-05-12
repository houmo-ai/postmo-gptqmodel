# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import copy
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
import transformers
from torch.nn import Module

from ..looper.gptq_processor import GPTQProcessor
from ..looper.loop_processor import DTYPE_SIZE_COLUMN, MODULE_FEATURE_COLUMN
from ..looper.named_module import NamedModule
from ..models import BaseQModel
from ..models.writer import (
    PROCESS_LOG_FWD_TIME,
    PROCESS_LOG_LAYER,
    PROCESS_LOG_MODULE,
    PROCESS_LOG_NAME,
    PROCESS_LOG_TIME,
    PROCESS_USED_MEMORY,
    QUANT_LOG_DAMP,
    QUANT_LOG_LOSS,
    QUANT_LOG_NSAMPLES,
)
from ..quantization.auto_round import (
    auto_round_loss,
    auto_round_fake_quantize,
    build_input_hessian,
    clamp_auto_round_state_,
    init_auto_round_state,
    restore_auto_round_state,
    snapshot_auto_round_state,
)
from ..quantization.config import METHOD, QuantizeConfig
from ..utils.failsafe import normalize_failsafe
from ..utils.logger import setup_logger


log = setup_logger()


@dataclass
class _AutoRoundTask:
    module: NamedModule
    qcfg: QuantizeConfig
    failsafe: Optional[str] = None
    inputs: List[torch.Tensor] = field(default_factory=list)
    outputs: List[torch.Tensor] = field(default_factory=list)
    fwd_counter: int = 0

    def add_batch(self, inp: torch.Tensor, out: torch.Tensor) -> None:
        self.inputs.append(inp.detach().to(device="cpu").clone())
        self.outputs.append(out.detach().to(device="cpu").clone())
        self.fwd_counter += 1

    def free(self) -> None:
        self.inputs.clear()
        self.outputs.clear()


class AutoRoundProcessor(GPTQProcessor):
    def preprocess(self, module: NamedModule, failsafe=None, **kwargs):
        if self.qcfg.dynamic_get(layer_name=module.full_name) is False:
            return

        qcfg_clone = copy.deepcopy(self.qcfg)

        if self.qcfg.dynamic is not None:
            qcfg_clone.bits = self.qcfg.dynamic_get(module.full_name, "bits", qcfg_clone.bits)
            qcfg_clone.sym = self.qcfg.dynamic_get(module.full_name, "sym", qcfg_clone.sym)
            qcfg_clone.mse = self.qcfg.dynamic_get(module.full_name, "mse", qcfg_clone.mse)
            qcfg_clone.group_size = self.qcfg.dynamic_get(module.full_name, "group_size", qcfg_clone.group_size)
            desc_act_override = self.qcfg.dynamic_get(module.full_name, "desc_act", None)
            if desc_act_override is not None:
                qcfg_clone.desc_act = desc_act_override
            failsafe_override = self.qcfg.dynamic_get(module.full_name, "failsafe", None)
            if failsafe_override is not None:
                qcfg_clone.failsafe = normalize_failsafe(failsafe_override, qcfg_clone.failsafe)

        self.qcfg_dynamic = qcfg_clone
        self.tasks[module.name] = _AutoRoundTask(
            module=module,
            qcfg=qcfg_clone,
            failsafe=normalize_failsafe(failsafe, qcfg_clone.failsafe),
        )

    def pre_process_fwd_hook(self, name: str) -> Callable[[Module, Tuple[torch.Tensor, ...], torch.Tensor], None]:
        def tmp(module, inp: Tuple[torch.Tensor, ...], out: torch.Tensor):
            task = self.tasks[name]
            if not isinstance(out, torch.Tensor):
                raise TypeError("Auto-Round only supports tensor outputs for quantized submodules.")
            task.add_batch(inp[0].data, out.data)
            del inp, out

        return tmp

    def _module_forward(self, module: Module, inp: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        bias = getattr(module, "bias", None)
        if isinstance(module, torch.nn.Linear):
            return F.linear(inp, weight, bias)
        if isinstance(module, transformers.pytorch_utils.Conv1D):
            size_out = inp.shape[:-1] + (weight.shape[1],)
            out = torch.addmm(
                bias if bias is not None else torch.zeros(weight.shape[1], device=inp.device, dtype=inp.dtype),
                inp.reshape(-1, inp.shape[-1]),
                weight,
            )
            return out.view(size_out)
        raise NotImplementedError(f"Auto-Round does not support module type `{type(module).__name__}`.")

    def process(
        self,
        module: NamedModule,
        device: torch.device = None,
        subset: Optional[Dict[str, NamedModule]] = None,
        name_index: Optional[int] = None,
        return_fp_outputs: bool = False,
        previous_subset: Optional[Dict[str, NamedModule]] = None,
        subset_index: Optional[int] = None,
        subset_total: Optional[int] = None,
        **kwargs,
    ):
        del subset, name_index, return_fp_outputs, previous_subset, subset_index, subset_total, kwargs

        task = self.tasks[module.name]
        if not task.inputs:
            raise ValueError(f"Auto-Round captured no calibration inputs for `{module.full_name}`.")

        with torch.inference_mode(False):
            target_device = device if isinstance(device, torch.device) else torch.device(device or module.weight.device)
            base_weight = module.weight.detach().to(device=target_device).clone()
            inputs = [tensor.to(device=target_device, dtype=base_weight.dtype).clone() for tensor in task.inputs]
            outputs = [tensor.to(device=target_device, dtype=torch.float32).clone() for tensor in task.outputs]

            hessian_diag = build_input_hessian(inputs) if task.qcfg.hessian_mse else None
            state = init_auto_round_state(base_weight, task.qcfg, hessian_diag=hessian_diag)

            params = [state.value]
            if task.qcfg.enable_minmax_tuning:
                params.append(state.max_scale)
                if state.min_scale is not None:
                    params.append(state.min_scale)

            best_loss = float("inf")
            best_snapshot = snapshot_auto_round_state(state)
            iterations = max(int(task.qcfg.iters), 0)

            if params and iterations > 0:
                param_groups = [{"params": [state.value], "lr": task.qcfg.lr}]
                if task.qcfg.enable_minmax_tuning:
                    tune_params = [state.max_scale]
                    if state.min_scale is not None:
                        tune_params.append(state.min_scale)
                    param_groups.append({"params": tune_params, "lr": task.qcfg.minmax_lr})
                optimizer = torch.optim.Adam(param_groups)

                for _ in range(iterations):
                    optimizer.zero_grad(set_to_none=True)
                    qdq_weight, _, _, _ = auto_round_fake_quantize(base_weight, state)
                    total_loss = 0.0
                    for inp, ref_out in zip(inputs, outputs):
                        pred = self._module_forward(module.module, inp.to(dtype=qdq_weight.dtype), qdq_weight)
                        total_loss = total_loss + auto_round_loss(pred, ref_out, qcfg=task.qcfg)
                    total_loss.backward()
                    optimizer.step()
                    clamp_auto_round_state_(state)

                    loss_value = float(total_loss.detach().item())
                    if loss_value < best_loss:
                        best_loss = loss_value
                        best_snapshot = snapshot_auto_round_state(state)
            else:
                qdq_weight, _, _, _ = auto_round_fake_quantize(base_weight, state)
                total_loss = 0.0
                for inp, ref_out in zip(inputs, outputs):
                    pred = self._module_forward(module.module, inp.to(dtype=qdq_weight.dtype), qdq_weight)
                    total_loss = total_loss + auto_round_loss(pred, ref_out, qcfg=task.qcfg)
                best_loss = float(total_loss.detach().item())

            if not task.qcfg.not_use_best_mse:
                restore_auto_round_state(state, best_snapshot)

            start = time.perf_counter()
            qdq_weight, q_scales, q_zeros, q_g_idx = auto_round_fake_quantize(base_weight, state)
            duration = time.perf_counter() - start

        module.stream_state_payload_to_cpu(
            {
                "q_scales": q_scales,
                "q_zeros": q_zeros,
                "q_g_idx": q_g_idx,
            },
        )

        if self.calculate_w_wq_diff:
            w_wq_diff = module.weight.data.to(dtype=torch.float32) - qdq_weight.to(dtype=torch.float32)
            with self.lock:
                module.state.update(
                    {
                        "w_wq_diff": w_wq_diff,
                        "wq": qdq_weight.detach().clone(),
                    }
                )

        nsamples = sum(int(t.shape[0]) if t.ndim > 1 else 1 for t in task.inputs)
        stat = {
            PROCESS_LOG_NAME: self.name(),
            PROCESS_LOG_LAYER: module.layer_index,
            PROCESS_LOG_MODULE: module.name,
            MODULE_FEATURE_COLUMN: self.module_feature_summary(module),
            DTYPE_SIZE_COLUMN: self.module_dtype_size_summary(module),
            QUANT_LOG_LOSS: f"{best_loss:.10f}",
            QUANT_LOG_NSAMPLES: f"{nsamples}",
            QUANT_LOG_DAMP: "0.00000",
            PROCESS_LOG_TIME: f"{duration:.3f}",
            PROCESS_LOG_FWD_TIME: self.formatted_fwd_time(),
            PROCESS_USED_MEMORY: self.device_memory_report(),
        }
        if self.qcfg.dynamic is not None:
            stat["dynamic"] = self.qcfg.dynamic_get(layer_name=module.full_name)

        with self.lock:
            self.avg_losses.append(best_loss)
            self.durations.append(duration)
            self.module_names.append(f"layer-{module.layer_index}-{module.name}")
            self.log.append(stat)

        self.log_new_row(stat)

        task.free()
        module.weight.data = qdq_weight.detach().to(dtype=module.weight.dtype)

    def finalize(self, model: BaseQModel, **kwargs):
        model.quantized = True
        model.quantize_config.quant_method = METHOD.AUTO_ROUND
        super(GPTQProcessor, self).finalize(model=model, **kwargs)

    @classmethod
    def name(cls) -> str:
        return "auto_round"

    def has_captured_input_ids(self, name: str) -> bool:
        return self.tasks[name].fwd_counter > 0
