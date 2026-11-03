# Copyright 2026 HOUMO AI
# SPDX-License-Identifier: Apache-2.0
"""Device-aware native GDN dispatch for GPTQModel-owned Qwen3.5 instances."""
from functools import update_wrapper, wraps
from inspect import unwrap
from types import FunctionType, MethodType


_TORCH_KERNELS = (
    "torch_chunk_gated_delta_rule", "torch_recurrent_gated_delta_rule",
    "causal_conv1d_fn", "causal_conv1d_update",
)


def _cell(value):
    return (lambda: value).__closure__[0]


def _with_torch_kernels(function, kernels):
    """Copy a native forward's namespace while retaining its hook decorators."""
    wrapped = getattr(function, "__wrapped__", None)
    inner = _with_torch_kernels(wrapped, kernels) if isinstance(wrapped, FunctionType) else None
    closure = function.__closure__
    if closure and inner is not None:
        closure = tuple(_cell(inner) if cell.cell_contents is wrapped else cell for cell in closure)
    namespace = dict(function.__globals__)
    namespace.update({name: kernel for name, kernel in kernels.items() if name in namespace})
    copied = FunctionType(function.__code__, namespace, function.__name__, function.__defaults__, closure)
    update_wrapper(copied, function)
    copied.__kwdefaults__ = function.__kwdefaults__
    if inner is not None:
        copied.__wrapped__ = inner
    return copied


def _device_forward(original, cpu):
    @wraps(original)
    def forward(self, hidden_states, *args, **kwargs):
        implementation = cpu if hidden_states.device.type == "cpu" else original
        return implementation(self, hidden_states, *args, **kwargs)

    forward._gptqmodel_cpu_gdn = True
    return forward


def guard_qwen35_cpu_kernels(model):
    """Keep optional FLA/causal-conv kernels away from CPU calibration tensors.

    HF's newer fallback decorators select optional packages at import time,
    even for CPU inputs. Bind a private namespace to each owned GDN instance;
    its original forward and accelerator dispatch remain available on GPU.
    """
    for module in model.modules():
        if type(module).__name__ not in {"Qwen3_5GatedDeltaNet", "Qwen3_5MoeGatedDeltaNet"}:
            continue
        # Accelerate's outer hook calls _old_forward after moving inputs.
        slot = "_old_forward" if hasattr(module, "_hf_hook") else "forward"
        bound = getattr(module, slot)
        if getattr(bound, "_gptqmodel_cpu_gdn", False):
            continue
        original = getattr(bound, "__func__", None)
        if not isinstance(original, FunctionType):
            continue
        namespace = unwrap(original).__globals__
        kernels = {name: unwrap(namespace[name]) for name in _TORCH_KERNELS
                   if callable(namespace.get(name)) and hasattr(namespace[name], "__wrapped__")}
        if "torch_chunk_gated_delta_rule" not in kernels:
            continue  # Older HF versions dispatch through instance attributes.
        cpu = _with_torch_kernels(original, kernels)
        setattr(module, slot, MethodType(_device_forward(original, cpu), module))
    return model
