"""Bounded-device Qwen search replay using native multimodal input preparation."""
from __future__ import annotations

from contextlib import contextmanager

import torch
import torch.nn.functional as F


class _InputsCaptured(Exception):
    pass


def _move(value, device, memo=None):
    """Move a tensor tree, preserving shared masks/rotary tensors within a record."""
    memo = {} if memo is None else memo
    if isinstance(value, torch.Tensor):
        if id(value) not in memo:
            memo[id(value)] = value.detach().to(device)
        return memo[id(value)]
    if isinstance(value, dict):
        return {k: _move(v, device, memo) for k, v in value.items()}
    if isinstance(value, tuple):
        return tuple(_move(v, device, memo) for v in value)
    if isinstance(value, list):
        return [_move(v, device, memo) for v in value]
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    raise TypeError(f"Unsupported search replay input: {type(value).__name__}")


@contextmanager
def _on_device(modules, device):
    """Only the current stage is resident, including on exceptional exits."""
    try:
        for module in modules:
            module.to(device)
        yield
    finally:
        for module in modules:
            module.cpu()


class _CaptureBlock(torch.nn.Module):
    def __init__(self, calls, last):
        super().__init__()
        self.calls, self.last = calls, last

    def forward(self, hidden_states, **kwargs):
        self.calls.append((hidden_states, kwargs))
        if self.last:
            raise _InputsCaptured
        return hidden_states


class LayerwiseSearchReplay:
    """Sweep samples through one real decoder block at a time.

    Native forward prepares vision features, M-RoPE and both attention masks.
    Temporary capture blocks stop it before norm/head; no decoder computation is
    substituted during scoring. Only immutable preparation inputs are cached:
    every candidate reruns every decoder block with its own hidden states.
    """

    def __init__(self, model, device):
        self.model, self.device = model, device
        self.language = getattr(model.model, "language_model", model.model)
        self.layers = self.language.layers
        if not self.layers:
            raise ValueError("Layerwise search requires decoder blocks")
        self.prepared = {}
        self.sweeps = 0

    def prepare(self, records):
        modules = [self.language.embed_tokens, self.language.rotary_emb]
        if any(r["modality"] == "image" for r in records):
            modules.append(self.model.model.visual)
        calls = []
        catchers = torch.nn.ModuleList(
            [_CaptureBlock(calls, i == len(self.layers) - 1) for i in range(len(self.layers))])
        try:
            self.language.layers = catchers
            with _on_device(modules, self.device):
                for record in records:
                    calls.clear()
                    try:
                        self.model(**_move(record["inputs"], self.device), use_cache=False,
                                   logits_to_keep=record["positions"].to(self.device), return_dict=True)
                    except _InputsCaptured:
                        pass
                    else:
                        raise RuntimeError("Native model did not reach the decoder capture boundary")
                    # Shared tensors are copied to CPU only once per sample.
                    self.prepared[id(record)] = _move(
                        (calls[0][0], [kwargs for _, kwargs in calls]), "cpu")
                    calls.clear()
        finally:
            calls.clear()
            self.language.layers = self.layers

    def forward(self, records):
        self.sweeps += 1
        hidden = [self.prepared[id(r)][0] for r in records]
        for index, layer in enumerate(self.layers):
            with _on_device([layer], self.device):
                for i, record in enumerate(records):
                    kwargs = _move(self.prepared[id(record)][1][index], self.device)
                    output = layer(hidden[i].to(self.device), **kwargs)
                    if not isinstance(output, torch.Tensor):
                        raise TypeError("Qwen decoder replay expects a tensor output")
                    hidden[i] = output.cpu()
                    del output, kwargs
            if (index + 1) % 8 == 0 or index + 1 == len(self.layers):
                print(f"[auto_mixed_precision] replay={self.sweeps} block={index + 1}/{len(self.layers)} "
                      f"samples={len(records)}", flush=True)
        results = []
        with _on_device([self.language.norm, self.model.lm_head], self.device):
            for state, record in zip(hidden, records):
                normalized = self.language.norm(state.to(self.device))
                selected = normalized[:, record["positions"].to(self.device), :]
                logits = self.model.lm_head(selected)
                results.append(F.log_softmax(logits[0].float(), dim=-1).cpu())
                del normalized, selected, logits
        return results
