from __future__ import annotations

from functools import wraps
from types import MethodType
from typing import List, Optional

import torch

from ..expert_stats_adapter import ExpertStatsAdapter


def _resolve_layers_container(model):
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "layers"):
        return model.layers
    raise ValueError("Qwen3MoeExpertStatsAdapter: model does not expose `layers`.")


def _patch_moe_block(block, num_beams: int):
    """Dynamically add EBSS counting to a standard Qwen3MoeSparseMoeBlock."""
    num_experts = block.gate.num_experts if hasattr(block, "gate") and hasattr(block.gate, "num_experts") else None
    if num_experts is None:
        return False

    block.save_expert_cnt = False
    block.top_cnt = torch.zeros(num_beams, num_experts)

    def clear_expert_cnt(self):
        ne = self.gate.num_experts if hasattr(self, "gate") and hasattr(self.gate, "num_experts") else self.top_cnt.shape[1]
        self.top_cnt = torch.zeros(self.top_cnt.shape[0], ne)

    block.clear_expert_cnt = MethodType(clear_expert_cnt, block)

    original_forward = block.forward

    @wraps(original_forward)
    def counting_forward(hidden_states):
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states_reshaped = hidden_states.view(-1, hidden_dim)
        _, routing_weights, selected_experts = block.gate(hidden_states_reshaped)

        if block.save_expert_cnt:
            for m in range(selected_experts.shape[0]):
                idx = m // sequence_length
                for n in range(selected_experts.shape[1]):
                    block.top_cnt[idx, selected_experts[m, n].item()] += 1

        final_hidden_states = block.experts(hidden_states_reshaped, selected_experts, routing_weights)
        return final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)

    block._ebss_original_forward = original_forward
    block.forward = counting_forward
    return True


class Qwen3MoeExpertStatsAdapter(ExpertStatsAdapter):
    def __init__(self, model, num_beams: int):
        super().__init__(model=model, num_beams=num_beams)
        self.layers = []
        self._attached = False

    def _has_counting_attrs(self, mlp):
        return all(hasattr(mlp, name) for name in ("save_expert_cnt", "top_cnt", "clear_expert_cnt"))

    def attach(self) -> None:
        layers = _resolve_layers_container(self.model)
        self.layers = []
        for layer in layers:
            mlp = getattr(layer, "mlp", None)
            if mlp is None:
                self.layers.append(None)
                continue
            # Already has counting attrs (old forked model or previously patched)
            if self._has_counting_attrs(mlp):
                self.layers.append(mlp)
            # Standard model: dynamically patch
            elif hasattr(mlp, "gate") and hasattr(mlp, "experts"):
                if _patch_moe_block(mlp, self.num_beams):
                    self.layers.append(mlp)
                else:
                    self.layers.append(None)
            else:
                self.layers.append(None)
        self._attached = True

    def enable_collection(self) -> None:
        for mlp in self.layers:
            if mlp is not None:
                mlp.save_expert_cnt = True

    def disable_collection(self) -> None:
        for mlp in self.layers:
            if mlp is not None:
                mlp.save_expert_cnt = False

    def get_step_counts(self) -> List[Optional[torch.Tensor]]:
        step_counts: List[Optional[torch.Tensor]] = []
        for mlp in self.layers:
            if mlp is None or getattr(mlp, "top_cnt", None) is None:
                step_counts.append(None)
            else:
                step_counts.append(mlp.top_cnt.detach().clone())
        return step_counts

    def clear_step_counts(self) -> None:
        for mlp in self.layers:
            if mlp is not None:
                mlp.clear_expert_cnt()

    def reorder_state(self, beam_idx: torch.Tensor) -> None:
        return None

    def detach(self) -> None:
        self.disable_collection()
        self.layers = []
        self._attached = False
