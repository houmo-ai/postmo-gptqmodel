from __future__ import annotations

from functools import wraps
from types import MethodType
from typing import List, Optional

import torch

from ..expert_stats_adapter import ExpertStatsAdapter
from .qwen3_moe import _resolve_layers_container


def _patch_glm4_moe_block(block, num_beams: int) -> bool:
    num_experts = getattr(block, "n_routed_experts", None)
    if num_experts is None and hasattr(block, "gate"):
        num_experts = getattr(block.gate, "n_routed_experts", None)
    if num_experts is None:
        return False

    block.save_expert_cnt = False
    block.top_cnt = torch.zeros(num_beams, num_experts)

    def clear_expert_cnt(self):
        ne = getattr(self, "n_routed_experts", None)
        if ne is None and hasattr(self, "gate"):
            ne = getattr(self.gate, "n_routed_experts", self.top_cnt.shape[1])
        self.top_cnt = torch.zeros(self.top_cnt.shape[0], ne, device=self.top_cnt.device, dtype=self.top_cnt.dtype)

    block.clear_expert_cnt = MethodType(clear_expert_cnt, block)

    original_forward = block.forward

    @wraps(original_forward)
    def counting_forward(hidden_states):
        residuals = hidden_states
        orig_shape = hidden_states.shape
        router_logits = block.gate(hidden_states)
        topk_indices, topk_weights = block.route_tokens_to_experts(router_logits)

        if block.save_expert_cnt:
            batch_size, sequence_length = orig_shape[:2]
            for token_index in range(topk_indices.shape[0]):
                beam_index = token_index // sequence_length
                for topk_slot in range(topk_indices.shape[1]):
                    block.top_cnt[beam_index, topk_indices[token_index, topk_slot].item()] += 1

        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
        hidden_states = block.experts(hidden_states, topk_indices, topk_weights).view(*orig_shape)
        hidden_states = hidden_states + block.shared_experts(residuals)
        return hidden_states

    block._ebss_original_forward = original_forward
    block.forward = counting_forward
    return True


class Glm4MoeLiteExpertStatsAdapter(ExpertStatsAdapter):
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
            if self._has_counting_attrs(mlp):
                self.layers.append(mlp)
            elif hasattr(mlp, "gate") and hasattr(mlp, "experts") and hasattr(mlp, "route_tokens_to_experts"):
                if _patch_glm4_moe_block(mlp, self.num_beams):
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
