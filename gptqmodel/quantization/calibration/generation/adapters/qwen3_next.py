from __future__ import annotations

from typing import List, Optional

import torch

from ..expert_stats_adapter import ExpertStatsAdapter
from ...moe_ebss.qwen3_next_ebss import patch_qwen3_next_sparse_moe_block
from .qwen3_moe import _resolve_layers_container


class Qwen3NextExpertStatsAdapter(ExpertStatsAdapter):
    def __init__(self, model, num_beams: int):
        super().__init__(model=model, num_beams=num_beams)
        self.layers = []
        self._attached = False

    def attach(self) -> None:
        patch_qwen3_next_sparse_moe_block()
        layers = _resolve_layers_container(self.model)
        self.layers = []
        for layer in layers:
            mlp = getattr(layer, "mlp", None)
            if mlp is not None and all(hasattr(mlp, name) for name in ("save_expert_cnt", "top_cnt", "clear_expert_cnt")):
                self.layers.append(mlp)
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
