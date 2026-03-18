from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional

import torch


class ExpertStatsAdapter(ABC):
    def __init__(self, model, num_beams: int):
        self.model = model
        self.num_beams = num_beams

    @abstractmethod
    def attach(self) -> None:
        pass

    @abstractmethod
    def enable_collection(self) -> None:
        pass

    @abstractmethod
    def disable_collection(self) -> None:
        pass

    @abstractmethod
    def get_step_counts(self) -> List[Optional[torch.Tensor]]:
        pass

    @abstractmethod
    def clear_step_counts(self) -> None:
        pass

    @abstractmethod
    def reorder_state(self, beam_idx: torch.Tensor) -> None:
        pass

    @abstractmethod
    def detach(self) -> None:
        pass