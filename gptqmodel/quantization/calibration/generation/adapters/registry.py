from __future__ import annotations

from typing import Dict, Type

from ..expert_stats_adapter import ExpertStatsAdapter
from .glm4_moe_lite import Glm4MoeLiteExpertStatsAdapter
from .qwen3_moe import Qwen3MoeExpertStatsAdapter
from .qwen3_next import Qwen3NextExpertStatsAdapter


_ADAPTERS: Dict[str, Type[ExpertStatsAdapter]] = {
    "glm4_moe_lite": Glm4MoeLiteExpertStatsAdapter,
    "qwen3_moe": Qwen3MoeExpertStatsAdapter,
    "qwen3-next": Qwen3NextExpertStatsAdapter,
    "qwen3_next": Qwen3NextExpertStatsAdapter,
}


def register_expert_stats_adapter(model_type: str, adapter_cls: Type[ExpertStatsAdapter]) -> None:
    key = str(model_type).strip().lower()
    if not key:
        raise ValueError("ExpertStatsAdapter registry: model_type is required.")
    _ADAPTERS[key] = adapter_cls


def get_expert_stats_adapter_cls(model_type: str) -> Type[ExpertStatsAdapter]:
    key = str(model_type).strip().lower()
    adapter_cls = _ADAPTERS.get(key)
    if adapter_cls is None:
        raise ValueError(f"ExpertStatsAdapter registry: unsupported model_type `{model_type}`.")
    return adapter_cls


def create_expert_stats_adapter(model, num_beams: int, model_type: str | None = None) -> ExpertStatsAdapter:
    resolved_model_type = model_type
    if resolved_model_type is None and getattr(model, "config", None) is not None:
        resolved_model_type = getattr(model.config, "model_type", None)
    adapter_cls = get_expert_stats_adapter_cls(str(resolved_model_type))
    adapter = adapter_cls(model=model, num_beams=num_beams)
    adapter.attach()
    return adapter