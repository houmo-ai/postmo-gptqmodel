# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

"""Mix Precision config builders (Mix Precision pipeline config abstraction).

Why this file exists
--------------------
Similar to DiceQ config pattern, this module provides a base config builder
`MixPrecisionConfigBase` that allows each model definition to customize
mix precision search parameters without duplicating default values.

Config structure refresher
--------------------------
Mix Precision config (returned by `build()`) is a dict:

{
    "enabled": bool,
    "weight_bits": List[int],
    "act_bits": List[int],
    "policy": str,  # "topk" or "threshold"
    "topk": float,
    "threshold": float,
    "metric": str,  # "kl", "sqnr", "l1", "mse", "hessian_mse"
    "layer_include": Optional[List[str]],
    "layer_exclude": Optional[List[str]],
    "output_path": Optional[str],
    "moe_unified": bool,
    "moe_unify_strategy": str,  # "max", "min", "avg"
    "moe_unify_layer_level": bool,
}
"""

from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional


class MixPrecisionConfigBase:
    """Base builder for Mix Precision search config.

    Subclasses can override any attribute to customize the search behavior.
    Use this as a template in model definitions: `return MyMixPrecisionConfig().build()`.
    
    Note: enabled defaults to False. To enable mix precision search:
    1. Set quantize_config.mix_precision_search=True
    2. Override enabled=True in model's mix_precision_config_template()
    """

    enabled: bool = False
    weight_bits: List[int] = [4, 8]
    act_bits: List[int] = [8]
    policy: str = "topk"
    topk: float = 0.1
    threshold: float = 1e-6
    metric: str = "kl"
    layer_include: Optional[List[str]] = None
    layer_exclude: Optional[List[str]] = None
    output_path: Optional[str] = None
    moe_unified: bool = True
    moe_unify_strategy: str = "max"
    moe_unify_layer_level: bool = False

    def build(self) -> Dict[str, Any]:
        """Build and return the complete config dict.

        Returns:
            Dict containing all mix precision search parameters.
        """
        config = {
            "enabled": bool(self.enabled),
            "weight_bits": list(self.weight_bits),
            "act_bits": list(self.act_bits),
            "policy": str(self.policy),
            "topk": float(self.topk),
            "threshold": float(self.threshold),
            "metric": str(self.metric),
            "moe_unified": bool(self.moe_unified),
            "moe_unify_strategy": str(self.moe_unify_strategy),
            "moe_unify_layer_level": bool(self.moe_unify_layer_level),
        }

        if self.layer_include is not None:
            config["layer_include"] = list(self.layer_include)

        if self.layer_exclude is not None:
            config["layer_exclude"] = list(self.layer_exclude)

        if self.output_path is not None:
            config["output_path"] = str(self.output_path)

        return config


class DefaultMixPrecisionConfig(MixPrecisionConfigBase):
    """Default mix precision config for most models.

    This class provides sensible defaults that work well for most cases.
    Override in model definitions for specific requirements.
    
    Note: This class sets enabled=True, assuming the user explicitly chose
    to use this config. The dual-switch still requires mix_precision_search=True
    in QuantizeConfig.
    """
    enabled: bool = True


class MoEMixPrecisionConfig(MixPrecisionConfigBase):
    """Mix precision config optimized for MoE models.

    Key differences from default:
    - moe_unified=True: Ensures all experts in a MoE group use the same bit
    - moe_unify_strategy="max": Uses the highest bit for conservative quantization
    
    Note: This class sets enabled=True, assuming the user explicitly chose
    to use this config. The dual-switch still requires mix_precision_search=True
    in QuantizeConfig.
    """

    enabled: bool = True
    moe_unified: bool = True
    moe_unify_strategy: str = "max"
    moe_unify_layer_level: bool = True


def validate_mix_precision_config(config: Dict[str, Any]) -> None:
    """Validate mix precision config parameters.

    Args:
        config: Config dict to validate

    Raises:
        ValueError: If any parameter is invalid
    """
    if not config.get("enabled", True):
        return

    valid_policies = ["topk", "threshold"]
    if config.get("policy") not in valid_policies:
        raise ValueError(
            f"mix_precision.policy must be one of {valid_policies}, "
            f"got {config.get('policy')}"
        )

    valid_metrics = ["kl", "sqnr", "l1", "mse", "hessian_mse"]
    if config.get("metric") not in valid_metrics:
        raise ValueError(
            f"mix_precision.metric must be one of {valid_metrics}, "
            f"got {config.get('metric')}"
        )

    valid_moe_strategies = ["max", "min", "avg"]
    if config.get("moe_unify_strategy") not in valid_moe_strategies:
        raise ValueError(
            f"mix_precision.moe_unify_strategy must be one of {valid_moe_strategies}, "
            f"got {config.get('moe_unify_strategy')}"
        )

    weight_bits = config.get("weight_bits", [])
    if not weight_bits:
        raise ValueError("mix_precision.weight_bits cannot be empty")

    act_bits = config.get("act_bits", [])
    if not act_bits:
        raise ValueError("mix_precision.act_bits cannot be empty")

    topk = config.get("topk", 0.1)
    if topk <= 0 or topk > 1:
        raise ValueError(f"mix_precision.topk must be in (0, 1], got {topk}")

    threshold = config.get("threshold", 1e-6)
    if threshold < 0:
        raise ValueError(f"mix_precision.threshold must be non-negative, got {threshold}")
