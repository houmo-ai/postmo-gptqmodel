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
    "weight_bits": List[int],           # e.g., [4, 8]
    "policy": str,                      # "topk" or "threshold"
    "topk": float,                      # 0.1
    "threshold": float,                  # 1e-6
    "metric": str,                      # "kl", "sqnr", "l1", "mse", "ppl"
    "layer_include": Optional[List[str]],
    "layer_exclude": Optional[List[str]],
    "group_by_path": Optional[Dict[str, str]],  # {"path_pattern": "group_name"}
    "fixed_bits": Optional[Dict[str, Optional[int]]],  # {"module_path": bits or None}
    "output_path": Optional[str],
    "moe_unified": bool,                       # unify routed experts per layer
}

Examples:
- group_by_path: {"model.layers.*.mlp.experts": "experts"} 
  -> Each MoE layer's experts become a group (not all MoE layers merged)
  
- fixed_bits: {
      "model.layers.0.mlp.gate_proj": 8,  # exact path: only layer 0
      "mlp.gate_proj": 4,                   # wildcard: all layers
      "lm_head": None,                      # do not quantize
    }
    
- MoE grouping: model.layers.0.mlp.experts -> group 0
                model.layers.1.mlp.experts -> group 1
                (each MoE layer is independent group)
"""

from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional, Union


class MixPrecisionConfigBase:
    """Base builder for Mix Precision search config.

    Subclasses can override any attribute to customize the search behavior.
    Use this as a template in model definitions: `return MyMixPrecisionConfig().build()`.
    
    Note: enabled defaults to False. To enable mix precision search:
    1. Set quantize_config.mix_precision_search=True
    2. Override enabled=True in model's mix_precision_config_template()
    
    Features:
    - Weight-only bit search (no activation quantization)
    - Flexible layer grouping via group_by_path
    - Fixed bit configuration for specific layers (including None to skip)
    - MoE routing expert unification per layer (each MoE layer is independent group)
    """

    enabled: bool = False
    weight_bits: List[int] = [4, 8]
    policy: str = "topk"
    topk: float = 0.1
    threshold: float = 1e-6
    metric: str = "kl"
    layer_include: Optional[List[str]] = None
    layer_exclude: Optional[List[str]] = None
    group_by_path: Optional[Dict[str, str]] = None
    fixed_bits: Optional[Dict[str, Optional[int]]] = None
    output_path: Optional[str] = None
    moe_unified: bool = True

    def build(self) -> Dict[str, Any]:
        """Build and return the complete config dict.

        Returns:
            Dict containing all mix precision search parameters.
        """
        config = {
            "enabled": bool(self.enabled),
            "weight_bits": list(self.weight_bits),
            "policy": str(self.policy),
            "topk": float(self.topk),
            "threshold": float(self.threshold),
            "metric": str(self.metric),
            "moe_unified": bool(self.moe_unified),
        }

        if self.layer_include is not None:
            config["layer_include"] = list(self.layer_include)

        if self.layer_exclude is not None:
            config["layer_exclude"] = list(self.layer_exclude)

        if self.group_by_path is not None:
            config["group_by_path"] = dict(self.group_by_path)

        if self.fixed_bits is not None:
            config["fixed_bits"] = dict(self.fixed_bits)

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
    - moe_unified=True: Each MoE layer's routed experts use the same bit (as a group)
    - Shared experts: Always independent (each linear gets its own bit)
    
    Note: This class sets enabled=True, assuming the user explicitly chose
    to use this config. The dual-switch still requires mix_precision_search=True
    in QuantizeConfig.
    """

    enabled: bool = True
    moe_unified: bool = True


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

    valid_metrics = ["kl", "sqnr", "l1", "mse", "ppl", "perplexity"]
    if config.get("metric") not in valid_metrics:
        raise ValueError(
            f"mix_precision.metric must be one of {valid_metrics}, "
            f"got {config.get('metric')}"
        )

    weight_bits = config.get("weight_bits", [])
    if not weight_bits:
        raise ValueError("mix_precision.weight_bits cannot be empty")

    topk = config.get("topk", 0.1)
    if topk <= 0 or topk > 1:
        raise ValueError(f"mix_precision.topk must be in (0, 1], got {topk}")

    threshold = config.get("threshold", 1e-6)
    if threshold < 0:
        raise ValueError(f"mix_precision.threshold must be non-negative, got {threshold}")
