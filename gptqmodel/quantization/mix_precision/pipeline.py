# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

"""Mix Precision pipeline interface.

This module provides the main entry point for mix precision search,
following the same pattern as diceq_pipeline.py for consistency.
"""

import copy
from typing import Any, Dict, List, Optional, Union

import torch

from ...utils.logger import setup_logger
from .mixsearch_config import DefaultMixPrecisionConfig, validate_mix_precision_config
from .search_engine import run_mix_precision_search, expand_bit_allocation_to_modules

log = setup_logger()


def run_mix_precision_pipeline(
    model: torch.nn.Module,
    quantize_config: Any,
    calibration_data: List,
    device: Union[str, torch.device],
    mix_precision_config_template: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Dict[str, int]]]:
    """Execute mix precision search pipeline and update dynamic config.

    Requires both quantize_config.mix_precision_search=True AND
    mix_precision_config_template.enabled=True to run.

    This is the main entry point for mix precision search, designed to be
    called from base.py with minimal code changes.

    Args:
        model: Model instance
        quantize_config: QuantizeConfig instance (will be modified in-place)
        calibration_data: Calibration dataset for sensitivity computation
        device: Device for computation
        mix_precision_config_template: Model-specific config template from
            mix_precision_config_template() method. If None, uses default.

    Returns:
        Bit allocation dict or None if search is disabled
    """
    if not quantize_config.mix_precision_search:
        return None

    config = get_mix_precision_config(
        quantize_config=quantize_config,
        mix_precision_config_template=mix_precision_config_template,
    )

    if not config.get("enabled", False):
        log.info("Mix precision config.enabled=False, skipping mix precision search")
        return None

    log.info("Running mix precision search...")

    bit_allocation = run_mix_precision_search(
        model=model,
        config=config,
        calibration_data=calibration_data,
        output_path=config.get("output_path"),
        device=device,
    )

    expanded_allocation = expand_bit_allocation_to_modules(
        bit_allocation, model, config
    )

    if quantize_config.dynamic is None:
        quantize_config.dynamic = {}

    for module_name, bits_config in expanded_allocation.items():
        quantize_config.dynamic[module_name] = {
            "bits": bits_config["weight_bits"],
        }

    log.info(f"Mix precision search completed. Updated {len(expanded_allocation)} layers.")

    return expanded_allocation


def get_mix_precision_config(
    quantize_config: Any,
    mix_precision_config_template: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Get mix precision config with model-specific overrides.

    Args:
        quantize_config: QuantizeConfig instance
        mix_precision_config_template: Model-specific config template.
            If None, uses default config.

    Returns:
        Complete mix precision config dict
    """
    if mix_precision_config_template is None:
        cfg = DefaultMixPrecisionConfig().build()
    else:
        cfg = copy.deepcopy(mix_precision_config_template)

    override = quantize_config.extension_get("mix_precision")
    if isinstance(override, dict):
        merged = copy.deepcopy(cfg)
        merged.update(override)
        cfg = merged

    validate_mix_precision_config(cfg)

    return cfg
