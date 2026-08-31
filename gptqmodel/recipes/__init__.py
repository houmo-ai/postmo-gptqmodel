"""Stable recipe APIs for model-family-specific quantization flows."""

from .gemma4 import Gemma4QuantRecipeResult, quantize_gemma4
from .deepseek_v4_flash import (
    DeepSeekV4FlashPreparationResult,
    DeepSeekV4FlashQuantRecipeResult,
    prepare_deepseek_v4_flash_bf16,
    quantize_deepseek_v4_flash,
)
from .ling3_flash import Ling3FlashQuantRecipeResult, quantize_ling3_flash
from .laguna_autoround import LagunaAutoRoundRecipeResult, quantize_laguna_autoround
from .minicpm_o_4_5 import MiniCPMO45QuantRecipeResult, quantize_minicpm_o_4_5
from .minicpm5 import MiniCPM5QuantRecipeResult, quantize_minicpm5
from .qwen3_next import Qwen3NextQuantRecipeResult, quantize_qwen3_next
from .qwen35 import Qwen35QuantRecipeResult, quantize_qwen35
from .qwen35_autoround import Qwen35AutoRoundRecipeResult, quantize_qwen35_autoround

__all__ = [
    "Gemma4QuantRecipeResult",
    "DeepSeekV4FlashPreparationResult",
    "DeepSeekV4FlashQuantRecipeResult",
    "Ling3FlashQuantRecipeResult",
    "LagunaAutoRoundRecipeResult",
    "MiniCPMO45QuantRecipeResult",
    "MiniCPM5QuantRecipeResult",
    "Qwen3NextQuantRecipeResult",
    "Qwen35QuantRecipeResult",
    "Qwen35AutoRoundRecipeResult",
    "quantize_gemma4",
    "prepare_deepseek_v4_flash_bf16",
    "quantize_deepseek_v4_flash",
    "quantize_ling3_flash",
    "quantize_laguna_autoround",
    "quantize_minicpm_o_4_5",
    "quantize_minicpm5",
    "quantize_qwen3_next",
    "quantize_qwen35",
    "quantize_qwen35_autoround",
]
