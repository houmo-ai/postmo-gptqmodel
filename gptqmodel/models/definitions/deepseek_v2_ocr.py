# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

import copy
from typing import Any, Dict

from ..base import BaseQModel
from ...quantization.diceq.diceq_config import DiceQConfigBase


"""
DeepSeek V2 OCR QText model definition.

说明：
- 这里的实现对应的是 DeepSeek-OCR 文本侧（`hf_deepseekocr/modeling_deepseekv2.py`），
  注意力结构为 MHA（LlamaAttention），而不是官方 DeepSeek-V2 MLA 结构。
- 为了避免混淆，将原来的 `DeepSeekV2QModel` 重命名为 `DeepSeekV2OCRQModel`。
"""


class DeepSeekV2OCRQModel(BaseQModel):
    # deepseek_v2_ocr requires custom model code
    require_trust_remote_code = True

    # allow dynamic expert index for layer_modules so we don't need to write out 64 layers here
    # config.num_experts contains the actual expert count used for index
    dynamic_expert_index = "n_routed_experts"

    pre_lm_head_norm_module = "model.norm"

    # DeepSeek V2-Lite uses dynamic modules based on lora(rank):
    # https://huggingface.co/deepseek-ai/DeepSeek-V2-Lite/blob/main/modeling_deepseek.py#L712
    layer_modules_strict = False

    module_tree = [
        "model",
        "layers",
        "#",
        {
            "input_layernorm": ("input_layernorm:!",),
            "self_attn": ("q_proj:0", "k_proj:0", "v_proj:0", "o_proj:1"),
            "post_attention_layernorm": ("post_attention_layernorm:!",),
            "mlp": {
                "": ("gate_proj:0", "up_proj:0", "down_proj:1"),
                "experts": {
                    "#": ("gate_proj:0", "up_proj:0", "down_proj:1"),
                },
                "shared_experts": ("gate_proj:0", "up_proj:0", "down_proj:1"),
            },
        }
    ]

    @classmethod
    def diceq_config_template(cls) -> Dict[str, Any]:
        class DeepSeekV2OCRDiceQConfig(DiceQConfigBase):
            def build_layer_config(self) -> Dict[str, Any]:
                base_weight = {"data_type": "weight", "debug": True}
                return {
                    "Q_block": self.target(
                        description="block之间的旋转矩阵",
                        folder="Q_block",
                        layers=["self_attn.q_proj", "mlp.gate"],
                        save_mode="separate",
                        filename_template="sample_{idx}_layer_{layer_idx}_Q_block.pt",
                        data_type=base_weight["data_type"],
                        debug=base_weight["debug"],
                        train_args_overrides={
                            "train_data": "all",
                            "weight_decay": 0.01,
                            "subset_size": 0.1,
                            "eval_max_batches": 2,
                            "device": "auto",
                            "train_nsamples": 1024,
                            "parallel_workers": 1,
                        },
                    ),
                }

        return DeepSeekV2OCRDiceQConfig().build()

    def get_diceq_pipeline_config(self) -> Dict[str, Any]:
        cfg = self.diceq_config_template()
        override = self.quantize_config.extension_get("diceq")
        if isinstance(override, dict):
            merged = copy.deepcopy(cfg)
            merged.update(override)
            cfg = merged

        layer_config = cfg.get("layer_config")
        if not isinstance(layer_config, dict) or not layer_config:
            cfg["layer_config"] = copy.deepcopy(self.diceq_config_template()["layer_config"])
            layer_config = cfg["layer_config"]

        # Optional global override: apply one subset size to all DiceQ targets.
        global_subset_size = cfg.get("subset_size", None)
        if global_subset_size is not None:
            if isinstance(global_subset_size, bool):
                raise TypeError("diceq.subset_size must be int (>0) or float in (0,1].")
            for _, target_cfg in layer_config.items():
                if not isinstance(target_cfg, dict):
                    continue
                train_args = target_cfg.get("train_args")
                if not isinstance(train_args, dict):
                    train_args = {}
                    target_cfg["train_args"] = train_args
                if isinstance(global_subset_size, int):
                    if global_subset_size <= 0:
                        raise ValueError(f"diceq.subset_size(int) must be > 0, got {global_subset_size}")
                    train_args["subset_size"] = int(global_subset_size)
                    train_args.pop("train_subset_size", None)
                elif isinstance(global_subset_size, float):
                    ratio = float(global_subset_size)
                    if not (0.0 < ratio <= 1.0):
                        raise ValueError(f"diceq.subset_size(float) must be in (0,1], got {ratio}")
                    train_args["train_subset_size"] = ratio
                    train_args.pop("subset_size", None)
                else:
                    raise TypeError(
                        f"diceq.subset_size must be int or float, got {type(global_subset_size).__name__}"
                    )

        # Optional global override: apply one layer/head-wise parallel workers to all targets.
        global_parallel_workers = cfg.get("parallel_workers", None)
        if global_parallel_workers is not None:
            global_parallel_workers = int(global_parallel_workers)
            if global_parallel_workers <= 0:
                raise ValueError(f"diceq.parallel_workers must be > 0, got {global_parallel_workers}")
            for _, target_cfg in layer_config.items():
                if not isinstance(target_cfg, dict):
                    continue
                train_args = target_cfg.get("train_args")
                if not isinstance(train_args, dict):
                    train_args = {}
                    target_cfg["train_args"] = train_args
                train_args["parallel_workers"] = global_parallel_workers

        if not cfg.get("targets"):
            cfg["targets"] = [k for k, v in layer_config.items() if isinstance(v, dict)]
        return cfg

