# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

import copy
from typing import Any, Dict

from ..base import BaseQModel


"""
DeepSeek V2 MLA+MoE QText model definition（标准 DeepSeek-V2 文本结构）.

说明：
- 对应 `hf_models/hf_deepseekv2/modeling_deepseek_v2.py` / `configuration_deepseek_v2.py`，
  使用 MLA 注意力（q_a_proj / kv_a_proj_with_mqa / kv_b_proj 等）+ MoE MLP 结构。
- 与 OCR 版 `DeepSeekV2OCRQModel` 区分开：OCR 使用的是 MHA（LlamaAttention）。
"""


class DeepSeekV2QModel(BaseQModel):
    # 标准 deepseek_v2 需要自定义 HF model 实现（hf_deepseekv2）
    require_trust_remote_code = True

    # 允许动态 expert 数量：配置里 `n_routed_experts` 控制
    dynamic_expert_index = "n_routed_experts"

    pre_lm_head_norm_module = "model.norm"

    # DeepSeek V2-Lite / V2 使用基于 rank 的动态模块（如 q_lora_rank/kv_lora_rank）
    layer_modules_strict = False

    # MLA 结构下的 module tree：
    # - attention: q_a_proj / q_b_proj / kv_a_proj_with_mqa / kv_b_proj / o_proj
    # - mlp: 支持 dense / experts / shared_experts 三种路径
    module_tree = [
        "model",
        "layers",
        "#",
        {
            "input_layernorm": ("input_layernorm:!",),
            "self_attn": ("q_a_proj:0", "q_b_proj:0", "kv_a_proj_with_mqa:0", "kv_b_proj:0", "o_proj:1"),
            "post_attention_layernorm": ("post_attention_layernorm:!",),
            "mlp": {
                "": ("gate_proj:0", "up_proj:0", "down_proj:1"),
                "experts": {
                    "#": ("gate_proj:0", "up_proj:0", "down_proj:1"),
                },
                "shared_experts": ("gate_proj:0", "up_proj:0", "down_proj:1"),
            },
        },
    ]

    @classmethod
    def diceq_config_template(cls) -> Dict[str, Any]:
        """
        DiceQ 旋转训练配置（标准 DeepSeek-V2 MLA）：
        - Q_block: 对 block 级 hidden 表示做正交旋转（作用在 q_a_proj / mlp.gate 上）
        - Q_q_lowrank: 针对 q_a_proj / q_b_proj 的低秩空间
        - Q_kv_lowrank: 针对 kv_a_proj_with_mqa / kv_b_proj 的低秩空间
        """
        layer_config = {
            "Q_block": {
                "description": "block 之间的旋转矩阵（作用在 hidden 维度）",
                "folder": "Q_block",
                "layers": ["self_attn.q_a_proj", "mlp.gate"],
                "save_mode": "separate",
                "filename_template": "sample_{idx}_layer_{layer_idx}_Q_block.pt",
                "train_args": {
                    "train_data": "all",
                    "train_batch_size": 64,
                    "num_epochs": 10,
                    "learning_rate": 1e-3,
                    "weight_decay": 0.01,
                    "optimizer": "RAdam",
                    "cos_lr": True,
                    "init_mode": "hadamard",
                    "num_workers": 8,
                    "prefetch_factor": 3,
                    "subset_size": 0.1,
                    "skip_eval": False,
                    "eval_max_batches": 2,
                    "device": "auto",
                    "train_nsamples": 1024,
                    "parallel_workers": 1,
                },
            },
            "Q_q_lowrank": {
                "description": "q 低秩投影旋转矩阵（q_a_proj / q_b_proj）",
                "folder": "Q_q_lowrank",
                "layers": ["self_attn.q_b_proj"],
                "save_mode": "combined",
                "filename_template": "layer_{layer_idx}_Q_q_lowrank.pt",
                "sampling": {"max_tokens": 2048},
                "train_args": {
                    "train_data": "layer_wise",
                    "train_batch_size": 4,
                    "num_epochs": 10,
                    "learning_rate": 1e-3,
                    "weight_decay": 0,
                    "optimizer": "RAdam",
                    "cos_lr": True,
                    "init_mode": "hadamard",
                    "num_workers": 0,
                    "prefetch_factor": 2,
                    "subset_size": 1.0,
                    "sampling_policy": "random",
                    "skip_eval": False,
                    "device": "cuda",
                    "train_nsamples": 128,
                    "parallel_workers": 1,
                },
            },
            "Q_kv_lowrank": {
                "description": "KV 低秩投影旋转矩阵（kv_a_proj_with_mqa / kv_b_proj）",
                "folder": "Q_kv_lowrank",
                "layers": ["self_attn.kv_b_proj"],
                "save_mode": "combined",
                "train_data": "layer_wise",
                "filename_template": "layer_{layer_idx}_Q_kv_lowrank.pt",
                "train_args": {
                    "train_data": "layer_wise",
                    "train_batch_size": 4,
                    "num_epochs": 10,
                    "learning_rate": 1e-3,
                    "weight_decay": 0,
                    "optimizer": "RAdam",
                    "cos_lr": True,
                    "init_mode": "hadamard",
                    "num_workers": 0,
                    "prefetch_factor": 2,
                    "subset_size": 1.0,
                    "sampling_policy": "random",
                    "skip_eval": False,
                    "device": "cuda",
                    "train_nsamples": 128,
                    "parallel_workers": 1,
                },
            },
        }
        targets = [k for k, v in layer_config.items() if isinstance(v, dict)]

        return {
            "enabled": True,
            "layer_config": layer_config,
            "targets": targets,
            "calib_dataset": "wikitext2",
            "nsamples": 128,
            "seqlen": 2048,
            "seed": 0,
            "subset_size": None,
            "parallel_workers": None,
            "log_level": "INFO",
            "tmp_data_dir": "./diceq_tmp",
        }

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

        # 可选：全局 subset_size 覆盖
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

        # 可选：全局 parallel_workers 覆盖
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


