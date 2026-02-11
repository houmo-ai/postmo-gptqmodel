# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

import copy
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from ..base import BaseQModel
from ...quantization.diceq.config import DiceQConfigBase
from .deepseek_v3 import MLA, Glm4MoeLiteNaiveMoeNew


def convert_glm4_moe_lite_convert(module, config):
    from transformers.models.glm4_moe_lite.modeling_glm4_moe_lite import Glm4MoeLiteAttention, Glm4MoeLiteNaiveMoe
    # 遍历模块树，替换 HF 原始注意力层
    # to_replace = []
    # for name, sub_module in module.named_modules():
    #     if isinstance(sub_module, Glm4MoeLiteAttention):
    #         to_replace.append((name, sub_module))

    # for name, sub_module in to_replace:
    #     parent_name, _, child_name = name.rpartition(".")
    #     parent = module.get_submodule(parent_name) if parent_name else module
    #     new_module = MLA(config=config, ori_attention=sub_module)
    #     setattr(parent, child_name, new_module)
    # torch.cuda.empty_cache()

    # 遍历模块树，替换 HF 原始注意力层
    to_replace = []
    for name, sub_module in module.named_modules():
        if isinstance(sub_module, Glm4MoeLiteNaiveMoe):
            to_replace.append((name, sub_module))

    for name, sub_module in to_replace:
        parent_name, _, child_name = name.rpartition(".")
        parent = module.get_submodule(parent_name) if parent_name else module
        new_module = Glm4MoeLiteNaiveMoeNew(config=config, ori_moe=sub_module)
        setattr(parent, child_name, new_module)
    torch.cuda.empty_cache()
    return module


class Glm4MoeLiteQModel(BaseQModel):    
    # glm4_moe_lite requires custom model code
    require_trust_remote_code = True
    
    # allow dynamic expert index for layer_modules
    # config.n_routed_experts contains the actual expert count
    dynamic_expert_index = "n_routed_experts"  # 修复：使用 config 中的正确属性名
    
    pre_lm_head_norm_module = "model.norm"
    support_offload_to_disk = False
    
    # Set to False since GLM-4.7-Flash may have dynamic module structures
    layer_modules_strict = False
    
    module_tree = [
        "model",
        "layers",
        "#",
        {
            "input_layernorm": ("input_layernorm:!",),
            "self_attn": (
                # 原始HF结构的模块名（因为attention转换被注释掉了）
                "q_a_proj:0",
                "q_b_proj:1",
                "kv_a_proj_with_mqa:0",
                "kv_b_proj:1",
                "o_proj:2",
            ),
            "post_attention_layernorm": ("post_attention_layernorm:!",),
            "mlp:?": {
                # MoE structure - shared_experts uses standard MLP
                "shared_experts": ("gate_proj:0", "up_proj:0", "down_proj:1"),
                # Router should not be quantized (it's a nn.Linear with weight as router)
                "gate": ("gate:!",),
                # experts 转换后结构是: experts.gate_proj[i], experts.up_proj[i], experts.down_proj[i]
                # 路径直接是 experts.gate_proj.0, experts.gate_proj.1 等
                # 使用 {expert_index} 占位符，由 build_moe_modules_if_need 展开
                "experts": (
                    "gate_proj.{expert_index}:0",
                    "up_proj.{expert_index}:0",
                    "down_proj.{expert_index}:1",
                ),
                # Standard MLP for dense layers (non-MoE)
                "gate_proj": ("gate_proj:0",),
                "up_proj": ("up_proj:0",),
                "down_proj": ("down_proj:1",),
            },
        }
    ]

    @classmethod
    def diceq_config_template(cls) -> Dict[str, Any]:
        class Glm4MoeLiteDiceQConfig(DiceQConfigBase):
            def build_layer_config(self) -> Dict[str, Any]:
                # Weight-training mode by default for this model config.
                base_weight = {"data_type": "weight", "debug": True}

                q_block = self.target(
                    description="block之间的旋转矩阵",
                    folder="Q_block",
                    layers=["self_attn.q_a_proj", "mlp.gate"],
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
                )

                q_q_lowrank = self.target(
                    description="q 低秩投影层",
                    folder="Q_q_lowrank",
                    layers=["self_attn.q_b_proj"],
                    save_mode="combined",
                    filename_template="layer_{layer_idx}_Q_q_lowrank.pt",
                    sampling={"max_tokens": 2048},
                    data_type=base_weight["data_type"],
                    debug=base_weight["debug"],
                    train_args_overrides={
                        "train_data": "layer_wise",
                        "train_batch_size": 4,
                        "weight_decay": 0,
                        "num_workers": 0,
                        "prefetch_factor": 2,
                        "subset_size": 1.0,
                        "sampling_policy": "random",
                        "device": "cuda",
                        "train_nsamples": 128,
                        "parallel_workers": 1,
                    },
                )

                q_kv_lowrank = self.target(
                    description="KV 低秩投影层",
                    folder="Q_kv_lowrank",
                    layers=["self_attn.kv_b_proj"],
                    save_mode="combined",
                    filename_template="layer_{layer_idx}_Q_kv_lowrank.pt",
                    data_type=base_weight["data_type"],
                    debug=base_weight["debug"],
                    train_args_overrides={
                        "train_data": "layer_wise",
                        "train_batch_size": 4,
                        "weight_decay": 0,
                        "num_workers": 0,
                        "prefetch_factor": 2,
                        "subset_size": 1.0,
                        "sampling_policy": "random",
                        "device": "cuda",
                        "train_nsamples": 128,
                        "parallel_workers": 1,
                    },
                )

                return {
                    "Q_block": q_block,
                    "Q_q_lowrank": q_q_lowrank,
                    "Q_kv_lowrank": q_kv_lowrank,
                }

        return Glm4MoeLiteDiceQConfig().build()

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
    
    def before_model_load(self, load_quantized_model=False):
        if load_quantized_model:
            import transformers.models.glm4_moe_lite.modeling_glm4_moe_lite as glm4_moe_lite_modeling

            glm4_moe_lite_modeling.Glm4MoeLiteNaiveMoe = Glm4MoeLiteNaiveMoeNew

    def pre_quantize_generate_hook_start(self):
        """
        Before quantization, fuse MLA weights for efficient inference.
        This converts the original MLA structure to a fused structure that:
        1. Reduces KV cache size by storing only latent vectors
        2. Absorbs kv_b_proj into q and o projections
        """
        super().pre_quantize_generate_hook_start()

        pass

    def after_model_load(self, model, load_quantized_model):
        self.model = super().after_model_load(model, load_quantized_model)
        
        # 只在首次加载未量化模型时执行MoE结构转换
        # 加载量化模型时，由于 before_model_load 已经进行了 monkey patch，
        # from_config 创建的模型已经是 Glm4MoeLiteNaiveMoeNew 结构
        if not load_quantized_model:
            self.model = convert_glm4_moe_lite_convert(self.model, self.config)
        
        return self.model
    
    
