# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

import copy
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from ..base import BaseQModel
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
        layer_config = {
            "Q_block": {
                "description": "block之间的旋转矩阵",  # 当前 target 的说明文字
                "folder": "Q_block",  # 激活缓存与训练数据的子目录名
                "layers": ["self_attn.q_a_proj", "mlp.gate"],  # 通过子串匹配要挂 hook 的模块名
                "save_mode": "separate",  # 数据保存方式: combined=每层合并保存, separate=逐样本保存
                "filename_template": "sample_{idx}_layer_{layer_idx}_Q_block.pt",  # 训练数据文件名模板
                "train_args": {
                    "train_data": "all",  # 训练模式: all/layer_wise/head_wise
                    "train_batch_size": 64,  # 实测更快: Q_block 在本机上 bsz=16/32 明显快于 bsz=8
                    "num_epochs": 10,  # 训练轮数
                    "learning_rate": 1e-3,  # 学习率
                    "weight_decay": 0.01,  # 权重衰减
                    "optimizer": "RAdam",  # 优化器: sgd/adam/adamw/RSGD/RAdam
                    "cos_lr": True,  # 是否使用 CosineAnnealing 学习率调度
                    "init_mode": "hadamard",  # 旋转矩阵初始化: hadamard/random/identity
                    "num_workers": 8,  # 实测更快: Q_block 在本机上 num_workers=2 显著快于 0
                    "prefetch_factor": 3,  # 每个 worker 预取批次数(num_workers>0 时生效)
                    "subset_size":0.1,  # all 模式每个 epoch 抽样样本数（提高稳定性）
                    # "sampling_policy": "outlier_mix",  # 抽样策略: random/outlier_mix
                    # "outlier_mix_ratio": 0.35,  # 每个 epoch 中 outlier 样本占比
                    # "outlier_topk_ratio": 0.2,  # 从分数 Top-K 池中抽取 outlier（按比例）
                    # "outlier_score_metric": "p99",  # outlier 分数: absmax/p99/p99_over_median
                    "skip_eval": False,  # 开启轻量评估，避免全程不可观测
                    "eval_max_batches": 2,  # 轻量评估时最多评估的 batch 数
                    "device": "auto",  # 训练设备: auto/cuda/cpu
                    "train_nsamples": 1024,  # 训练旋转矩阵时最多使用的样本数（每个pt文件抽取的样本数）
                    "parallel_workers": 1,  # all 模式不按层并行，保持 1
                },
            },
            "Q_q_lowrank": {
                "description": "q 低秩投影层",  # 当前 target 的说明文字
                "folder": "Q_q_lowrank",  # 激活缓存与训练数据的子目录名
                "layers": ["self_attn.q_b_proj"],  # 通过子串匹配要挂 hook 的模块名
                "save_mode": "combined",  # 数据保存方式
                "filename_template": "layer_{layer_idx}_Q_q_lowrank.pt",  # 训练数据文件名模板
                "sampling": {"max_tokens": 2048},  # 可选采样限制: 每条样本最多 token 数
                "train_args": {
                    "train_data": "layer_wise",  # 按层分别训练旋转矩阵
                    "train_batch_size": 4,  # 实测更快: Q_q_lowrank 在本机上 bsz=4 吞吐最高
                    "num_epochs": 10,  # 训练轮数
                    "learning_rate": 1e-3,  # 学习率
                    "weight_decay": 0,  # 权重衰减
                    "optimizer": "RAdam",  # 优化器 #RSGD/RAdam + geoopt（use_geoopt=True），前向时不走 QR，而是依赖流形参数化来保持正交约束
                    "cos_lr": True,  # 是否使用余弦退火学习率
                    "init_mode": "hadamard",  # 旋转矩阵初始化方式
                    "num_workers": 0,  # 在线训练默认关闭多进程 DataLoader，减少调度开销
                    "prefetch_factor": 2,  # 预取批次数(num_workers>0 时生效)
                    "subset_size": 1.0,  # 提高每个 epoch 的有效样本数（原先 4 过小）
                    "sampling_policy": "random",  # 抽样策略: random/outlier_mix
                    "skip_eval": False,  # 开启轻量评估，避免全程不可观测
                    # "eval_max_batches": 2,  # 轻量评估时最多评估的 batch 数
                    "device": "cuda",  # 训练设备: auto/cuda/cpu
                    "train_nsamples": 128,  # 训练旋转矩阵时最多使用的样本数
                    "parallel_workers": 1,  # 单卡在线训练默认关闭多进程层并行
                },
            },
            "Q_kv_lowrank": {
                "description": "KV 低秩投影层",  # 当前 target 的说明文字
                "folder": "Q_kv_lowrank",  # 激活缓存与训练数据的子目录名
                "layers": ["self_attn.kv_b_proj"],  # 通过子串匹配要挂 hook 的模块名
                "save_mode": "combined",  # 数据保存方式
                "train_data": "layer_wise",  # 兼容字段，实际训练以 train_args.train_data 为准
                "filename_template": "layer_{layer_idx}_Q_kv_lowrank.pt",  # 训练数据文件名模板
                "train_args": {
                    "train_data": "layer_wise",  # 按层分别训练旋转矩阵
                    "train_batch_size": 4,  # 实测更快: Q_q_lowrank 在本机上 bsz=4 吞吐最高
                    "num_epochs": 10,  # 训练轮数
                    "learning_rate": 1e-3,  # 学习率
                    "weight_decay": 0,  # 权重衰减
                    "optimizer": "RAdam",  # 优化器 #RSGD/RAdam + geoopt（use_geoopt=True），前向时不走 QR，而是依赖流形参数化来保持正交约束
                    "cos_lr": True,  # 是否使用余弦退火学习率
                    "init_mode": "hadamard",  # 旋转矩阵初始化方式
                    "num_workers": 0,  # 在线训练默认关闭多进程 DataLoader，减少调度开销
                    "prefetch_factor": 2,  # 预取批次数(num_workers>0 时生效)
                    "subset_size": 1.0,  # 提高每个 epoch 的有效样本数（原先 4 过小）
                    "sampling_policy": "random",  # 抽样策略: random/outlier_mix
                    "skip_eval": False,  # 开启轻量评估，避免全程不可观测
                    # "eval_max_batches": 2,  # 轻量评估时最多评估的 batch 数
                    "device": "cuda",  # 训练设备: auto/cuda/cpu
                    "train_nsamples": 128,  # 训练旋转矩阵时最多使用的样本数
                    "parallel_workers": 1,  # 单卡在线训练默认关闭多进程层并行
                },
            },
        }
        targets = [k for k, v in layer_config.items() if isinstance(v, dict)]

        return {
            "enabled": True,  # 是否开启 DiceQ 旋转矩阵训练流程
            "layer_config": layer_config,  # 每个 target 的层匹配、存储、训练参数配置
            "targets": targets,  # 本次要训练的 target 列表(默认 layer_config 的全部 key)
            "calib_dataset": "wikitext2",  # 校准数据集名称: wikitext2/ptb/c4
            "nsamples": 128,  # 激活采集时使用的样本数
            "seqlen": 2048,  # 每条样本截断/采样长度
            "seed": 0,  # 随机种子(采样与训练复现)
            "subset_size": None,  # 可选: 统一覆盖所有 target 的每个 epoch 样本数(int)或比例(float)
            "parallel_workers": None,  # 可选: 统一覆盖所有 target 的 layer/head-wise 并行进程数
            "log_level": "INFO",  # DiceQ 日志等级
            "tmp_data_dir": "./diceq_tmp",  # 激活缓存与旋转训练产物的临时目录
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
    
    
