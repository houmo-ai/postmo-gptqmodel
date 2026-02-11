"""DiceQ config builders (DiceQ pipeline config abstraction).

Why this file exists
--------------------
Historically each model definition kept a large ``layer_config`` dict with many
repeated fields + per-key comments. That was easy to understand but very
redundant and error-prone to maintain.

This module provides a base config builder ``DiceQConfigBase``:
- keep **pipeline-level defaults** in one place
- keep **train_args defaults** aligned with rotation trainer defaults
- let each model only override what is different (layers/folder/filename/etc.)

Config structure refresher
--------------------------
DiceQ pipeline config (returned by ``diceq_config_template()``) is a dict:

{
  "enabled": bool,
  "layer_config": { <target_name>: <target_cfg>, ... },
  "targets": [<target_name>, ...],
  "calib_dataset": str,
  "nsamples": int,
  "seqlen": int,
  "seed": int,
  "subset_size": Optional[int|float],
  "parallel_workers": Optional[int],
  "log_level": str,
  "tmp_data_dir": str,
}

Where each ``target_cfg`` is a dict:

{
  "description": str,
  "data_type": "activation" | "weight",
  "debug": bool,
  "folder": str,
  "layers": [pattern_str, ...],
  "save_mode": "separate" | "combined",
  "filename_template": str,
  "sampling": Optional[{"max_tokens": int, ...}],
  "train_args": {...}  # merged defaults + per-target overrides
}
"""

from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional

from .get_rotation_matrix import DEFAULT_TRAIN_ARGS


class DiceQConfigBase:
    """Base builder for DiceQ pipeline config.

    Subclasses should override `build_layer_config()` to return a mapping like:
    {
      "Q_block": {...},
      "Q_q_lowrank": {...},
      ...
    }
    """

    # Pipeline-level defaults (can be overridden in subclasses)
    #
    # - enabled: 是否开启 DiceQ pipeline（搜集训练数据 + 训练旋转矩阵 + 应用旋转）
    # - calib_dataset/nsamples/seqlen/seed: 仅在 data_type="activation" 的 target
    #   上用于激活数据搜集；weight 模式不会用到 dataloader，但字段保留以兼容混合模式。
    # - subset_size/parallel_workers: optional 全局 override（当前在 BaseQModel.get_diceq_pipeline_config
    #   里也支持 extension override）
    # - log_level: rotation 训练日志等级
    # - tmp_data_dir: DiceQ 训练数据与 artifacts 的临时目录根路径
    enabled: bool = True
    calib_dataset: str = "wikitext2"
    nsamples: int = 128
    seqlen: int = 2048
    seed: int = 0
    subset_size: Optional[float | int] = None
    parallel_workers: Optional[int] = None
    log_level: str = "INFO"
    tmp_data_dir: str = "./diceq_tmp"

    def default_train_args(self) -> Dict[str, Any]:
        """Return a deep-copied default train_args dict."""
        return copy.deepcopy(DEFAULT_TRAIN_ARGS)

    def make_train_args(self, overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Merge user overrides into defaults."""
        base = self.default_train_args()
        if overrides:
            base.update(overrides)
        return base

    def target(
        self,
        *,
        description: str,
        folder: str,
        layers: List[str],
        save_mode: str,
        filename_template: str,
        train_args_overrides: Optional[Dict[str, Any]] = None,
        sampling: Optional[Dict[str, Any]] = None,
        data_type: str = "activation",
        debug: bool = False,
    ) -> Dict[str, Any]:
        """Build one DiceQ target entry.

        Args:
            description: 说明文字，仅用于人类阅读/日志
            folder: 保存训练数据的子目录名（位于 tmp_data_dir/<model_type>/data/<folder>）
            layers: 子串匹配 pattern 列表，用于选择需要搜集的模块
            save_mode:
                - "separate": 每个 sample 单独保存一个 pt 文件（sample_{idx}_layer_{layer_idx}_...）
                - "combined": 每层合并保存（layer_{layer_idx}_...），可能包含多条 sample
            filename_template: 保存文件名模板（需与 scan_tensor_files 的解析规则兼容）
            train_args_overrides: 覆盖训练参数（默认来自 DEFAULT_TRAIN_ARGS）
            sampling: 可选采样限制，例如 {"max_tokens": 2048}（主要用于 combined）
            data_type:
                - "activation": 使用校准集前向，搜集 activation（训练时等价于 A@R）
                - "weight": 不用校准集，直接搜集 Linear.weight（训练时 W@R）
            debug:
                - True: 保留中间数据/产物（便于排查）
                - False: 训练完成后清理中间训练数据（默认）
        """
        if data_type not in {"activation", "weight"}:
            raise ValueError(f"data_type must be 'activation' or 'weight', got {data_type}")
        entry: Dict[str, Any] = {
            "description": description,
            "data_type": data_type,
            "debug": bool(debug),
            "folder": folder,
            "layers": list(layers),
            "save_mode": save_mode,
            "filename_template": filename_template,
            "train_args": self.make_train_args(train_args_overrides),
        }
        if sampling is not None:
            entry["sampling"] = dict(sampling)
        return entry

    def build_layer_config(self) -> Dict[str, Any]:
        """Override in subclasses to return non-empty dict of targets."""
        return {}

    def build(self) -> Dict[str, Any]:
        layer_config = self.build_layer_config()
        if not isinstance(layer_config, dict) or not layer_config:
            raise ValueError("DiceQConfigBase.build_layer_config() must return non-empty dict.")
        targets = [k for k, v in layer_config.items() if isinstance(v, dict)]
        return {
            "enabled": bool(self.enabled),
            "layer_config": layer_config,
            "targets": targets,
            "calib_dataset": str(self.calib_dataset),
            "nsamples": int(self.nsamples),
            "seqlen": int(self.seqlen),
            "seed": int(self.seed),
            "subset_size": self.subset_size,
            "parallel_workers": self.parallel_workers,
            "log_level": str(self.log_level),
            "tmp_data_dir": str(self.tmp_data_dir),
        }



## example config 
class MlaMoeDiceQConfig(DiceQConfigBase):
    """A detailed example config (copy & customize).

    Note:
    - This class is for documentation only; it is not referenced by the pipeline.
    - Use it as a template in model definitions: `return MyDiceQConfig().build()`.
    """

    def build_layer_config(self) -> Dict[str, Any]:
        # Example 1: activation-based training (uses calib dataset to collect activations)
        q_block = self.target(
            description="example: block-level rotation from activations",
            folder="Q_block",
            layers=["self_attn.q_proj", "mlp.gate_proj"],
            save_mode="separate",
            filename_template="sample_{idx}_layer_{layer_idx}_Q_block.pt",
            data_type="activation",
            debug=False,
            train_args_overrides={
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
        )

        # Example 2: weight-based training (no calib dataset; directly use Linear.weight)
        q_q_lowrank = self.target(
            description="example: lowrank rotation from weights",
            folder="Q_lowrank",
            layers=["self_attn.q_b_proj"],
            save_mode="combined",
            filename_template="layer_{layer_idx}_Q_lowrank.pt",
            sampling={"max_tokens": 2048},
            data_type="activation",
            debug=True,
            train_args_overrides={
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
        )

        q_kv_lowrank = self.target(
            description="KV 低秩投影层",
            folder="Q_kv_lowrank",
            layers=["self_attn.kv_b_proj"],
            save_mode="combined",
            filename_template="layer_{layer_idx}_Q_kv_lowrank.pt",
            data_type="activation",
            debug=True,
            train_args_overrides={
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
        )

        return {
            "Q_block": q_block,
            "Q_q_lowrank": q_q_lowrank,
            "Q_kv_lowrank": q_kv_lowrank,
        }

class MhaMoeDiceQConfig(DiceQConfigBase):
    """A detailed example config (copy & customize).

    Note:
    - This class is for documentation only; it is not referenced by the pipeline.
    - Use it as a template in model definitions: `return MyDiceQConfig().build()`.
    """

    def build_layer_config(self) -> Dict[str, Any]:
        # Example 1: activation-based training (uses calib dataset to collect activations)
        q_block = self.target(
            description="example: block-level rotation from activations",
            folder="Q_block",
            layers=["self_attn.q_proj", "mlp.gate"],
            save_mode="separate",
            filename_template="sample_{idx}_layer_{layer_idx}_Q_block.pt",
            data_type="activation",
            debug=False,
            train_args_overrides={
                "train_data": "all",
                "train_batch_size": 64,
                "num_epochs": 10,
                "learning_rate": 1e-3,
                "optimizer": "RAdam",
                "subset_size": 0.1,
                "device": "auto",
                "train_nsamples": 1024,
                "parallel_workers": 1,
            },
        )
        return {
            "Q_block": q_block,
        }

class MhaMlpDiceQConfig(DiceQConfigBase):
    """A detailed example config (copy & customize).

    Note:
    - This class is for documentation only; it is not referenced by the pipeline.
    - Use it as a template in model definitions: `return MyDiceQConfig().build()`.
    """

    def build_layer_config(self) -> Dict[str, Any]:
        # Example 1: activation-based training (uses calib dataset to collect activations)
        q_block = self.target(
            description="example: block-level rotation from activations",
            folder="Q_block",
            layers=["self_attn.q_proj", "mlp.gate_proj"],
            save_mode="separate",
            filename_template="sample_{idx}_layer_{layer_idx}_Q_block.pt",      
            data_type="activation",
            debug=False,
            train_args_overrides={
                "train_data": "all",
                "train_batch_size": 64,
                "num_epochs": 10,
                "learning_rate": 1e-3,
                "optimizer": "RAdam",
                "subset_size": 0.1,
                "device": "auto",
                "train_nsamples": 1024,
                "parallel_workers": 1,
            },
        )
        return {
            "Q_block": q_block,
        }

