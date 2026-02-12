# DiceQ 旋转矩阵训练

## 概述

DiceQ（Data-Informed Calibration via Equivariant Quantization）是一种通过训练旋转矩阵来优化量化精度的方法。其核心思想是通过学习最优的正交旋转矩阵，将激活或权重变换到更利于量化的空间，从而减少量化误差。

与传统量化方法不同，DiceQ 通过数据驱动的方式自动学习最优旋转，无需手动调整量化参数，能够显著提升低比特量化后的模型精度。

## 算法原理

### 1. 核心思想

量化误差通常在激活或权重分布不均匀时较大。DiceQ 通过学习一个正交旋转矩阵 R，将数据变换到更"量化友好"的分布：

**激活模式**：`A' = A @ R`
- A: 激活张量 `[batch*seq_len, hidden_size]`
- R: 旋转矩阵 `[hidden_size, hidden_size]`
- A': 旋转后的激活

**权重模式**：`W' = W @ R`
- W: 权重张量 `[output_channels, input_channels]`
- R: 旋转矩阵 `[input_channels, input_channels]`
- W': 旋转后的权重

### 2. Q-SR 损失函数

DiceQ 使用 Q-SR（Quantization-aware Smooth-Rotation）损失函数来优化旋转矩阵：

```
L_QSR = scale_term * distortion_term
```

其中：
- `scale_term`: 控制旋转后数据的整体尺度
- `distortion_term`: 控制量化失真程度

该损失函数鼓励旋转后的数据分布更加均匀，减少离群值，从而降低量化误差。

### 3. 训练模式

DiceQ 支持三种训练模式：

| 模式 | 描述 | 适用场景 |
|------|------|----------|
| `all` | 所有层共享同一个旋转矩阵 | 参数效率高，适合层间结构相似的情况 |
| `layer_wise` | 每层独立训练旋转矩阵 | 精度更高，适合层间差异大的模型 |
| `head_wise` | 每个 attention head 独立训练 | 适合注意力层，考虑 head 间的差异 |

### 4. 正交性约束

旋转矩阵必须保持正交性（R^T @ R = I）。DiceQ 支持两种方式：

1. **QR 分解**：每次前向传播后通过 QR 分解强制正交化
2. **黎曼优化**：使用 geoopt 库在 Stiefel 流形上直接优化（推荐 RSGD/RAdam 优化器）

## 数据收集模式

DiceQ 支持两种数据收集模式：

### 激活模式 (activation)

使用校准数据集进行前向传播，收集各层的输入激活：

```python
# 需要校准数据集
"Q_block": {
    "data_type": "activation",
    "layers": ["self_attn.q_proj", "mlp.gate_proj"],
    ...
}
```

### 权重模式 (weight)

直接从模型权重中提取，无需前向传播：

```python
# 无需校准数据集
"Q_weight": {
    "data_type": "weight",
    "layers": ["self_attn.q_proj"],
    ...
}
```

## 配置

### 基础配置

```python
from gptqmodel import QuantizeConfig

qcfg = QuantizeConfig(
    bits=4,
    diceq_enabled=True,
)
```

### 模型特定配置

在模型定义中重写 `diceq_config_template()` 方法：

```python
from gptqmodel.quantization.diceq.config import DiceQConfigBase

class MyModelDiceQConfig(DiceQConfigBase):
    """自定义 DiceQ 配置"""
    
    enabled: bool = True
    calib_dataset: str = "wikitext2"
    nsamples: int = 128
    seqlen: int = 2048
    seed: int = 0
    tmp_data_dir: str = "./diceq_tmp"
    
    def build_layer_config(self):
        # 配置 Q_block：块级旋转（激活模式）
        q_block = self.target(
            description="块级旋转矩阵训练",
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
                "init_mode": "hadamard",
                "subset_size": 0.1,
            },
        )
        
        return {"Q_block": q_block}
```

### 训练参数详解

| 参数 | 类型 | 默认值 | 描述 |
|------|------|--------|------|
| `train_data` | str | "all" | 训练模式：all/layer_wise/head_wise |
| `train_batch_size` | int | 64 | 训练批次大小 |
| `num_epochs` | int | 10 | 训练轮数 |
| `learning_rate` | float | 1e-3 | 学习率 |
| `optimizer` | str | "RAdam" | 优化器：sgd/adam/adamw/RSGD/RAdam |
| `init_mode` | str | "hadamard" | 初始化方式：hadamard/random/identity |
| `cos_lr` | bool | True | 是否使用余弦退火学习率 |
| `subset_size` | float | 0.1 | 每个 epoch 采样的数据比例 |
| `sampling_policy` | str | "random" | 采样策略：random/outlier_mix |
| `device` | str | "auto" | 训练设备：auto/cuda/cpu |
| `train_nsamples` | int | 1024 | 最大训练样本数 |
| `parallel_workers` | int | 1 | 并行训练的进程数 |

### Q-SR 损失参数

| 参数 | 默认值 | 描述 |
|------|--------|------|
| `qsr_alpha` | 80.0 | 平滑最大值的温度参数 |
| `qsr_tau` | 0.05 | 门控函数的温度参数 |
| `qsr_qmax` | 7.0 | 量化范围上限 |
| `qsr_eps` | 1e-6 | 数值稳定性参数 |
| `loss_scale` | 10.0 | 损失缩放因子 |

### Outlier-Mix 采样策略

当 `sampling_policy="outlier_mix"` 时，可以配置：

| 参数 | 默认值 | 描述 |
|------|--------|------|
| `outlier_mix_ratio` | 0.5 | outlier 样本占比 |
| `outlier_topk_ratio` | 0.2 | 从 Top-K 池中抽取 outlier |
| `outlier_score_metric` | "p99_over_median" | outlier 分数计算方式 |

## 使用示例

### 完整示例

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from gptqmodel import GPTQModel, QuantizeConfig

# 加载模型
model = AutoModelForCausalLM.from_pretrained("model_path")
tokenizer = AutoTokenizer.from_pretrained("model_path")

# 配置 DiceQ
qcfg = QuantizeConfig(
    bits=4,
    diceq_enabled=True,
)

# 准备校准数据
calibration_data = [tokenizer(text) for text in sample_texts]

# 量化（DiceQ 流程会自动执行）
gptq_model = GPTQModel(model, qcfg)
gptq_model.quantize(calibration_data)

# 保存
gptq_model.save("quantized_model")
```

### 独立运行数据收集

```bash
python -m gptqmodel.quantization.diceq.get_train_data_hook \
    --model /path/to/model \
    --calib_dataset wikitext2 \
    --nsamples 128 \
    --seqlen 2048 \
    --r_config_file config/layer_config.json \
    --r_path ./diceq_data
```

### 独立训练旋转矩阵

```bash
python -m gptqmodel.quantization.diceq.get_rotation_matrix \
    --config_file config/layer_config.json \
    --target Q_block \
    --data_path ./diceq_data/Q_block \
    --output_dir ./trained_rotation \
    --device cuda
```

## 输出格式

训练完成后，旋转矩阵保存为 `.pt` 文件，包含以下字段：

```python
{
    "format_version": "dicequant.rotation.v1",
    "created_at": "2024-01-01 12:00:00",
    "target_name": "Q_block",
    "train_mode": "all",
    "hidden_size": 4096,
    "rotation_matrices": {
        "Q_block": tensor([[...], ...])  # [hidden_size, hidden_size]
    },
    "metrics": {
        "Q_block": {
            "final_train_loss": 0.00123,
            "eval_loss": 0.00145,
            "orthogonality_error": 1e-8
        }
    },
    "train_args": {...},
}
```

## 架构

```
┌─────────────────────────────────────────────────────────────┐
│                      DiceQ Pipeline                          │
├─────────────────────────────────────────────────────────────┤
│  run_diceq_pipeline()                                       │
│  ├─ _resolve_diceq_pipeline_config()                        │
│  │   └─ model.get_diceq_pipeline_config()                   │
│  ├─ collect_training_data_for_diceq()                       │
│  │   ├─ get_loaders() (activation mode)                     │
│  │   └─ get_Rs_training_data()                              │
│  │       ├─ ActivationDataCollector                         │
│  │       └─ collect_layer_activations/weights()             │
│  ├─ train_rotation_for_targets()                            │
│  │   └─ train_rotation_for_target()                         │
│  │       ├─ train_all_mode()                                │
│  │       ├─ train_layer_or_head_wise_mode()                 │
│  │       └─ train_one_rotation()                            │
│  └─ 返回旋转矩阵字典                                         │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                    GPTQ 量化应用                             │
├─────────────────────────────────────────────────────────────┤
│  应用旋转矩阵到模型权重                                       │
│  ├─ 输入层：W' = W @ R                                       │
│  └─ 输出层：W' = R.T @ W                                     │
└─────────────────────────────────────────────────────────────┘
```

## 文件结构

```
gptqmodel/quantization/diceq/
├── __init__.py              # 模块导出
├── config.py                # 配置类（DiceQConfigBase）
├── diceq_pipeline.py        # 主流水线入口
├── get_rotation_matrix.py   # 旋转矩阵训练入口
├── rotation_training_core.py # 核心训练逻辑
├── get_train_data_hook.py   # 激活/权重数据收集
├── rotation_data.py         # 数据加载和采样
├── data_utils.py            # 校准数据集加载
└── README.md                # 本文件
```

## 重要说明

### 1. 计算开销

DiceQ 训练需要：
- **激活模式**：一次完整的前向传播收集激活 + 旋转矩阵训练
- **权重模式**：直接提取权重 + 旋转矩阵训练

建议：
- 使用 GPU 加速训练
- 合理设置 `train_nsamples` 和 `subset_size`
- 启用缓存避免重复收集数据

### 2. 内存管理

- 激活数据收集时会自动清理中间数据
- 设置 `debug=True` 可保留中间数据用于调试
- 权重模式内存占用更小

### 3. 优化器选择

| 优化器 | 正交性保证 | 推荐场景 |
|--------|-----------|----------|
| RAdam | 黎曼优化 | 推荐，精度高 |
| RSGD | 黎曼优化 | 大规模训练 |
| Adam | QR 分解 | 通用场景 |
| SGD | QR 分解 | 简单场景 |

### 4. 初始化方式

| 方式 | 描述 | 适用场景 |
|------|------|----------|
| `hadamard` | 随机 Hadamard 矩阵 | 推荐，收敛快 |
| `block_hadamard` | 分块 Hadamard | 大维度场景 |
| `random` | 随机正交矩阵 | 通用场景 |
| `identity` | 单位矩阵 | 微调场景 |

### 5. 与 GPTQ 集成

DiceQ 与 GPTQ 量化流程无缝集成：
1. 在量化前自动执行 DiceQ 流程
2. 训练好的旋转矩阵自动应用到模型权重
3. 支持缓存，避免重复训练

## 参考文献

- [Quantization-aware Rotation for Neural Networks](https://arxiv.org/)
- [GPTQ: Accurate Post-Training Quantization](https://arxiv.org/abs/2210.17323)
- [Riemannian Optimization on Stiefel Manifold](https://arxiv.org/)
