# r1_config_qr 配置说明（JSON）

## 1. 配置结构

配置文件顶层是一个对象，每个键代表一个训练目标（层名），例如：

```json
{
  "Q_block": {
    "folder": "Q_block",
    "train_args": {
      "train_data": "all",
      "train_batch_size": 64,
      "num_epochs": 10,
      "learning_rate": 0.001,
      "optimizer": "RAdam",
      "init_mode": "hadamard"
    }
  }
}
```

> 顶层键名（如 `Q_block`、`Q_q_lowrank`）会直接用于输出字典键名前缀，需与业务定义保持一致。

## 2. 顶层字段

### 2.1 `folder`

- 类型：`string`
- 含义：训练数据所在子目录名（脚本会自动尝试 `folder` 和 `folder_train`）
- 默认值：若缺失，回退为该目标名称（顶层键）
- 示例：`"Q_q_lowrank"`

### 2.2 `train_args`

- 类型：`object`
- 含义：训练参数集合
- 默认值：缺失字段将自动补齐为脚本默认值（见第 3 节）

### 2.3 `train_data`（可选，顶层）

- 类型：`string`
- 允许值：`"all"` / `"layer_wise"` / `"head_wise"`
- 说明：若同时存在 `train_args.train_data`，以后者为准

## 3. train_args 字段说明

| 字段 | 类型 | 允许范围/枚举 | 默认值 | 说明 |
|---|---|---|---|---|
| `train_data` | string | `all` / `layer_wise` / `head_wise` | `all` | 训练模式 |
| `train_batch_size` | int | `> 0` | `64` | batch size |
| `num_epochs` | int | `> 0` | `10` | 训练轮数 |
| `learning_rate` | float | `> 0` | `1e-3` | 学习率 |
| `momentum` | float | `[0, 1]` | `0.9` | SGD 动量 |
| `weight_decay` | float | `>= 0` | `0.0` | 权重衰减 |
| `optimizer` | string | `sgd` / `adam` / `adamw` / `RSGD` / `RAdam` | `RAdam` | 优化器 |
| `cos_lr` | bool | `true/false` | `true` | 余弦学习率调度 |
| `warmup_ratio` | float | `[0, 1]` | `0.0` | 预热比例（预留字段） |
| `init_mode` | string | `hadamard` / `random` / `identity` | `hadamard` | 旋转初始化方式 |
| `train_subset_size` | float | `(0, 1]` | `0.1` | 每 epoch 采样比例 |
| `accumulation_steps` | int | `> 0` | `1` | 梯度累积步数 |
| `qsr_alpha` | float | `> 0` | `80.0` | Q-SR 的 softmax 温度 |
| `qsr_tau` | float | `> 0` | `0.05` | Q-SR 的 sigmoid 温度 |
| `qsr_qmax` | float | `> 0` | `7.0` | 量化上界（INT4 常用 7） |
| `qsr_eps` | float | `> 0` | `1e-6` | 数值稳定项 |
| `loss_scale` | float | `> 0` | `10.0` | 损失缩放因子 |
| `num_workers` | int | `>= 0` | `0` | DataLoader workers |
| `prefetch_factor` | int | `> 0` | `2` | 预取因子（`num_workers>0` 生效） |
| `pin_memory` | bool | `true/false` | `true` | 是否启用 pin memory |
| `head_num` | int/null | `>0` 或 `null` | `null` | `head_wise` 模式头数（可自动推断） |

## 4. 参数别名（兼容旧风格）

脚本支持以下别名并自动映射：

- `ep` -> `num_epochs`
- `bsz` -> `train_batch_size`
- `lr` -> `learning_rate`
- `mom` -> `momentum`
- `optim` -> `optimizer`

## 5. 三种模式的推荐配置

### 5.1 all 模式

```json
{
  "Q_block": {
    "folder": "Q_block",
    "train_args": {
      "train_data": "all",
      "train_batch_size": 64,
      "num_epochs": 10,
      "learning_rate": 0.001,
      "optimizer": "RAdam",
      "init_mode": "hadamard"
    }
  }
}
```

### 5.2 layer_wise 模式

```json
{
  "Q_q_lowrank": {
    "folder": "Q_q_lowrank",
    "train_args": {
      "train_data": "layer_wise",
      "train_batch_size": 128,
      "num_epochs": 10,
      "learning_rate": 0.001,
      "optimizer": "adamw",
      "weight_decay": 0.01,
      "init_mode": "hadamard"
    }
  }
}
```

### 5.3 head_wise 模式

```json
{
  "Q_q_lowrank": {
    "folder": "Q_q_lowrank",
    "train_args": {
      "train_data": "head_wise",
      "head_num": 24,
      "train_batch_size": 128,
      "num_epochs": 10,
      "learning_rate": 0.001,
      "optimizer": "adamw",
      "init_mode": "hadamard"
    }
  }
}
```

## 6. 常见错误提示

- `Unsupported train_args keys`: 出现未支持字段，请检查拼写
- `train_data must be one of ...`: 模式值非法
- `hidden_size must be divisible by head_num`: 分头设置不合法
- `No .pt files found`: 数据目录错误或未生成数据
- `Sample must be 2D after normalization`: 文件内容格式不符合要求
