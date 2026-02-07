# 动态 R 类型配置指南

## 概述

修改后的 `get_train_data_hook.py` 支持完全动态配置的 R 类型，不再硬编码 `r1`、`r2`、`r4`。您现在可以通过配置文件定义任意数量和名称的 R 类型。

## 使用方法

### 方式 1：使用配置文件（推荐）

```bash
python calibrater/get_train_data_hook.py \
    --model meta-llama/Llama-2-7b-hf \
    --r_config_file calibrater/my_r_config.json \
    --nsamples 128 \
    --seqlen 2048 \
    --smooth
```

### 方式 2：使用命令行参数（向后兼容）

```bash
python calibrater/get_train_data_hook.py \
    --model meta-llama/Llama-2-7b-hf \
    --r_list r1 r2 r3 r4 \
    --hook_layers "self_attn.q_proj,mlp.up_proj,mlp.down_proj" \
    --nsamples 128
```

## 配置文件格式

### 完整示例

```json
{
    "r1": {
        "description": "输入投影层（R1）",
        "folder": "r1_train",
        "layers": {
            "q_proj": {
                "pattern": "self_attn.q_proj",
                "save_mode": "separate",
                "filename_template": "sample_{idx}_layer_{layer_idx}_q_proj.pt"
            },
            "k_proj": {
                "pattern": "self_attn.k_proj",
                "save_mode": "separate",
                "filename_template": "sample_{idx}_layer_{layer_idx}_k_proj.pt"
            }
        },
        "smooth_target": false
    },
    "r2": {
        "description": "输出投影层（R2）",
        "folder": "r2_train",
        "layers": {
            "o_proj": {
                "pattern": "self_attn.o_proj",
                "save_mode": "combined",
                "filename_template": "layer_{layer_idx}_o_proj.pt"
            }
        },
        "smooth_target": true
    },
    "smooth": {
        "description": "平滑缩放因子",
        "folder": "smooth_scale"
    }
}
```

### 字段说明

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `description` | string | 否 | R 类型的描述信息 |
| `folder` | string | 是 | 保存数据的文件夹名称 |
| `layers` | object | 是 | 层配置对象 |
| `smooth_target` | boolean | 否 | 是否为平滑缩放目标层 |
| `layers.pattern` | string | 是 | 要匹配的层名称模式 |
| `layers.save_mode` | string | 是 | 保存模式：`separate`（单独保存每个样本）或 `combined`（合并保存） |
| `layers.filename_template` | string | 是 | 文件名模板，支持变量：`{idx}`（样本索引）、`{layer_idx}`（层索引）、`{name}`（数据类型名称） |
| `layers.sampling.max_tokens` | int | 否 | 采样配置，最大 token 数（仅在 `combined` 模式下有效） |

## 简化配置文件

如果不需要详细配置，可以使用简化格式：

```json
{
    "r1": {
        "folder": "r1_train"
    },
    "r2": {
        "folder": "r2_train"
    },
    "r4": {
        "folder": "r4_train"
    }
}
```

在简化格式下：
- `layers` 将使用默认配置（`q_proj`、`up_proj` 保存到 `r1`；`o_proj` 保存到 `r2`；`down_proj` 保存到 `r4`）
- `save_mode` 默认使用 `separate`（`down_proj` 使用 `combined`）
- `filename_template` 使用默认模板

## 示例场景

### 场景 1：仅收集 R1 和 R2

```json
{
    "r1": {
        "folder": "r1_train"
    },
    "r2": {
        "folder": "r2_train"
    }
}
```

运行命令：
```bash
python calibrater/get_train_data_hook.py \
    --model meta-llama/Llama-2-7b-hf \
    --r_config_file my_r1_r2_config.json \
    --nsamples 128
```

### 场景 2：自定义 R 类型名称

```json
{
    "input_proj": {
        "folder": "input_projections"
    },
    "output_proj": {
        "folder": "output_projections"
    },
    "mlp_down": {
        "folder": "mlp_down_projections"
    }
}
```

运行命令：
```bash
python calibrater/get_train_data_hook.py \
    --model meta-llama/Llama-2-7b-hf \
    --r_config_file custom_r_config.json \
    --nsamples 128
```

输出目录结构：
```
output/
└── DICEQuant/
    └── data/
        └── wikitext2_128samples/
            └── Llama-2-7b-hf/
                ├── input_projections/
                ├── output_projections/
                └── mlp_down_projections/
```

### 场景 3：只收集 K/V 投影层

```json
{
    "kv_proj": {
        "folder": "kv_projections",
        "layers": {
            "k_proj": {
                "pattern": "self_attn.k_proj",
                "save_mode": "separate",
                "filename_template": "sample_{idx}_layer_{layer_idx}_k_proj.pt"
            },
            "v_proj": {
                "pattern": "self_attn.v_proj",
                "save_mode": "separate",
                "filename_template": "sample_{idx}_layer_{layer_idx}_v_proj.pt"
            }
        }
    }
}
```

### 场景 4：带采样的 R 类型

```json
{
    "r4": {
        "folder": "r4_train",
        "layers": {
            "down_proj": {
                "pattern": "mlp.down_proj",
                "save_mode": "combined",
                "filename_template": "layer_{layer_idx}_down_proj.pt",
                "sampling": {
                    "max_tokens": 512
                }
            }
        }
    }
}
```

## 变量说明

### filename_template 支持的变量

| 变量 | 说明 | 示例值 |
|------|------|-------|
| `{idx}` | 样本索引（从 0 开始） | `0`, `1`, `2`, ... |
| `{layer_idx}` | 层索引（从 0 开始） | `0`, `1`, `2`, ... |
| `{name}` | 数据类型名称 | `q_proj`, `o_proj`, `down_proj` |

示例：
- `sample_{idx}_layer_{layer_idx}_q_proj.pt` → `sample_0_layer_0_q_proj.pt`
- `layer_{layer_idx}_o_proj.pt` → `layer_0_o_proj.pt`

## 向后兼容性

如果您不使用 `--r_config_file` 参数，脚本将使用默认配置（r1、r2、r4），保持与旧版本兼容：

```bash
python calibrater/get_train_data_hook.py \
    --model meta-llama/Llama-2-7b-hf \
    --r_list r1 r2 \
    --nsamples 128
```

## 注意事项

1. **路径创建**：脚本会自动创建所有配置的文件夹
2. **平滑缩放**：只有标记为 `smooth_target: true` 的层才会计算平滑缩放因子
3. **内存使用**：大量样本和长序列会消耗较多 GPU 内存
4. **模型支持**：目前主要支持 LLaMA 系列模型

## 命令行参数完整列表

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--model` | 必需 | 模型名称或路径 |
| `--r_config_file` | None | R 类型配置文件路径（JSON 格式） |
| `--r_list` | `['r1', 'r2']` | R 类型列表（无配置文件时使用） |
| `--hook_layers` | None | 直接指定的层模式（无配置文件时使用） |
| `--r_path` | `output/DICEQuant/data` | 保存数据的根路径 |
| `--nsamples` | 128 | 样本数量 |
| `--seqlen` | 2048 | 序列长度 |
| `--calib_dataset` | `wikitext2` | 校准数据集（wikitext2, ptb, c4） |
| `--smooth` | False | 是否计算平滑缩放 |
| `--alpha` | 0.85 | 平滑参数 |
| `--seed` | 0 | 随机种子 |
| `--hf_token` | None | HuggingFace token |
