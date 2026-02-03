# GLM-4.7-Flash (glm4_moe_lite) GPTQModel 适配说明

## 概述

GLM-4.7-Flash 使用了 MLA (Multi-head Latent Attention) 架构，需要在量化前进行权重融合以实现高效推理。

## MLA 架构特点

### 原始结构
```
Q 路径：
  hidden → q_a_proj → q_a_layernorm → q_b_proj → [q_nope, q_rope]

KV 路径：
  hidden → kv_a_proj_with_mqa → [latent, rope]
           latent → kv_a_layernorm → kv_b_proj → [k_nope, v]
           rope → k_rope (直接使用)
```

### 融合后结构
```
Q 路径：
  hidden → q_a_proj → q_a_layernorm → q_absorbed_proj (融合了 W_UK)
                                    → q_rope_proj (直接提取)

KV 路径：
  hidden → kv_a_proj_latent → kv_a_layernorm → (输出直接作为 k_nope/v)
         → kv_a_proj_rope → (应用 RoPE)

输出：
  attn_output → o_proj (融合了 W_UV)
```

## 量化分组说明

根据模型的 forward 执行流程，模块按以下分组进行量化：

### Group 0: 输入层（并行执行）
- `q_a_proj` - Q 路径的输入压缩
- `kv_a_proj_latent` - KV 路径的 latent 分支
- `kv_a_proj_rope` - KV 路径的 rope 分支

### Group 1: 内容投影层（并行，依赖 Group 0）
- `q_absorbed_proj` - Q 的 nope 部分（已吸收 kv_b_proj 的 W_UK）
- `q_rope_proj` - Q 的 rope 部分（从 q_b_proj 直接提取）

### Group 2: 输出投影（依赖 attention 计算）
- `o_proj` - 输出投影（已融合 kv_b_proj 的 W_UV）

## 关键融合操作

### 1. 拆分 kv_a_proj_with_mqa
```python
W_KV_A[:kv_lora_rank, :] → kv_a_proj_latent
W_KV_A[kv_lora_rank:, :] → kv_a_proj_rope
```

### 2. 融合 kv_b_proj 到 q_b_proj
```python
W_UK = kv_b_proj[:, :qk_nope_head_dim, :]
W_Q_absorbed = W_Q_nope @ W_UK  # Einstein summation
```

### 3. 融合 kv_b_proj 到 o_proj
```python
W_UV = kv_b_proj[:, qk_nope_head_dim:, :]
W_O_fused = W_O @ W_UV  # Einstein summation
```

## MoE 结构说明

- **shared_experts**: 使用标准 MLP (gate_proj, up_proj, down_proj)，可量化
- **routed experts**: 使用 nn.Parameter (gate_up_proj, down_proj)，**不可量化**
- **gate (router)**: 不应量化

## 使用示例

```python
from gptqmodel import GPTQModel, QuantizeConfig

# 配置
quantize_config = QuantizeConfig(
    bits=4,
    group_size=128,
    desc_act=True,
    sym=True,
)

# 加载并量化（MLA 融合会自动执行）
model = GPTQModel.from_pretrained(
    "THUDM/glm-4-9b-chat",
    quantize_config=quantize_config,
    trust_remote_code=True,  # 必需
)

# 执行量化
model.quantize(calibration_data)

# 保存
model.save("./glm-4.7-flash-gptq-int4")
```

## 文件清单

1. `gptqmodel/models/definitions/glm4_moe_lite.py` - 模型定义和 MLA 融合逻辑
2. `gptqmodel/models/definitions/__init__.py` - 添加 Glm4MoeLiteQModel 导入
3. `gptqmodel/models/auto.py` - 注册 glm4_moe_lite 模型类型
4. `examples/quantization/glm4_moe_lite_quant.py` - 量化示例脚本

## 技术细节

- MLA 融合在 `pre_quantize_generate_hook_start()` 中自动执行
- 融合后 KV cache 只需存储 latent vectors，显著降低内存占用
- 量化分组确保数值精度和依赖关系的正确性
