# 混合精度搜索

## 概述

混合精度搜索是一种自动化的权重位宽分配算法，用于确定神经网络中每一层的最优量化位宽。通过分析层敏感度，它为敏感层分配较高的位宽，为不敏感层分配较低的位宽，从而在模型精度和压缩率之间取得平衡。

**注意**：本实现仅针对**权重(weight)**进行bit搜索，不包含激活量化。

## 算法原理

### 1. 敏感度分析

核心思想是测量每层在量化时对整体模型输出退化的程度：

```
Sensitivity(layer_i) = Metric(FP_output, Quantized_output_i)
```

其中：
- `FP_output`：全精度模型输出（基准）
- `Quantized_output_i`：仅量化 layer_i 权重时的输出
- `Metric`：敏感度测量函数（KL 散度、MSE 等）

### 2. 位宽分配策略

根据计算出的敏感度，对层进行排序并分配位宽：

**TopK 策略：**
- 前 `k%` 最敏感的层 → 高位宽（如 8-bit）
- 其余层 → 低位宽（如 4-bit）

**阈值策略：**
- 敏感度 > 阈值 → 高位宽
- 敏感度 <= 阈值 → 低位宽

### 3. MoE 专家分组

对于混合专家模型（MoE），算法按**每个MoE层**进行分组：

```
model.layers.0.mlp.experts   -> 组0 (layer 0的所有路由专家)
model.layers.1.mlp.experts   -> 组1 (layer 1的所有路由专家)
model.layers.2.mlp.experts   -> 组2 (layer 2的所有路由专家)
...
```

每个MoE层独立参与排序和bit分配：
- 整个MoE组作为一个节点计算敏感度（组内所有子节点的敏感度取平均）
- 整个MoE组作为一个节点参与排序
- 组内所有层使用相同的bit

**路由专家 vs 共享专家**：
- **路由专家**：`model.layers.*.mlp.experts.*` - 参与分组统一
- **共享专家**：`model.layers.*.mlp.shared_experts.*` - 每个独立处理

### 4. 固定位宽配置

支持为特定层设置固定位宽，跳过敏感度搜索：

```python
fixed_bits = {
    "model.layers.0.mlp.gate_proj": 8,  # 精确路径：仅第0层
    "mlp.gate_proj": 4,                  # 通配符：所有层的gate_proj
    "self_attn.k_proj": 8,              # 所有k_proj层
}
```

### 5. 不量化特定层

使用 `fixed_bits` 设为 `None` 表示不量化：

```python
fixed_bits = {
    "lm_head": None,                     # 不量化输出层
    "model.embed_tokens": None,          # 不量化embedding层
}
```

或者使用 `layer_exclude` 排除：

```python
layer_exclude = ["lm_head", "embed_tokens"],  # 排除这些层不参与搜索
```

## 支持的度量指标

| 指标 | 描述 | 适用场景 |
|------|------|----------|
| `kl` | KL 散度 | 通用场景，分布比较 |
| `mse` | 均方误差 | 回归任务 |
| `sqnr` | 信号量化噪声比 | 信号处理 |
| `l1` | L1 距离 | 对异常值鲁棒 |

## 配置

### 基础配置

```python
from gptqmodel import QuantizeConfig

# 启用混合精度搜索
qcfg = QuantizeConfig(
    mix_precision_search=True,
)
```

### 通过扩展进行高级配置

```python
qcfg = QuantizeConfig(mix_precision_search=True)
qcfg.extension_set("mix_precision", {
    # 位宽候选（仅权重）
    "weight_bits": [4, 8],
    
    # 分配策略
    "policy": "topk",  # 或 "threshold"
    "topk": 0.1,       # 前 10% 获得高位宽
    "threshold": 1e-6, # 用于阈值策略
    
    # 敏感度度量
    "metric": "kl",
    
    # 层过滤 - 排除不参与搜索的层
    "layer_include": ["q_proj", "v_proj"],  # 可选：只搜索这些层
    "layer_exclude": ["lm_head", "embed_tokens"],  # 排除这些层
    
    # 固定位宽配置
    "fixed_bits": {
        "model.layers.0.mlp.gate_proj": 8,  # 精确路径
        "mlp.gate_proj": 4,                 # 通配符：所有层
        "lm_head": None,                    # 不量化
    },
    
    # 结果输出路径
    "output_path": "./mix_precision_result.json",
    
    # MoE 设置
    "moe_unified": True,  # 每个MoE层的路由专家作为一组
})
```

### 模型特定配置

对于模型定义，重写 `mix_precision_config_template()`：

```python
class MyMoEModel(BaseQModel):
    @classmethod
    def mix_precision_config_template(cls) -> Dict[str, Any]:
        class MyMoEMixPrecisionConfig(MixPrecisionConfigBase):
            moe_unified = True
            weight_bits = [4, 8]
            policy = "topk"
            topk = 0.1
            layer_exclude = ["lm_head"]  # 排除输出层
            fixed_bits = {
                "self_attn.q_proj": 8,  # 所有q_proj使用8bit
            }
        return MyMoEMixPrecisionConfig().build()
```

## 使用示例

### 示例1：基础使用

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from gptqmodel import GPTQModel, QuantizeConfig

# 加载模型
model = AutoModelForCausalLM.from_pretrained("model_path")
tokenizer = AutoTokenizer.from_pretrained("model_path")

# 配置混合精度搜索
qcfg = QuantizeConfig(
    bits=4,
    mix_precision_search=True,
)
qcfg.extension_set("mix_precision", {
    "weight_bits": [4, 8],
    "policy": "topk",
    "topk": 0.1,
    "metric": "kl",
})

# 准备校准数据
calibration_data = [tokenizer(text) for text in sample_texts]

# 使用混合精度量化
gptq_model = GPTQModel(model, qcfg)
gptq_model.quantize(calibration_data)

# 保存
gptq_model.save("quantized_model")
```

### 示例2：不量化特定层

```python
qcfg.extension_set("mix_precision", {
    "weight_bits": [4, 8],
    "layer_exclude": ["lm_head", "embed_tokens"],  # 排除这些层
    "fixed_bits": {
        "lm_head": None,  # 明确指定不量化
    },
})
```

### 示例3：MoE模型配置

```python
qcfg.extension_set("mix_precision", {
    "weight_bits": [4, 8],
    "policy": "topk",
    "topk": 0.1,
    "moe_unified": True,  # 每个MoE层的路由专家作为一组
    # 共享专家会自动独立处理
})
```

## 输出格式

搜索结果保存为 JSON：

```json
{
  "bit_allocation": {
    "model.layers.0.self_attention.q_proj": {"weight_bits": 8},
    "model.layers.0.self_attention.k_proj": {"weight_bits": 4},
    "model.layers.0.mlp.experts": {"weight_bits": 4},
    "model.layers.1.mlp.experts": {"weight_bits": 8}
  },
  "sensitivities": {
    "model.layers.0.self_attention.q_proj": 0.00123,
    "model.layers.0.self_attention.k_proj": 0.00045,
    "model.layers.0.mlp.experts": 0.00067,
    "model.layers.1.mlp.experts": 0.00189
  },
  "statistics": {
    "total_layers": 100,
    "bit_distribution": {4: 90, 8: 10},
    "avg_sensitivity": 0.00078,
    "high_bit_ratio": 0.1
  },
  "fixed_bits": {
    "lm_head": null
  }
}
```

## 重要说明

### 1. 计算开销

混合精度搜索需要：
- 一次完整的前向传播用于基准
- 每个候选层一次前向传播用于敏感度计算

对于大型模型，这可能很耗时。建议：
- 使用较小的校准数据集
- 启用结果缓存
- 如有 GPU 可用，在 GPU 上运行

### 2. MoE 模型分组逻辑

```
每个MoE层独立作为一个组：
- model.layers.0.mlp.experts -> 组0（layer 0的所有路由专家）
- model.layers.1.mlp.experts -> 组1（layer 1的所有路由专家）
- ...
- model.layers.0.mlp.shared_experts -> 独立（不参与分组）
- model.layers.1.mlp.shared_experts -> 独立（不参与分组）
```

分组后：
1. 计算整个组的平均敏感度
2. 所有组按敏感度排序
3. 整个组使用相同的bit

### 3. 层选择

使用 `layer_include` 和 `layer_exclude` 可以：
- 聚焦于特定层类型的搜索
- 跳过已知不敏感的层
- 排除应保持全精度的层

### 4. 固定位宽使用场景

- `bits = 8`：强制使用8bit
- `bits = 4`：强制使用4bit
- `bits = None`：不量化该层

### 5. 与 GPTQ 集成

位宽分配通过 `dynamic` 配置自动应用于 GPTQ 量化：
- 搜索结果更新 `quantize_config.dynamic`
- GPTQProcessor 在量化期间读取动态配置
- 不在配置中的层使用默认bits

## 架构

```
┌─────────────────────────────────────────────────────────────┐
│                    混合精度流水线                             │
├─────────────────────────────────────────────────────────────┤
│  run_mix_precision_pipeline()                               │
│  ├─ get_mix_precision_config()                              │
│  │   └─ 合并模板 + 用户覆盖                                   │
│  ├─ run_mix_precision_search()                              │
│  │   ├─ LayerSelector.select()                              │
│  │   │   ├─ 遍历所有Linear层                                 │
│  │   │   ├─ 应用include/exclude过滤                          │
│  │   │   ├─ 检查固定位宽配置                                  │
│  │   │   ├─ MoE分组：每个MoE层作为独立组                      │
│  │   │   └─ 共享专家独立处理                                  │
│  │   ├─ _run_baseline()                                     │
│  │   ├─ _compute_sensitivities()                           │
│  │   │   └─ MoE组敏感度 = 组内平均                            │
│  │   ├─ _allocate_bits()                                    │
│  │   │   └─ 按敏感度排序，统一分配                            │
│  │   └─ expand_bit_allocation_to_modules()                  │
│  └─ 更新 quantize_config.dynamic                            │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                      GPTQ 量化                                │
├─────────────────────────────────────────────────────────────┤
│  GPTQProcessor.preprocess()                                  │
│  └─ qcfg.dynamic_get(module_name, "bits", default_bits)     │
└─────────────────────────────────────────────────────────────┘
```

## 文件结构

```
gptqmodel/quantization/mix_precision/
├── __init__.py              # 模块导出
├── mixsearch_config.py      # 配置类定义
├── pipeline.py              # 主流水线入口
├── search_engine.py         # 搜索引擎实现
├── layer_selector.py        # 层选择逻辑
├── moe_handler.py           # MoE 结构处理
├── metrics.py              # 敏感度度量
└── MIXSEARCH_README.md     # 本文件
```

## 参考文献

- [HAWQ-V3: Dyadic Neural Network Quantization](https://arxiv.org/abs/2011.10680)
- [Automatic Bit-width Selection for Quantization](https://arxiv.org/abs/2004.10568)
- [GPTQ: Accurate Post-Training Quantization](https://arxiv.org/abs/2210.17323)
