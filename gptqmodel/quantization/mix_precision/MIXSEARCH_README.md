# 混合精度搜索

## 概述

混合精度搜索是一种自动化的位宽分配算法，用于确定神经网络中每一层的最优量化位宽。通过分析层敏感度，它为敏感层分配较高的位宽，为不敏感层分配较低的位宽，从而在模型精度和压缩率之间取得平衡。

## 算法原理

### 1. 敏感度分析

核心思想是测量每层在量化时对整体模型输出退化的贡献程度：

```
Sensitivity(layer_i) = Metric(FP_output, Quantized_output_i)
```

其中：
- `FP_output`：全精度模型输出（基准）
- `Quantized_output_i`：仅量化 layer_i 时的输出
- `Metric`：敏感度测量函数（KL 散度、MSE 等）

### 2. 位宽分配策略

根据计算出的敏感度，对层进行排序并分配位宽：

**TopK 策略：**
- 前 `k%` 最敏感的层 → 高位宽（如 8-bit）
- 其余层 → 低位宽（如 4-bit）

**阈值策略：**
- 敏感度 > 阈值 → 高位宽
- 敏感度 <= 阈值 → 低位宽

### 3. MoE 专家一致性

对于混合专家模型，同一 MoE 层内的所有专家必须使用相同的位宽，以确保：
- 一致的推理行为
- 简化的部署流程
- 数值稳定性

算法会自动：
1. 通过模式匹配识别 MoE 结构
2. 将专家分组
3. 根据组内最敏感的专家分配统一的位宽

## 支持的度量指标

| 指标 | 描述 | 适用场景 |
|------|------|----------|
| `kl` | KL 散度 | 通用场景，分布比较 |
| `mse` | 均方误差 | 回归任务 |
| `sqnr` | 信号量化噪声比 | 信号处理 |
| `l1` | L1 距离 | 对异常值鲁棒 |
| `hessian_mse` | Hessian 加权 MSE | 二阶敏感度 |

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
    # 位宽候选
    "weight_bits": [4, 8],
    "act_bits": [8],
    
    # 分配策略
    "policy": "topk",  # 或 "threshold"
    "topk": 0.1,       # 前 10% 获得高位宽
    "threshold": 1e-6, # 用于阈值策略
    
    # 敏感度度量
    "metric": "kl",
    
    # 层过滤
    "layer_include": ["q_proj", "v_proj"],  # 可选
    "layer_exclude": ["lm_head"],           # 可选
    
    # 结果输出路径
    "output_path": "./mix_precision_result.json",
    
    # MoE 设置
    "moe_unified": True,
    "moe_unify_strategy": "max",  # max, min 或 avg
    "moe_unify_layer_level": True,
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
            moe_unify_strategy = "max"
            moe_unify_layer_level = True
            weight_bits = [4, 8]
            policy = "topk"
            topk = 0.1
        return MyMoEMixPrecisionConfig().build()
```

## 使用示例

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

## 输出格式

搜索结果保存为 JSON：

```json
{
  "bit_allocation": {
    "model.layers.0.self_attention.q_proj": {"weight_bits": 8, "act_bits": 8},
    "model.layers.0.self_attention.k_proj": {"weight_bits": 4, "act_bits": 8},
    "model.layers.0.mlp.experts.0.gate_proj": {"weight_bits": 4, "act_bits": 8}
  },
  "sensitivities": {
    "model.layers.0.self_attention.q_proj": 0.00123,
    "model.layers.0.self_attention.k_proj": 0.00045
  },
  "statistics": {
    "total_layers": 100,
    "bit_distribution": {4: 90, 8: 10},
    "avg_sensitivity": 0.00078,
    "high_bit_ratio": 0.1
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

### 2. MoE 模型处理

对于 MoE 模型：
- MoE 组中的所有专家自动统一
- 推荐使用 `moe_unify_strategy="max"` 以保证精度
- `moe_unify_layer_level=True` 将共享专家与路由专家统一

### 3. 层选择

使用 `layer_include` 和 `layer_exclude` 可以：
- 聚焦于特定层类型的搜索
- 跳过已知不敏感的层
- 排除应保持全精度的输出层（lm_head）

### 4. 与 GPTQ 集成

位宽分配通过 `dynamic` 配置自动应用于 GPTQ 量化：
- 搜索结果更新 `quantize_config.dynamic`
- GPTQProcessor 在量化期间读取动态配置
- 无需手动干预

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
│  │   │   └─ MoEHandler.collect_moe_modules()               │
│  │   ├─ MixPrecisionSearchEngine.search()                   │
│  │   │   ├─ _run_baseline()                                 │
│  │   │   ├─ _compute_sensitivities()                        │
│  │   │   └─ _allocate_bits()                                │
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
├── __init__.py          # 模块导出
├── config.py            # 配置类
├── pipeline.py          # 主流水线入口
├── search_engine.py     # 搜索引擎实现
├── layer_selector.py    # 层选择逻辑
├── moe_handler.py       # MoE 结构处理
├── metrics.py           # 敏感度度量
└── README.md            # 本文件
```

## 参考文献

- [HAWQ-V3: Dyadic Neural Network Quantization](https://arxiv.org/abs/2011.10680)
- [Automatic Bit-width Selection for Quantization](https://arxiv.org/abs/2004.10568)
- [GPTQ: Accurate Post-Training Quantization](https://arxiv.org/abs/2210.17323)
