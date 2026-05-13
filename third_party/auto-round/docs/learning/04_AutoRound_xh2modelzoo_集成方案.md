# AutoRound → xh2modelzoo 集成方案

> 从 AutoRound 量化到 xh2modelzoo 部署的完整指南

---

## 1. 集成概览

### 1.1 目标

将 AutoRound（+ 可选 Hadamard 旋转）量化后的 Qwen3.5-27B 模型，通过 xhquant 导出流水线转换为 HMONNX 格式，部署到 xh2a 芯片。

### 1.2 完整流水线

```
                    AutoRound Pipeline
┌─────────────────────────────────────────────┐
│ 1. 加载 Qwen3.5-27B (完整 VLM)             │
│ 2. [可选] Hadamard 旋转 (含视觉 merger)    │
│ 3. AutoRound W4G64 量化                     │
│ 4. 保存为 transformers 兼容格式             │
└──────────────────┬──────────────────────────┘
                   ↓
                    xhquant Pipeline
┌─────────────────────────────────────────────┐
│ 5. transformers 加载量化模型                 │
│ 6. 反量化 → FP16/BF16                       │
│ 7. Graph 转换 (Prefill / Decode 分离)       │
│ 8. SSFP 4-bit 重量化 + SEFP 8-bit 激活量化 │
│ 9. HMONNX 导出                              │
│ 10. Golden 测试验证                          │
│ 11. 打包上传 modelzoo                        │
└─────────────────────────────────────────────┘
```

### 1.3 关键约束

1. 量化模型必须能被 `AutoModelForCausalLM.from_pretrained()` 正确加载
2. 量化层需要能正确反量化回 FP16
3. 模型的 config.json 必须完整正确
4. 视觉部分权重不能丢失

---

## 2. Step 1-4: AutoRound 量化与导出

### 2.1 推荐量化脚本

```python
#!/usr/bin/env python3
"""AutoRound + Rotate 量化 Qwen3.5-27B（保留完整VLM结构）"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from auto_round import AutoRound

# === 配置 ===
MODEL_PATH = "Qwen/Qwen3.5-27B"
OUTPUT_DIR = "./output/qwen35-27b-w4g64-rotated-autoround"
BITS = 4
GROUP_SIZE = 64
ITERS = 200
NSAMPLES = 128
SEQLEN = 2048
DATASET = "NeelNanda/pile-10k"
APPLY_ROTATION = True
ROTATE_VISION = "merger_only"  # none / merger_only / full

# === Step 1: 加载模型 ===
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

# === Step 2: [可选] 旋转 ===
if APPLY_ROTATION:
    from rotation_utils import apply_hadamard_rotation
    apply_hadamard_rotation(model, rotate_vision=ROTATE_VISION)

# === Step 3: AutoRound 量化 ===
ar = AutoRound(
    model=model,
    tokenizer=tokenizer,
    bits=BITS,
    group_size=GROUP_SIZE,
    iters=ITERS,
    nsamples=NSAMPLES,
    seqlen=SEQLEN,
    dataset=DATASET,
    enable_minmax_tuning=True,
)

# === Step 4: 保存 ===
model, folders = ar.quantize_and_save(
    output_dir=OUTPUT_DIR,
    format="auto_round"
)
print(f"量化模型已保存到: {folders}")
```

### 2.2 导出格式选择

对于 xh2modelzoo 集成，推荐使用 `auto_round` 或 `gptq` 格式：

| 格式 | 兼容性 | 反量化 | 推荐 |
|------|--------|--------|------|
| `auto_round` | 需 auto_round 包 | ✅ 支持 | ✅ 推荐 |
| `gptq` | 广泛兼容 | ✅ 支持 | ✅ 备选 |
| `fake` | transformers 原生 | 已反量化 | ⚠️ 无量化信息 |

---

## 3. Step 5-6: 量化模型加载与反量化

### 3.1 加载量化模型

```python
from transformers import AutoModelForCausalLM

# 加载 AutoRound 量化模型
model = AutoModelForCausalLM.from_pretrained(
    OUTPUT_DIR,
    device_map="auto",
    torch_dtype=torch.bfloat16,
)
```

### 3.2 反量化权重提取

```python
def extract_and_dequantize_weights(model):
    """从量化模型中提取权重并反量化"""
    dequantized_state_dict = {}
    quant_params = {}
    
    for name, module in model.named_modules():
        if hasattr(module, 'qweight'):
            # 提取量化参数
            qweight = module.qweight        # packed int4 weights
            scales = module.scales           # per-group scales
            zeros = module.qzeros           # per-group zero points
            g_idx = getattr(module, 'g_idx', None)
            
            # 保存量化参数
            quant_params[name] = {
                'qweight': qweight,
                'scales': scales,
                'zeros': zeros,
                'g_idx': g_idx,
                'bits': module.bits,
                'group_size': module.group_size,
            }
            
            # 反量化
            fp_weight = dequantize(qweight, scales, zeros, g_idx)
            dequantized_state_dict[name + '.weight'] = fp_weight
        else:
            # 非量化层直接保留
            for pname, param in module.named_parameters(recurse=False):
                dequantized_state_dict[name + '.' + pname] = param.data
    
    return dequantized_state_dict, quant_params
```

### 3.3 反量化函数

```python
def dequantize(qweight, scales, zeros, g_idx=None, bits=4):
    """将 packed int4 权重反量化为 FP16"""
    # 解包 int4 → int32
    unpacked = unpack_int4(qweight)  # [out_features, in_features]
    
    # 逐组反量化
    # weight_fp = (unpacked - zeros) * scales
    if g_idx is not None:
        # 使用 group index 映射
        weight_fp = (unpacked - zeros[g_idx]) * scales[g_idx]
    else:
        # 按连续分组
        weight_fp = (unpacked - zeros.repeat_interleave(group_size, dim=1)) \
                    * scales.repeat_interleave(group_size, dim=1)
    
    return weight_fp.half()
```

---

## 4. Step 7-11: xhquant 导出

### 4.1 导出脚本适配

需要修改 `qwen3_5_llm_xh2a_export.py` 以支持 AutoRound 量化模型输入：

```python
# 在 xhquant 导出脚本中添加 AutoRound 模型支持
def load_model(hf_model_dir, dtype):
    """加载模型（支持 AutoRound 量化格式）"""
    config = AutoConfig.from_pretrained(hf_model_dir)
    
    if hasattr(config, 'quantization_config'):
        quant_method = config.quantization_config.get('quant_method', '')
        if 'auto-round' in quant_method or 'gptq' in quant_method:
            # AutoRound/GPTQ 量化模型: 加载并反量化
            model = AutoModelForCausalLM.from_pretrained(
                hf_model_dir,
                device_map="auto",
                torch_dtype=getattr(torch, dtype),
            )
            # 反量化所有量化层
            dequantize_model_inplace(model)
            return model
    
    # 非量化模型: 直接加载
    return AutoModelForCausalLM.from_pretrained(
        hf_model_dir,
        torch_dtype=getattr(torch, dtype),
        device_map="auto",
    )
```

### 4.2 导出命令

```bash
python qwen3_5_llm_xh2a_export.py \
    --config configs/qwen3_5/qwen3_5_27b_instruct_xh2a.py \
    --hf_model_dir ./output/qwen35-27b-w4g64-rotated-autoround \
    --dtype bf16 \
    --valid \
    --golden \
    --upload
```

### 4.3 导出产物

```
release_dir/
├── prefill/
│   ├── hmquant_xh2_qwen3_5_27b_prefill_with_act.onnx
│   ├── hmquant_xh2_qwen3_5_27b_prefill_with_act_external_data
│   └── step_0/                     # Golden 数据
├── decode/
│   ├── hmquant_xh2_qwen3_5_27b_decode_with_act.onnx
│   ├── hmquant_xh2_qwen3_5_27b_decode_with_act_external_data
│   └── step_0/                     # Golden 数据
├── quant_embedding.pt              # 量化 Embedding
├── golden_meta_info.json           # Golden 元信息
└── export_meta_info.json           # 导出元信息
```

---

## 5. 视觉模块处理方案

### 5.1 设计原则

| 组件 | 旋转 | 量化 | xhquant 导出 |
|------|------|------|-------------|
| LLM backbone | ✅ Hadamard | ✅ W4G64 | ✅ SSFP 4-bit |
| Visual merger | ✅ 输出层必须旋转 | ❌ 全精度 | ❌ 不导出 |
| Vision encoder | ⚙️ 可选旋转 | ❌ 全精度 | ❌ 不导出 |
| LM head | ✅ 旋转 | ⚙️ 可选量化 | ✅ SEFP 8-bit |

### 5.2 视觉旋转配置

```python
# rotation_utils.py 中的配置选项
ROTATE_VISION_OPTIONS = {
    "none": "不旋转视觉部分（LLM 旋转时 merger 输出不匹配，仅用于纯 LLM 场景）",
    "merger_only": "仅旋转 merger 输出层（推荐：保证视觉-文本接口对齐）",
    "full": "旋转完整视觉编码器（可选：进一步平滑视觉特征分布）",
}
```

### 5.3 merger_only 实现

```python
def rotate_visual_merger_qwen3_5(model, Q, device="cuda"):
    """旋转 Qwen3.5 视觉 merger 输出层"""
    # Qwen3.5 使用 deepstack merger
    for merger in model.visual.merger.deepstack_merger_list:
        w = merger.linear_fc2.weight.data.to(device=device, dtype=torch.float64)
        merger.linear_fc2.weight.data = (Q.T @ w).to(w.dtype).to(w.device)
        if merger.linear_fc2.bias is not None:
            b = merger.linear_fc2.bias.data.to(device=device, dtype=torch.float64)
            merger.linear_fc2.bias.data = (Q.T @ b).to(b.dtype).to(b.device)
```

---

## 6. 端到端验证清单

### 6.1 量化阶段验证

- [ ] 量化模型 PPL 增幅 < 3%
- [ ] 量化模型对话质量无损
- [ ] 量化模型文件完整（config.json, safetensors, tokenizer）
- [ ] 量化模型包含视觉权重（如果是 VLM）

### 6.2 导出阶段验证

- [ ] transformers 能正确加载量化模型
- [ ] 反量化后权重与 FP16 参考值误差 < 1e-3
- [ ] HMONNX 模型能正常推理
- [ ] Golden 测试通过
- [ ] Prefill 和 Decode 分离正确

### 6.3 部署阶段验证

- [ ] 模型成功上传 modelzoo
- [ ] xh2a 芯片上推理正常
- [ ] 推理精度与 Golden 对齐

---

## 7. 风险与缓解措施

| 风险 | 影响 | 缓解措施 |
|------|------|---------|
| AutoRound 格式 xhquant 不识别 | 无法导出 | 使用 GPTQ 格式作为中间格式 |
| 视觉权重丢失 | VLM 推理失败 | 使用完整 VLM 加载路径 |
| 旋转数学等价性错误 | 输出偏移 | 旋转前后 logits 对比验证 |
| 显存不足 (27B 完整 VLM) | OOM | 使用 offload 策略 |
| GLIBC 版本不兼容 | 后端加载失败 | 使用纯 Python 后端 |
