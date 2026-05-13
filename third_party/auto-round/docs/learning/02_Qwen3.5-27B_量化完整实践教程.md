# Qwen3.5-27B AutoRound 量化完整实践教程

> 4 种量化模式：从模型加载到量化、旋转、保存、加载推理的完整流程

---

## 目录

1. [环境准备](#1-环境准备)
2. [模型加载](#2-模型加载)
3. [4 种量化模式](#3-4-种量化模式)
4. [量化模型保存（GPTQModel 格式）](#4-量化模型保存gptqmodel-格式)
5. [量化模型加载与推理](#5-量化模型加载与推理)
6. [精度评估](#6-精度评估)
7. [视觉模块旋转详解](#7-视觉模块旋转详解)
8. [xh2modelzoo 导出](#8-xh2modelzoo-导出)
9. [常见问题与解决方案](#9-常见问题与解决方案)

---

## 1. 环境准备

### 1.1 安装依赖

```bash
# 激活环境
conda activate xhquant  # torch 2.8.0+cu128, transformers 5.3.0

# AutoRound
cd /data01/home/yujy/work/auto-round
pip install -e .

# 评估工具
pip install lm-eval
```

### 1.2 模型路径

```python
MODEL_PATH = "/data01/datasets/Qwen3.5-27B"
# 模型结构: Qwen3_5ForConditionalGeneration (VLM)
# LLM: Qwen3_5ForCausalLM, 64 layers, hidden_size=5120
# 视觉: ViT encoder (LayerNorm) + Merger (LayerNorm → Linear → GELU → Linear)
# 注意力类型: self_attn (full attention) + linear_attn (linear/recurrent)
```

### 1.3 GLIBC 兼容性（重要）

系统 GLIBC 2.31，marlin/exllama 后端需要 2.32。**必须在 import transformers 之前修复**：

```python
from auto_round.inference.backend import BackendInfos
to_remove = [k for k in list(BackendInfos.keys()) if "marlin" in k or "exllama" in k]
for k in to_remove:
    del BackendInfos[k]
```

---

## 2. 模型加载

### 2.1 完整 VLM 方式（推荐 — 保留视觉部分）

```python
from transformers import AutoModelForVision2Seq, AutoTokenizer
import torch

model = AutoModelForVision2Seq.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
# 实际类型: Qwen3_5ForConditionalGeneration
# 包含: model.visual (ViT) + model.language_model (CausalLM)
```

### 2.2 纯 CausalLM 方式（仅 LLM 部分）

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True
)
# 实际类型: Qwen3_5ForCausalLM
# 只有 LLM 部分，无视觉编码器
```

**注意**: CausalLM 方式会丢失视觉编码器。量化模型如需保留完整结构，应使用 VLM 方式加载。

---

## 3. 4 种量化模式

统一脚本 `scripts/quantize.py` 支持 4 种模式：

| 模式 | 旋转范围 | 量化范围 | 适用场景 |
|------|---------|---------|---------|
| `llm-only` | 无 | LLM W4G64 | 纯文本，最简单 |
| `rotate-merger` | LLM + merger输出层 | LLM W4G64 | **推荐** — 激活友好，视觉精度不变 |
| `rotate-full` | LLM + 完整视觉 | LLM W4G64 | 全量旋转，视觉也均匀化 |
| `rotate-quant-all` | LLM + 完整视觉 | 视觉W8G64 + LLM W4G64 | 最高压缩率 |

### 3.1 模式 1: llm-only — 只量化 LLM

```bash
python scripts/quantize.py \
    --model /data01/datasets/Qwen3.5-27B \
    --mode llm-only \
    --bits 4 --group_size 64 \
    --iters 200 --nsamples 128 --seqlen 2048 \
    --format auto_gptq \
    --output_dir ./output/qwen35-w4g64
```

**流程**: 加载模型 → AutoRound W4G64 量化 LLM → 保存（GPTQModel 格式）

### 3.2 模式 2: rotate-merger — 旋转 LLM + merger，量化 LLM（推荐）

```bash
python scripts/quantize.py \
    --model /data01/datasets/Qwen3.5-27B \
    --mode rotate-merger \
    --bits 4 --group_size 64 \
    --iters 200 --nsamples 128 --seqlen 2048 \
    --format auto_gptq \
    --output_dir ./output/qwen35-rot-merger-w4g64
```

**流程**:
1. 加载完整 VLM 模型
2. 融合 LLM RMSNorm（`(1+weight)*x` → 吸收 weight 到后续 linear）
3. 生成 Hadamard 旋转矩阵 Q（5120×5120）
4. 旋转 LLM 所有 self_attn/linear_attn/MLP 层的权重
5. **同步旋转 merger.linear_fc2.weight**（`Q.T @ W`）
6. AutoRound W4G64 量化 LLM 部分
7. 保存为 GPTQModel 兼容格式

### 3.3 模式 3: rotate-full — 旋转全部，量化 LLM

```bash
python scripts/quantize.py \
    --model /data01/datasets/Qwen3.5-27B \
    --mode rotate-full \
    --bits 4 --group_size 64 \
    --output_dir ./output/qwen35-rot-full-w4g64
```

**流程**: 同模式 2 + 额外旋转视觉编码器（融合 ViT LayerNorm + 旋转 patch_embed/blocks/merger）

### 3.4 模式 4: rotate-quant-all — 旋转全部 + 量化视觉 + 量化 LLM

```bash
python scripts/quantize.py \
    --model /data01/datasets/Qwen3.5-27B \
    --mode rotate-quant-all \
    --bits 4 --group_size 64 \
    --vision_bits 8 --vision_group_size 64 \
    --output_dir ./output/qwen35-rot-quant-all
```

**流程**: 同模式 3 + 视觉部分 W8G64 量化（通过 AutoRound 的 `layer_config` 对视觉层设置 8-bit）

### 3.5 量化时间参考

| 阶段 | 时间 |
|------|------|
| 模型加载 | ~5 min |
| Hadamard 旋转（含融合） | ~4 min |
| 校准数据准备 | ~2 min |
| Block-wise 量化 (64 blocks × 200 iters) | ~2.5 hours |
| 权重打包 + 保存 | ~5 min |
| **总计（含旋转）** | **~2.8 hours** |

---

## 4. 量化模型保存（GPTQModel 格式）

### 4.1 保存格式

使用 `format="auto_gptq"` 保存，生成标准 GPTQModel 兼容格式：

```
output_dir/
├── config.json                    # HF config + quantization_config
├── quantize_config.json           # GPTQModel 标准量化配置
├── model-00001-of-00005.safetensors  # 量化权重 (qweight, scales, zeros)
├── model.safetensors.index.json   # 分片索引
├── tokenizer.json
├── tokenizer_config.json
├── special_tokens_map.json
└── generation_config.json
```

### 4.2 quantize_config.json 格式

```json
{
  "bits": 4,
  "group_size": 64,
  "desc_act": false,
  "sym": false,
  "quant_method": "gptq",
  "checkpoint_format": "gptq",
  "model_file_base_name": "model"
}
```

### 4.3 为什么选择 auto_gptq 格式

1. **xh2modelzoo 兼容**: `GPTQModel.from_quantized()` 可直接加载
2. **transformers 兼容**: `AutoModelForCausalLM.from_pretrained()` 自动检测
3. **标准化**: qweight/scales/zeros 存储格式是 GPTQ 社区标准
4. **后端丰富**: 支持 tritonv2、exllamav2、marlin 等多种推理后端

---

## 5. 量化模型加载与推理

### 5.1 加载（需先修复 GLIBC）

```python
# Step 0: 修复 GLIBC 兼容性（必须在 import transformers 之前）
from auto_round.inference.backend import BackendInfos
to_remove = [k for k in list(BackendInfos.keys()) if "marlin" in k or "exllama" in k]
for k in to_remove: del BackendInfos[k]

# Step 1: 加载量化模型
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model = AutoModelForCausalLM.from_pretrained(
    "./output/qwen35-w4g64",
    device_map="auto",
    torch_dtype=torch.bfloat16
)
tokenizer = AutoTokenizer.from_pretrained("./output/qwen35-w4g64")
```

### 5.2 推理

```python
prompt = "请解释量子计算的基本原理"
messages = [{"role": "user", "content": prompt}]
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(text, return_tensors="pt").to(model.device)

outputs = model.generate(
    **inputs,
    max_new_tokens=512,
    temperature=0.7,
    top_p=0.9,
    do_sample=True
)
response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
print(response)
```

### 5.3 验证量化层

```python
# 检查量化层数量
quant_count = 0
for name, module in model.named_modules():
    if hasattr(module, 'qweight'):
        quant_count += 1
print(f"量化层数量: {quant_count}")  # 预期: ~400
```

---

## 6. 精度评估

### 6.1 PPL 评估 (WikiText-2)

```bash
python scripts/eval/eval_quant_ppl.py \
    --model_path ./output/qwen35-w4g64 \
    --eval_dataset wikitext2
```

### 6.2 对话质量评估

```bash
# 基础对话评估 (20 场景)
python scripts/eval/eval_chat_20.py \
    --model_path ./output/qwen35-w4g64

# Think 能力评估
python scripts/eval/eval_chat_think.py \
    --model_path ./output/qwen35-w4g64
```

### 6.3 预期结果

| 指标 | BF16 | AutoRound W4G64 | Rotate+AutoRound W4G64 |
|------|:---:|:---:|:---:|
| PPL (wikitext-2) | 7.5335 | 7.6891 (+2.07%) | 7.6937 (+2.13%) |
| 对话质量 | 基准 | 无损 | 无损 |
| Think tokens | 基准 | -13.1% | +6.7% |

---

## 7. 视觉模块旋转详解

### 7.1 旋转原理

Hadamard 旋转通过正交矩阵 Q 变换权重分布：
- 使权重分布更均匀，减少量化误差的离群值
- 使激活分布更适合量化（芯片部署的激活量化友好）
- PPL 代价仅 ~0.06%

### 7.2 旋转不改变层类型（关键发现）

**旋转仅修改权重数值，不改变层类型或模型结构**：
- LayerNorm → 仍是 LayerNorm（权重被吸收后重置为 `weight=1, bias=0`）
- RMSNorm → 仍是 RMSNorm（权重被吸收后重置为 `weight=0`）
- 模型 save/load 完全正常，无需任何特殊处理

### 7.3 Norm 融合细节

| Norm 类型 | 缩放公式 | 融合操作 | 重置操作 |
|-----------|---------|---------|---------|
| Qwen3.5 RMSNorm | `(1+w)*x/rms(x)` | `scale = 1 + weight` | `weight.zero_()` |
| 标准 LayerNorm | `w*x/std(x) + b` | `scale = weight` | `weight.fill_(1.0), bias.zero_()` |
| ViT LayerNorm | 同标准 LayerNorm | `scale = weight` | `weight.fill_(1.0), bias.zero_()` |

融合后，scale 被乘到后续 linear 层的权重上：`W_new = W_old * diag(scale)`

### 7.4 Merger 同步旋转（必须）

当 LLM 被旋转后，LLM 的第一层输入期望的是旋转后的特征空间。视觉 merger 的输出直接送入 LLM，因此：

```python
# merger.linear_fc2 是视觉→LLM 的连接层
# 其输出必须在旋转后的 LLM 输入空间中
w = model.visual.merger.linear_fc2.weight.data  # [hidden_size, merger_out]
model.visual.merger.linear_fc2.weight.data = (Q_llm.T @ w.float()).to(w.dtype)
```

**不旋转 merger 会导致**: 视觉特征与 LLM 输入空间不匹配，视觉理解能力完全失效。

### 7.5 完整视觉旋转（可选）

视觉编码器使用独立的旋转矩阵 Q_vision（因为 vision_hidden_size ≠ llm_hidden_size）：

1. 融合 ViT 的 LayerNorm
2. 旋转 patch_embed → blocks → merger 的权重
3. merger 最终输出层仍需用 Q_llm.T 旋转

```python
# 视觉旋转使用独立矩阵
Q_vision = random_hadamard_matrix(vision_hidden_size, device)
Q_llm = random_hadamard_matrix(llm_hidden_size, device)
# 视觉内部用 Q_vision，merger 输出用 Q_llm.T
```

---

## 8. xh2modelzoo 导出

### 8.1 导出流水线

```
AutoRound 量化模型 (auto_gptq 格式)
  → transformers / GPTQModel 加载
  → 反量化到 BF16
  → xhquant Graph 转换 → Prefill/Decode 分离
  → SSFP 4-bit 重量化 → HMONNX 导出
  → Golden 验证 → 打包上传 modelzoo
```

### 8.2 反量化（手动 GPTQ 格式解包）

```python
# 解包 int32 → int4
pack_factor = 32 // bits  # 8 for 4-bit
wf = torch.arange(0, 32, bits)  # [0, 4, 8, 12, 16, 20, 24, 28]
weight_unpacked = (qweight.unsqueeze(1) >> wf.unsqueeze(-1)) & ((1 << bits) - 1)

# 反量化
weight_fp = (w_float - zeros[g_idx]) * scales[g_idx]
```

### 8.3 使用导出脚本

```bash
python scripts/export_xh2.py \
    --quantized_model ./output/qwen35-w4g64 \
    --output_dir ./output/qwen35-xh2-export \
    --dtype bf16
```

输出: 反量化 BF16 模型 + 量化参数目录（qweight/scales/zeros per layer）

### 8.4 已验证结果

- 400 层成功反量化
- 51GB BF16 模型（12 shards）
- 401 个量化参数目录

---

## 9. 常见问题与解决方案

### Q1: GLIBC_2.32 加载失败

**原因**: gptqmodel/marlin/exllama 后端需要 GLIBC_2.32，系统只有 2.31。

**解决**: 在 import transformers 之前删除不兼容后端（见 §1.3），并设置 `backend=auto_round:tritonv2`。

### Q2: Qwen3.5 batch_size 被限制为 1

**原因**: 已注册为 MLLM，`mllms_with_limited_bs` 限制。

**解决**: 正常行为。MLLMCompressor 路径需要 batch_size=1，量化速度稍慢但结果正确。

### Q3: 旋转后 save/load 有问题吗？

**答**: 没有。旋转只修改权重数值，不改变模型结构或层类型。标准 `save_pretrained` / `from_pretrained` 正常工作。

### Q4: 自生成数据效果差

**原因**: 自生成数据针对 GPTQ 的 Hessian-MSE 优化，不适合 SignSGD。

**解决**: 始终使用 `NeelNanda/pile-10k`（PPL +2.07% vs 自生成 +2.36%）。

### Q5: 量化后缺少视觉部分

**原因**: 使用 CausalLM 加载丢失了视觉编码器。

**解决**: 使用 VLM 方式加载（`AutoModelForVision2Seq`），量化 LLM 后保存时包含完整视觉权重。

### Q6: 如何选择量化模式？

| 场景 | 推荐模式 |
|------|---------|
| 纯文本部署 | `llm-only` |
| VLM 部署 + 芯片激活量化 | `rotate-merger` ⭐ |
| VLM 部署 + 极致均匀化 | `rotate-full` |
| VLM 部署 + 最大压缩 | `rotate-quant-all` |
