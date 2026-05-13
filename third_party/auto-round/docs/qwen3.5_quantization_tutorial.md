# Qwen3.5-9B W4A16 量化教程 (auto-round)

本教程完整演示如何使用 [auto-round](https://github.com/intel/auto-round) 对 **Qwen3.5-9B** 进行 W4A16 量化，并对比量化前后的精度（PPL + 对话质量）。

## 目录

- [环境准备](#环境准备)
- [第一步：BF16 基线评测](#第一步bf16-基线评测)
- [第二步：W4A16 量化](#第二步w4a16-量化)
- [第三步：量化模型评测](#第三步量化模型评测)
- [评测结果对比](#评测结果对比)
- [注意事项](#注意事项)

---

## 环境准备

### 硬件要求

| 资源 | 最低要求 | 推荐 |
|---|---|---|
| GPU | 1× A100-40GB | 1× A100-80GB |
| RAM | 32GB | 64GB |
| 磁盘 | 50GB 可用空间 | 100GB |

### 安装依赖

```bash
# 创建 conda 环境 (也可使用已有环境)
conda create -n autoround python=3.12 -y
conda activate autoround

# 安装 auto-round (从源码安装，支持最新模型)
cd /path/to/auto-round
pip install -e .

# 安装 PPL 评测依赖
pip install lm_eval
```

### 验证安装

```bash
python -c "import auto_round; print(auto_round.__version__)"
# 预期输出: 0.12.0.dev301 或更新版本
```

---

## 第一步：BF16 基线评测

在量化之前，先评测 BF16 原始模型作为精度基线。

### 1.1 PPL 评测 (wikitext)

```bash
CUDA_VISIBLE_DEVICES=0 auto-round \
  --model weights/Qwen3.5-9B \
  --eval \
  --tasks wikitext \
  --eval_bs 4
```

### 1.2 对话质量评测

```bash
CUDA_VISIBLE_DEVICES=0 python eval_chat.py \
  --model weights/Qwen3.5-9B \
  --device cuda:0
```

> `eval_chat.py` 脚本会用 4 个中文问题测试模型的基础质量（知识问答、数学、写作、编程）。
> 注意：Qwen3.5 默认启用思考模式，脚本中使用 `enable_thinking=False` 禁用以获得简洁回复。

---

## 第二步：W4A16 量化

使用 auto-round CLI 进行量化。核心参数：

| 参数 | 值 | 说明 |
|---|---|---|
| `--bits` | 4 | 权重量化位数 |
| `--group_size` | 128 | 量化分组大小 |
| `--sym` | (默认启用) | 对称量化 |
| `--iters` | 200 (默认) | 优化迭代次数 |
| `--format` | auto_round | 导出格式 |

### 执行量化

```bash
CUDA_VISIBLE_DEVICES=0 auto-round \
  --model weights/Qwen3.5-9B \
  --bits 4 \
  --group_size 128 \
  --format auto_round \
  --output_dir weights/Qwen3.5-9B-W4A16-autoround
```

**实测数据** (1× A100-80GB):

| 指标 | 数值 |
|---|---|
| 量化时间 | ~52 分钟 |
| 峰值 VRAM | 21.69 GB |
| 峰值 RAM | 27.97 GB |
| 量化层数 | 248 / 359 层 (visual 部分不量化) |
| 模型大小 | 37 GB → 8.2 GB (压缩 4.5×) |

量化完成后，模型保存在 `weights/Qwen3.5-9B-W4A16-autoround/Qwen3.5-9B-w4g128/` 目录。

---

## 第三步：量化模型评测

### 3.1 PPL 评测

```bash
CUDA_VISIBLE_DEVICES=0 python eval_quant_ppl.py \
  --model weights/Qwen3.5-9B-W4A16-autoround/Qwen3.5-9B-w4g128 \
  --device cuda:0
```

### 3.2 对话质量评测

```bash
CUDA_VISIBLE_DEVICES=0 python eval_chat.py \
  --model weights/Qwen3.5-9B-W4A16-autoround/Qwen3.5-9B-w4g128 \
  --device cuda:0 \
  --quantized
```

> **`--quantized` 标志是必需的**，详见下方 [注意事项](#qwen35-vlm-架构加载问题)。

---

## 评测结果对比

### PPL (wikitext, 越低越好)

| 指标 | BF16 基线 | W4A16 量化 | 变化 |
|---|---|---|---|
| word_perplexity | **9.1964** | **9.4279** | +0.23 (+2.5%) |
| byte_perplexity | 1.5143 | 1.5213 | +0.007 |
| bits_per_byte | 0.5986 | 0.6053 | +0.007 |

### 对话质量

| 问题 | BF16 | W4A16 | 判定 |
|---|---|---|---|
| "中国最长的河流是什么？" | ✅ 正确 (长江) | ✅ 正确 (长江) | 一致 |
| "解方程 2x + 5 = 15" | ✅ x = 5 | ✅ x = 5 | 一致 |
| "用100字描述春天的景色" | ✅ 流畅优美 | ✅ 流畅优美 (措辞略异) | 等效 |
| "写斐波那契 Python 函数" | ✅ 代码正确 | ✅ 代码正确 | 一致 |

### 结论

- **PPL 损失仅 2.5%**，几乎可以忽略
- **对话质量无明显退化**，量化模型回复同样准确流畅
- **模型体积压缩 4.5×** (37 GB → 8.2 GB)

---

## 注意事项

### Qwen3.5 VLM 架构加载问题

Qwen3.5 使用 `Qwen3_5ForConditionalGeneration`（VLM 多模态架构），包含 `visual` + `language_model` 两个子模块。这带来一个关键问题：

**问题**: `AutoModelForCausalLM.from_pretrained()` 会将模型加载为 `Qwen3_5ForCausalLM`（纯文本子模型），该类使用嵌套的 `text_config`，导致顶层的 `quantization_config` 被忽略，量化层替换不被触发。

**解决方案**: 量化模型必须用原始架构类加载：

```python
# ✅ 正确方式
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForConditionalGeneration
model = Qwen3_5ForConditionalGeneration.from_pretrained(
    "weights/Qwen3.5-9B-W4A16-autoround/Qwen3.5-9B-w4g128",
    device_map="cuda:0",
    trust_remote_code=True,
)

# ❌ 错误方式 — 量化信息丢失，实际加载为 FP16 权重
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained(...)
```

### Qwen3.5 思考模式

Qwen3.5 默认启用 thinking 模式，生成时会先输出 `<think>...</think>` 标签中的推理过程。使用 `apply_chat_template` 时通过 `enable_thinking=False` 禁用：

```python
text = tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True,
    enable_thinking=False  # 禁用思考模式
)
```

### 自定义量化参数

更多参数选项参见 [auto-round 文档](https://github.com/intel/auto-round)：

```bash
auto-round \
  --model weights/Qwen3.5-9B \
  --bits 4 \
  --group_size 128 \
  --iters 400 \          # 更多迭代可能提升精度
  --nsamples 512 \       # 更多校准样本
  --seqlen 2048 \        # 更长序列长度
  --format auto_round \
  --output_dir weights/Qwen3.5-9B-W4A16-autoround-custom
```

---

## 评测脚本说明

| 脚本 | 用途 |
|---|---|
| `eval_chat.py` | 中文对话质量评测，支持 BF16 和量化模型 |
| `eval_quant_ppl.py` | 量化模型 PPL 评测 (使用 lm_eval + wikitext) |

BF16 模型的 PPL 可通过 `auto-round --eval --tasks wikitext` 直接评测，无需额外脚本。
