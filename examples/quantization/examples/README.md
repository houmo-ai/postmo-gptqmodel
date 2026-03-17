# 量化示例脚本

本目录包含针对不同模型架构的 GPTQ 量化脚本。以下以 **Qwen3.5 Dense** 为例，介绍完整的校准数据准备、量化、推理验证和 PPL 评测流程。

---

## Qwen3.5 Dense 量化教程

脚本：`example_qwen35dense.py`

### 1. 准备校准数据

脚本支持两种校准数据来源：

#### 方式一：使用 wikitext2（默认）

不需要额外准备，脚本会自动从 HuggingFace 下载 `wikitext-2-raw-v1` 数据集，并按 `--seqlen` 过滤短文本。

#### 方式二：使用 IVSG 生成 JSONL 校准数据

项目提供了基于 IVSG（Importance-Variance Scored Generation）的校准数据生成脚本，位于：

```
gptqmodel/quantization/calibration/dense_ivsg/qwen3_5_dense_gen.py
```

该脚本利用模型自身通过 beam search 生成高质量校准文本。生成过程会从词表中随机采样起始 token，先做短 greedy prefix 再用 beam search 扩展，确保数据多样性和覆盖度。

**生成校准数据：**

```bash
cd gptqmodel/quantization/calibration/dense_ivsg

python qwen3_5_dense_gen.py \
    --model_path /path/to/Qwen3.5-27B \
    --nsamples 256 \
    --seqlen 512 \
    --num_beams 4 \
    --min_new_tokens 300 \
    --gen_dir ./gen_data
```

生成完成后，输出文件为 `gen_data/Qwen3.5-27B.jsonl`（文件名取自模型目录名），格式如下：

```jsonl
{"text": "生成的校准文本第一条..."}
{"text": "生成的校准文本第二条..."}
```

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--model_path` | — | FP 模型路径 |
| `--nsamples` | 100 | 生成样本数量 |
| `--seqlen` | 512 | 每条文本最大序列长度 |
| `--num_beams` | 4 | beam search 宽度 |
| `--min_new_tokens` | 300 | 每条最少生成 token 数 |
| `--gen_dir` | gen_data | JSONL 输出目录 |
| `--seed` | 42 | 随机种子 |

#### 方式三：使用已有 JSONL 文件

也可以直接使用任意 JSONL 文件，每行一个 JSON 对象，包含一个文本字段：

```jsonl
{"text": "这是第一条校准文本，内容应该尽量涵盖模型的典型使用场景。"}
{"text": "GPTQ quantization reduces model size by approximating weights with low-bit representations."}
```

通过 `--calibration-jsonl` 指定文件路径，通过 `--calibration-text-key` 指定字段名（默认 `text`）。

**校准数据建议：**
- 样本数量：建议 128~512 条（`--nsamples`），越多越好但耗时越长
- 文本长度：建议每条至少 1024 tokens（`--seqlen`）
- 数据分布：尽量覆盖模型实际应用场景（中英文混合、代码、对话等）

### 2. 基础量化

使用 wikitext2 校准数据进行 4-bit 量化：

```bash
python example_qwen35dense.py \
    --model /path/to/Qwen3.5-27B \
    --out /path/to/Qwen3.5-27B-gptq-4bit \
    --bits 4 \
    --group-size 64 \
    --nsamples 256 \
    --device cuda:0
```

使用自定义 JSONL 校准数据（如 IVSG 生成的数据）：

```bash
python example_qwen35dense.py \
    --model /path/to/Qwen3.5-27B \
    --out /path/to/Qwen3.5-27B-gptq-4bit \
    --bits 4 \
    --group-size 64 \
    --nsamples 256 \
    --calibration-jsonl gptqmodel/quantization/calibration/dense_ivsg/gen_data/Qwen3.5-27B.jsonl \
    --device cuda:0
```

### 3. 带 Hadamard 旋转的量化

QuaRot 旋转可以在量化前对权重做正交变换，减少量化误差：

```bash
python example_qwen35dense.py \
    --model /path/to/Qwen3.5-27B \
    --out /path/to/Qwen3.5-27B-gptq-4bit-hadamard \
    --bits 4 \
    --group-size 64 \
    --rotation hadamard \
    --device cuda:0
```

### 4. 评测旋转前后的 PPL

在量化前评估旋转操作本身对模型精度的影响：

```bash
python example_qwen35dense.py \
    --model /path/to/Qwen3.5-27B \
    --out /path/to/Qwen3.5-27B-gptq-4bit-hadamard \
    --bits 4 \
    --rotation hadamard \
    --check-rotation-ppl \
    --device cuda:0
```

输出示例：
```
[rotation] FP model avg PPL before rotation: 8.1234
[rotation] Applying hadamard rotation to fp model ...
[rotation] FP model avg PPL after rotation: 8.1240
[rotation] PPL delta: +0.000600
```

### 5. 仅做旋转不量化

如果只想评估旋转效果或保存旋转后的 FP 模型：

```bash
python example_qwen35dense.py \
    --model /path/to/Qwen3.5-27B \
    --out /path/to/unused \
    --rotation hadamard \
    --check-rotation-ppl \
    --rotated-out /path/to/Qwen3.5-27B-rotated \
    --stop-after-rotate \
    --device cuda:0
```

### 6. 跳过量化直接推理和评测

如果已有量化模型，只想运行推理和 PPL 评测：

```bash
python example_qwen35dense.py \
    --out /path/to/Qwen3.5-27B-gptq-4bit \
    --skip-quantize \
    --infer-device cuda:0
```

### 7. 多卡推理

```bash
python example_qwen35dense.py \
    --out /path/to/Qwen3.5-27B-gptq-4bit \
    --skip-quantize \
    --device-map auto
```

### 完整参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--model` | — | FP 模型路径 |
| `--out` | — | 量化模型输出目录 |
| `--bits` | 4 | 量化位宽（2/3/4/5/8） |
| `--group-size` | 64 | 量化分组大小 |
| `--rotation` | None | QuaRot 旋转模式（hadamard/block_hadamard/random） |
| `--check-rotation-ppl` | False | 评测旋转前后 wikitext2 PPL |
| `--rotated-out` | None | 保存旋转后 FP 模型的目录 |
| `--stop-after-rotate` | False | 仅旋转不量化 |
| `--hessian-mse` | True | 启用 Hessian MSE 辅助优化 |
| `--nsamples` | 256 | 校准样本数量 |
| `--seqlen` | 1024 | wikitext2 最小文本长度过滤 |
| `--calibration-jsonl` | None | 自定义 JSONL 校准数据路径 |
| `--calibration-text-key` | text | JSONL 中文本字段名 |
| `--batch-size` | 1 | 量化 batch size |
| `--device` | auto | 量化计算设备 |
| `--infer-device` | None | 推理设备（默认跟随 --device） |
| `--device-map` | None | 多卡策略（auto/balanced/balanced_low_0/sequential） |
| `--offload-to-disk` | True | 启用磁盘 offload |
| `--max-new-tokens` | 128 | 生成最大 token 数 |
| `--prompt` | "请介绍一下 GPTQ 量化的核心思想。" | 推理验证的 prompt |
| `--skip-quantize` | False | 跳过量化直接推理 |
| `--seed` | 42 | 随机种子 |

---

## 其他模型脚本

| 脚本 | 模型 |
|------|------|
| `example_qwen35moe.py` | Qwen3.5 MoE |
| `example_qwen3_next.py` | Qwen3-Next (MoE + linear attention) |
| `example_qwen3_8b.py` | Qwen3-8B |
| `example_qwen3_moe.py` | Qwen3 MoE |
| `example_glm_4_7_flash.py` | GLM-4-7B Flash |
| `example_gptoss.py` | GPTOSS |
| `example_deepseekocr.py` | DeepSeek OCR |
| `example_qwen_vl.py` | Qwen-VL |
