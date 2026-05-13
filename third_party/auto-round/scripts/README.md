# Qwen3.5 AutoRound 量化脚本

基于 AutoRound 的 Qwen3.5-27B 量化工具，支持 4 种量化模式。

## 目录结构

```
scripts/
├── quantize.py           # 主量化脚本（4种模式）
├── rotation_utils.py     # Hadamard 旋转工具（旧版，已集成到 quantize.py）
├── e2e_verify.py         # 端到端验证（load→inference→dequant）
├── export_xh2.py         # xh2modelzoo 导出（反量化+量化参数提取）
└── eval/                 # 评估脚本
```

## 4 种量化模式

| 模式 | 旋转范围 | 量化范围 | 适用场景 |
|------|---------|---------|---------|
| `llm-only` | 无 | LLM W4G64 | 纯文本场景，最简单 |
| `rotate-merger` | LLM + merger输出层 | LLM W4G64 | 文本+视觉，激活友好 |
| `rotate-full` | LLM + 完整视觉 | LLM W4G64 | 全量旋转，视觉精度不变 |
| `rotate-quant-all` | LLM + 完整视觉 | 视觉W8G64 + LLM W4G64 | 最高压缩率 |

### 关键设计：

- **旋转不改变层类型**：LayerNorm 仍是 LayerNorm，RMSNorm 仍是 RMSNorm
- **旋转仅融合权重**：norm weight 被吸收到相邻 linear 层，norm 重置为恒等
- **merger 输出层必须同步旋转**：当 LLM 旋转时，merger.linear_fc2 必须旋转，否则视觉特征空间不匹配
- **保存格式**：默认 `auto_gptq`（GPTQModel 兼容），xh2modelzoo 可无缝加载

## 快速开始

```bash
# 模式 1: 只量化 LLM
python scripts/quantize.py --model /data01/datasets/Qwen3.5-27B \
    --mode llm-only --output_dir ./output/qwen35-w4g64

# 模式 2: 旋转 LLM + merger，量化 LLM（推荐）
python scripts/quantize.py --model /data01/datasets/Qwen3.5-27B \
    --mode rotate-merger --output_dir ./output/qwen35-rot-merger-w4g64

# 模式 3: 旋转全部，量化 LLM
python scripts/quantize.py --model /data01/datasets/Qwen3.5-27B \
    --mode rotate-full --output_dir ./output/qwen35-rot-full-w4g64

# 模式 4: 旋转全部，量化视觉 W8 + LLM W4
python scripts/quantize.py --model /data01/datasets/Qwen3.5-27B \
    --mode rotate-quant-all --output_dir ./output/qwen35-rot-quant-all \
    --vision_bits 8 --vision_group_size 64
```

## 验证与导出

```bash
# 端到端验证
CUDA_VISIBLE_DEVICES=0 python scripts/e2e_verify.py

# 导出到 xh2modelzoo（反量化到 BF16 + 保存量化参数）
python scripts/export_xh2.py \
    --quantized_model ./output/qwen35-w4g64 \
    --output_dir ./output/qwen35-xh2-export --dtype bf16
```

## 实验结论

| 方案 | PPL 增幅 | 对话质量 | 激活分布 | 推荐 |
|------|:---:|:---:|:---:|:---:|
| AutoRound W4G64 (pile-10k) | +2.07% | 无损 | 原始 | ✅ |
| **Rotate+AutoRound W4G64** | **+2.13%** | **无损** | **更均匀** | **⭐ 优选** |
| Rotate+RTN W4G64 | +3.31% | 无损 | 更均匀 | ❌ |

**优选**: Rotate+AutoRound (pile-10k) — PPL 仅多 0.06%，但激活量化友好
