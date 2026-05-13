# Qwen3.5-MoE AutoRound 量化脚本

基于 AutoRound 的 Qwen3.5-35B-A3B MoE 量化工具，参考 `scripts/` 目录补齐 4 种运行模式，并统一支持模型路径透传。

所有 `run_mode*.sh` 都支持：
- 第 1 个位置参数：模型路径
- 第 2 个位置参数：输出目录
- 第 3 个及之后的位置参数：继续透传给 `quantize_moe.py`，便于覆盖 batch size、低显存模式等高级参数
- 如果当前激活环境里的 `python` 无法正常初始化 CUDA，会自动回退到 `AR_FALLBACK_PYTHON`，否则尝试本机常用的 `xhquant` 环境解释器；也可手动指定 `AR_PYTHON`

其中 `run_mode*.sh` 默认都会：
- 设置 `CUDA_VISIBLE_DEVICES=0,1`
- 传入 `--device_map balanced`
- 开启 `--low_gpu_mem_usage`

`balanced` 现在默认走 **AutoRound 的 block-wise 多卡均衡映射**：
- 会把当前可见 GPU 转成 AutoRound 的多卡 `device_map`（例如 `0,1`）
- 让每个待量化 block 在量化时再按显存/层大小做均衡拆分
- 相比“前 20 层在卡 0、后 20 层在卡 1”的整层静态切分，更容易把两张卡都压满

如果你确实想保留旧的整层静态切分，可显式写成：

```bash
AR_DEVICE_MAP=balanced:manual bash scripts_qwen35moe/run_mode1_llm_only.sh ...
```

`balanced:manual` 的行为是：
- `embed_tokens` 放到 `cuda:0`
- `layers.0-19` 放到 `cuda:0`
- `layers.20-39` 放到 `cuda:1`
- `norm + lm_head` 放到 `cuda:1`

如需覆盖，可自行设置环境变量：

```bash
CUDA_VISIBLE_DEVICES=2,3 AR_DEVICE_MAP=balanced AR_LOW_GPU_MEM_USAGE=1 \
  bash scripts_qwen35moe/run_mode1_llm_only.sh ...
```

## 目录结构

```text
scripts_qwen35moe/
├── quantize_moe.py               # 主量化脚本（4种模式）
├── rotation_utils_moe.py         # MoE + Vision Hadamard 旋转工具
├── run_mode1_llm_only.sh         # Mode 1: 不旋转，只量化 LLM（当前推荐）
├── run_mode2_rotate_merger.sh    # Mode 2: 旋转 merger + LLM，量化 LLM
├── run_mode3_rotate_full.sh      # Mode 3: 全旋转，量化 LLM
├── run_mode4_rotate_quant_all.sh # Mode 4: 全旋转，量化 Vision+LLM
├── run_eval.sh                   # PPL 评测脚本（支持模型/量化目录传参）
└── eval/                         # 评测脚本
```

## 4 种量化模式

| 模式 | 旋转范围 | 量化范围 | 适用场景 |
|------|---------|---------|---------|
| `llm-only` | 无 | MoE LLM W4G64 | 当前实验最优，不旋转 |
| `rotate-merger` | MoE LLM + merger输出层 | MoE LLM W4G64 | 对齐视觉 merger 到 LLM 旋转空间 |
| `rotate-full` | MoE LLM + 完整视觉 | MoE LLM W4G64 | 全量旋转，视觉保持全精度 |
| `rotate-quant-all` | MoE LLM + 完整视觉 | 视觉 W8G64 + MoE LLM W4G64 | 最高压缩率 |

## 快速开始

```bash
# Mode 1：当前推荐，不旋转
bash scripts_qwen35moe/run_mode1_llm_only.sh \
  /data01/datasets/Qwen3.5-35B-A3B \
  ./output/Qwen3.5-35B-A3B-mode1-llm-only

# 低显存调参示例：进一步把 batch_size 降到 1
bash scripts_qwen35moe/run_mode1_llm_only.sh \
  /data01/datasets/Qwen3.5-35B-A3B \
  ./output/Qwen3.5-35B-A3B-mode1-llm-only \
  --batch_size 1 \
  --gradient_accumulate_steps 8

# 如果想临时关闭 low_gpu_mem_usage（默认已开启）
AR_LOW_GPU_MEM_USAGE=0 bash scripts_qwen35moe/run_mode1_llm_only.sh \
  /data01/datasets/Qwen3.5-35B-A3B \
  ./output/Qwen3.5-35B-A3B-mode1-llm-only

# Mode 2：仅旋转 merger 输出 + MoE LLM
bash scripts_qwen35moe/run_mode2_rotate_merger.sh /data01/datasets/Qwen3.5-35B-A3B

# Mode 3：全旋转但只量化 LLM
bash scripts_qwen35moe/run_mode3_rotate_full.sh /data01/datasets/Qwen3.5-35B-A3B

# Mode 4：全旋转并量化 Vision + LLM
bash scripts_qwen35moe/run_mode4_rotate_quant_all.sh /data01/datasets/Qwen3.5-35B-A3B
```

也可以直接调用主脚本：

```bash
python scripts_qwen35moe/quantize_moe.py \
  --model /data01/datasets/Qwen3.5-35B-A3B \
  --mode llm-only \
  --output_dir ./output/Qwen3.5-35B-A3B-mode1-llm-only
```

## 评测

```bash
bash scripts_qwen35moe/run_eval.sh \
  cuda:0 \
  /data01/datasets/Qwen3.5-35B-A3B \
  ./output/Qwen3.5-35B-A3B-mode1-llm-only
```
