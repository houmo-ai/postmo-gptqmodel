#!/bin/bash
# No Rotate: Quantize LLM with Attn/Shared Expert W8
# 适用场景: 不做旋转，Attention / Shared Expert 用 W8，其余 LLM 保持 W4
#
# 操作步骤:
#   1. 直接加载原始 Qwen3.6-MoE 模型
#   2. 不做 Vision / merger / LLM Hadamard 旋转
#   3. LLM 默认 W4G64，仅 Attention / Shared Expert 覆盖为 W8G64

set -euo pipefail

MODEL_PATH="${MODEL_PATH:-/data01/datasets/Qwen3.6-35B-A3B}"
OUTPUT_DIR="./output/qwen36moe-no-rotate-attn8-shared8"
LOG_DIR="./output/logs"
BATCH_SIZE="${BATCH_SIZE:-1}"

mkdir -p "${LOG_DIR}"

echo "=========================================="
echo "No Rotate: Quantize Qwen3.6-MoE LLM (Attn/Shared Expert W8, rest W4)"
echo "Model: ${MODEL_PATH}"
echo "Output: ${OUTPUT_DIR}"
echo "=========================================="

python scripts_qwen35moe/quantize_moe.py \
    --model "${MODEL_PATH}" \
    --mode llm-only \
    --output_dir "${OUTPUT_DIR}" \
    --llm_bits 4 \
    --llm_group_size 64 \
    --sym \
    --seed 42 \
    --deterministic \
    --iters 200 \
    --nsamples 128 \
    --seqlen 2048 \
    --batch_size "${BATCH_SIZE}" \
    --dataset "NeelNanda/pile-10k" \
    --format auto_round:gptqmodel \
    --attn_bits 8 \
    --shared_expert_bits 8 \
    2>&1 | tee "${LOG_DIR}/qwen36moe-no-rotate-attn8-shared8.log"

echo "Done! Output saved to ${OUTPUT_DIR}"
