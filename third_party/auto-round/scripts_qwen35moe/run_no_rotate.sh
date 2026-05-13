#!/bin/bash
# No Rotate: Quantize LLM only (W4G64)
# 适用场景: 不做旋转，只量化 LLM (Qwen3.5-35B-A3B MoE)
#
# 操作步骤:
#   1. 直接加载原始 Qwen3.5-MoE 模型
#   2. 不做 Vision / merger / LLM Hadamard 旋转
#   3. AutoRound 仅量化 LLM layers (W4G64)

set -euo pipefail

MODEL_PATH="/data01/nfs_shared/Qwen3.5-35B-A3B"
OUTPUT_DIR="./output/qwen35moe-no-rotate"
LOG_DIR="./output/logs"

mkdir -p "${LOG_DIR}"

echo "=========================================="
echo "No Rotate: Quantize Qwen3.5-MoE LLM only (W4G64)"
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
    --batch_size 8 \
    --dataset "NeelNanda/pile-10k" \
    --format auto_round:gptqmodel \
    2>&1 | tee "${LOG_DIR}/qwen35moe-no-rotate.log"

echo "Done! Output saved to ${OUTPUT_DIR}"
