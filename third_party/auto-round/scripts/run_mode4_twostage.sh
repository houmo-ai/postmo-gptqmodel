#!/bin/bash
# Mode 4 (two-stage): Rotate + Quant All (LLM W4G64 + Vision W8G64)
# 两阶段校准:
#   Stage 1: 使用 pile-10k 校准 LLM (保持文本质量)
#   Stage 2: 使用 llava_conv_58k 校准 Vision (使用多模态数据)
# 解决单阶段强制切换校准数据导致 LLM PPL 退化的问题

set -euo pipefail

MODEL_PATH="${1:-weights/Qwen3.5-27B}"
MODEL_NAME="$(basename "${MODEL_PATH%/}")"
OUTPUT_DIR="${2:-./output/${MODEL_NAME}-mode4-twostage}"
LOG_DIR="${LOG_DIR:-./output/logs}"
LOG_FILE="${LOG_DIR}/${MODEL_NAME}-mode4-twostage.log"

mkdir -p "${LOG_DIR}"

echo "=========================================="
echo "Mode 4 (Two-Stage): Rotate All → Quant LLM (W4G64) + Vision (W8G64)"
echo "  Stage 1: LLM calibrated with pile-10k"
echo "  Stage 2: Vision calibrated with llava_conv_58k"
echo "Model: ${MODEL_PATH}"
echo "Output: ${OUTPUT_DIR}"
echo "=========================================="

python scripts/quantize.py \
    --model "${MODEL_PATH}" \
    --mode rotate-quant-all \
    --output_dir "${OUTPUT_DIR}" \
    --llm_bits 4 \
    --llm_group_size 64 \
    --vision_bits 8 \
    --vision_group_size 64 \
    --iters 200 \
    --nsamples 128 \
    --seqlen 2048 \
    --batch_size 8 \
    --dataset "NeelNanda/pile-10k" \
    --format auto_gptq \
    --two_stage \
    2>&1 | tee "${LOG_FILE}"

echo "Done! Output saved to ${OUTPUT_DIR}"
