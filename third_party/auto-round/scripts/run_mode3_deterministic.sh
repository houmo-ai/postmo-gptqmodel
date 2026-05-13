#!/bin/bash
# Mode 3 (deterministic): Rotate LLM + Full Vision + 量化 LLM (W4G64)
# 使用确定性 CuBLAS 和 eager attention，确保可复现结果
# 与 Mode 2 deterministic 对比，验证视觉旋转不影响文本 PPL

set -euo pipefail

MODEL_PATH="${1:-weights/Qwen3.5-27B}"
MODEL_NAME="$(basename "${MODEL_PATH%/}")"
OUTPUT_DIR="${2:-./output/${MODEL_NAME}-mode3-deterministic}"
LOG_DIR="${LOG_DIR:-./output/logs}"
LOG_FILE="${LOG_DIR}/${MODEL_NAME}-mode3-deterministic.log"

mkdir -p "${LOG_DIR}"

echo "=========================================="
echo "Mode 3 (Deterministic): Rotate Full Vision + LLM → Quant LLM (W4G64)"
echo "Model: ${MODEL_PATH}"
echo "Output: ${OUTPUT_DIR}"
echo "=========================================="

python scripts/quantize.py \
    --model "${MODEL_PATH}" \
    --mode rotate-full \
    --output_dir "${OUTPUT_DIR}" \
    --llm_bits 4 \
    --llm_group_size 64 \
    --iters 200 \
    --nsamples 128 \
    --seqlen 2048 \
    --batch_size 8 \
    --dataset "NeelNanda/pile-10k" \
    --format auto_gptq \
    --deterministic \
    2>&1 | tee "${LOG_FILE}"

echo "Done! Output saved to ${OUTPUT_DIR}"
