#!/bin/bash
# Mode 1: 只量化 LLM 部分 (W4G64, 无旋转)
# 适用场景: 快速基线量化，不涉及旋转操作
#
# 模型结构保持不变，仅对 LLM 的 language_model.layers 做 W4G64 量化
# Vision encoder 和 merger 保持 FP16/BF16 不变

set -euo pipefail

MODEL_PATH="${1:-weights/Qwen3.5-27B}"
MODEL_NAME="$(basename "${MODEL_PATH%/}")"
OUTPUT_DIR="${2:-./output/${MODEL_NAME}-mode1-llm-only}"
LOG_DIR="${LOG_DIR:-./output/logs}"
LOG_FILE="${LOG_DIR}/${MODEL_NAME}-mode1-llm-only.log"

mkdir -p "${LOG_DIR}"

echo "=========================================="
echo "Mode 1: LLM-Only Quantization (W4G64)"
echo "Model: ${MODEL_PATH}"
echo "Output: ${OUTPUT_DIR}"
echo "=========================================="

python third_party/auto-round/scripts/quantize.py \
    --model "${MODEL_PATH}" \
    --mode llm-only \
    --output_dir "${OUTPUT_DIR}" \
    --llm_bits 4 \
    --llm_group_size 64 \
    --sym \
    --seed 42 \
    --iters 200 \
    --nsamples 128 \
    --seqlen 2048 \
    --batch_size 8 \
    --dataset "NeelNanda/pile-10k" \
    --format auto_gptq \
    2>&1 | tee "${LOG_FILE}"

echo "Done! Output saved to ${OUTPUT_DIR}"
