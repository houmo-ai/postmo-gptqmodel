#!/bin/bash
# Mode 3: Rotate LLM + Rotate Vision (full) + 量化 LLM (W4G64)
# 适用场景: 全旋转但只量化 LLM
#
# 操作步骤:
#   1. Fuse LLM RMSNorm weights → adjacent linears
#   2. Fuse Vision LayerNorm weights → adjacent linears
#   3. Hadamard rotate 整个 Vision encoder (patch_embed, blocks, merger input)
#   4. Rotate merger output → LLM 旋转空间
#   5. Hadamard rotate LLM backbone
#   6. AutoRound 量化 LLM layers (W4G64)
#
# Vision encoder 已旋转但保持全精度 (不量化)

set -euo pipefail

MODEL_PATH="${1:-weights/Qwen3.5-27B}"
MODEL_NAME="$(basename "${MODEL_PATH%/}")"
OUTPUT_DIR="${2:-./output/${MODEL_NAME}-mode3-rotate-full}"
LOG_DIR="${LOG_DIR:-./output/logs}"
LOG_FILE="${LOG_DIR}/${MODEL_NAME}-mode3-rotate-full.log"

mkdir -p "${LOG_DIR}"

echo "=========================================="
echo "Mode 3: Rotate LLM + Full Vision → Quant LLM (W4G64)"
echo "Model: ${MODEL_PATH}"
echo "Output: ${OUTPUT_DIR}"
echo "=========================================="

python scripts/quantize.py \
    --model "${MODEL_PATH}" \
    --mode rotate-full \
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
    # --format auto_gptq \
    2>&1 | tee "${LOG_FILE}"

echo "Done! Output saved to ${OUTPUT_DIR}"
