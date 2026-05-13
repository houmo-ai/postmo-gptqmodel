#!/bin/bash
# Mode 2: Rotate LLM + Vision Merger + 量化 LLM (W4G64)
# 适用场景: 推荐模式，旋转对齐 LLM 和 merger 接口
#
# 操作步骤:
#   1. Fuse LLM RMSNorm weights → adjacent linears
#   2. Rotate merger output (linear_fc2) → 与 LLM 旋转空间对齐
#   3. Hadamard rotate LLM backbone (embed, layers, lm_head)
#   4. AutoRound 量化 LLM layers (W4G64)
#
# Vision encoder 保持 FP16/BF16 不变，但 merger 输出已旋转

set -euo pipefail

MODEL_PATH="${1:-weights/Qwen3.5-27B}"
MODEL_NAME="$(basename "${MODEL_PATH%/}")"
OUTPUT_DIR="${2:-./output/${MODEL_NAME}-mode2-rotate-merger}"
LOG_DIR="${LOG_DIR:-./output/logs}"
LOG_FILE="${LOG_DIR}/${MODEL_NAME}-mode2-rotate-merger.log"

mkdir -p "${LOG_DIR}"

echo "=========================================="
echo "Mode 2: Rotate LLM + Merger → Quant LLM (W4G64)"
echo "Model: ${MODEL_PATH}"
echo "Output: ${OUTPUT_DIR}"
echo "=========================================="

python scripts/quantize.py \
    --model "${MODEL_PATH}" \
    --mode rotate-merger \
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
    --format auto_gptq \
    2>&1 | tee "${LOG_FILE}"

echo "Done! Output saved to ${OUTPUT_DIR}"
