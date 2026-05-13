#!/bin/bash
# Mode 4: Rotate LLM + Rotate Vision + Quant Vision (W8G64) + Quant LLM (W4G64)
# 适用场景: 最大压缩，Vision 和 LLM 均量化
#
# 操作步骤:
#   1. Fuse LLM RMSNorm weights → adjacent linears
#   2. Fuse Vision LayerNorm weights → adjacent linears
#   3. Hadamard rotate 整个 Vision encoder
#   4. Rotate merger output → LLM 旋转空间
#   5. Hadamard rotate LLM backbone
#   6. AutoRound 量化:
#      - Vision blocks: W8G64 (通过 layer_config 单独配置)
#      - LLM layers: W4G64 (默认配置)
#
# quant_nontext_module=True 启用视觉模块量化

set -euo pipefail

MODEL_PATH="${1:-weights/Qwen3.5-27B}"
MODEL_NAME="$(basename "${MODEL_PATH%/}")"
OUTPUT_DIR="${2:-./output/${MODEL_NAME}-mode4-rotate-quant-all}"
LOG_DIR="${LOG_DIR:-./output/logs}"
LOG_FILE="${LOG_DIR}/${MODEL_NAME}-mode4-rotate-quant-all.log"

mkdir -p "${LOG_DIR}"

echo "=========================================="
echo "Mode 4: Rotate + Quant All"
echo "  Vision: W8G64, LLM: W4G64"
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
    --sym \
    --seed 42 \
    --deterministic \
    --two_stage \
    --iters 200 \
    --nsamples 128 \
    --seqlen 2048 \
    --batch_size 8 \
    --dataset "NeelNanda/pile-10k" \
    --format auto_gptq \
    2>&1 | tee "${LOG_FILE}"

echo "Done! Output saved to ${OUTPUT_DIR}"
