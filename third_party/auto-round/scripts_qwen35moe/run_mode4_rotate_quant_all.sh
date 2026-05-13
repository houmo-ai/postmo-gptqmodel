#!/bin/bash
# Mode 4: Rotate MoE LLM + Rotate Vision + Quant Vision (W8G64) + Quant LLM (W4G64)
# 适用场景: 最大压缩，Vision 和 LLM 均量化

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

MODEL_PATH="${1:-weights/Qwen3.5-35B-A3B}"
MODEL_NAME="$(basename "${MODEL_PATH%/}")"
OUTPUT_DIR="${2:-./output/${MODEL_NAME}-mode4-rotate-quant-all}"
LOG_DIR="${LOG_DIR:-./output/logs}"
LOG_FILE="${LOG_DIR}/${MODEL_NAME}-mode4-rotate-quant-all.log"
EXTRA_ARGS=("${@:3}")
VISIBLE_GPUS="${CUDA_VISIBLE_DEVICES:-0,1}"
AUTOROUND_DEVICE_MAP="${AR_DEVICE_MAP:-balanced}"
LOW_GPU_MEM_USAGE="${AR_LOW_GPU_MEM_USAGE:-1}"
LOW_GPU_MEM_ARGS=()
PYTHON_BIN="$(resolve_qwen35moe_python)"

mkdir -p "${LOG_DIR}"
export CUDA_VISIBLE_DEVICES="${VISIBLE_GPUS}"

case "${LOW_GPU_MEM_USAGE}" in
  1|true|TRUE|yes|YES|on|ON) LOW_GPU_MEM_ARGS+=(--low_gpu_mem_usage) ;;
esac

echo "=========================================="
echo "Mode 4: Rotate + Quant All"
echo "  Vision: W8G64, LLM: W4G64"
echo "Model: ${MODEL_PATH}"
echo "Output: ${OUTPUT_DIR}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
echo "AutoRound device_map: ${AUTOROUND_DEVICE_MAP}"
echo "low_gpu_mem_usage: ${LOW_GPU_MEM_USAGE}"
echo "=========================================="

"${PYTHON_BIN}" scripts_qwen35moe/quantize_moe.py \
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
    --device_map "${AUTOROUND_DEVICE_MAP}" \
    "${LOW_GPU_MEM_ARGS[@]}" \
    --dataset "NeelNanda/pile-10k" \
    --format auto_gptq \
    "${EXTRA_ARGS[@]}" \
    2>&1 | tee "${LOG_FILE}"

echo "Done! Output saved to ${OUTPUT_DIR}"
