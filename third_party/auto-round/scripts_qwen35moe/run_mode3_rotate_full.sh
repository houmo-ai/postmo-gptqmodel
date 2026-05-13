#!/bin/bash
# Mode 3: Rotate MoE LLM + Rotate Vision (full) + 量化 LLM (W4G64)
# 适用场景: 全旋转但只量化 LLM (Qwen3.5-35B-A3B MoE)
#
# 操作步骤:
#   1. Fuse LLM RMSNorm weights → MoE experts + attention (1+weight scaling)
#   2. Fuse Vision LayerNorm weights → adjacent linears
#   3. Hadamard rotate 整个 Vision encoder (patch_embed, blocks, merger input)
#   4. Rotate merger output → LLM 旋转空间
#   5. Hadamard rotate MoE LLM backbone (gate, shared_expert, per-expert, attention)
#   6. AutoRound 量化 LLM layers (W4G64)
#
# Vision encoder 已旋转但保持全精度 (不量化)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

MODEL_PATH="${1:-weights/Qwen3.5-35B-A3B}"
MODEL_NAME="$(basename "${MODEL_PATH%/}")"
OUTPUT_DIR="${2:-./output/${MODEL_NAME}-mode3-rotate-full}"
LOG_DIR="${LOG_DIR:-./output/logs}"
LOG_FILE="${LOG_DIR}/${MODEL_NAME}-mode3-rotate-full.log"
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
echo "Mode 3: Rotate MoE LLM + Full Vision → Quant LLM (W4G64)"
echo "Model: ${MODEL_PATH}"
echo "Output: ${OUTPUT_DIR}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
echo "AutoRound device_map: ${AUTOROUND_DEVICE_MAP}"
echo "low_gpu_mem_usage: ${LOW_GPU_MEM_USAGE}"
echo "=========================================="

"${PYTHON_BIN}" scripts_qwen35moe/quantize_moe.py \
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
    --device_map "${AUTOROUND_DEVICE_MAP}" \
    "${LOW_GPU_MEM_ARGS[@]}" \
    --dataset "NeelNanda/pile-10k" \
    --format auto_gptq \
    "${EXTRA_ARGS[@]}" \
    2>&1 | tee "${LOG_FILE}"

echo "Done! Output saved to ${OUTPUT_DIR}"
