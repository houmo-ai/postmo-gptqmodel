#!/bin/bash
# Mode 1: 只量化 MoE LLM 部分 (W4G64, 无旋转)
# 适用场景: 当前最佳基线，不做旋转，直接对 language_model 做 AutoRound

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

MODEL_PATH="${1:-weights/Qwen3.5-35B-A3B}"
MODEL_NAME="$(basename "${MODEL_PATH%/}")"
ATTN_BITS="${AR_ATTN_BITS:-8}"
SHARED_EXPERT_BITS="${AR_SHARED_EXPERT_BITS:-8}"
ITERS="${AR_ITERS:-200}"
NSAMPLES="${AR_NSAMPLES:-128}"
BATCH_SIZE="${AR_BATCH_SIZE:-8}"
GRADIENT_ACCUMULATE_STEPS="${AR_GRADIENT_ACCUMULATE_STEPS:-1}"
RAW_ARGS=()
LOG_DIR="${LOG_DIR:-./output/logs}"
VISIBLE_GPUS="${CUDA_VISIBLE_DEVICES:-0}"
AUTOROUND_DEVICE_MAP="${AR_DEVICE_MAP:-0}"
LOW_GPU_MEM_USAGE="${AR_LOW_GPU_MEM_USAGE:-1}"
LOW_GPU_MEM_ARGS=()
QUANT_OVERRIDE_ARGS=()
PYTHON_BIN="$(resolve_qwen35moe_python)"

if [ "$#" -ge 2 ]; then
  if [[ "$2" == --* ]]; then
    RAW_ARGS=("${@:2}")
  else
    OUTPUT_DIR="$2"
    RAW_ARGS=("${@:3}")
  fi
fi

EXTRA_ARGS=()
index=0
while [ "${index}" -lt "${#RAW_ARGS[@]}" ]; do
  arg="${RAW_ARGS[$index]}"
  case "${arg}" in
    --attn_bits)
      if [ $((index + 1)) -lt "${#RAW_ARGS[@]}" ]; then
        index=$((index + 1))
        ATTN_BITS="${RAW_ARGS[$index]}"
      else
        EXTRA_ARGS+=("${arg}")
      fi
      ;;
    --attn_bits=*)
      ATTN_BITS="${arg#*=}"
      ;;
    --shared_expert_bits)
      if [ $((index + 1)) -lt "${#RAW_ARGS[@]}" ]; then
        index=$((index + 1))
        SHARED_EXPERT_BITS="${RAW_ARGS[$index]}"
      else
        EXTRA_ARGS+=("${arg}")
      fi
      ;;
    --shared_expert_bits=*)
      SHARED_EXPERT_BITS="${arg#*=}"
      ;;
    --iters)
      if [ $((index + 1)) -lt "${#RAW_ARGS[@]}" ]; then
        index=$((index + 1))
        ITERS="${RAW_ARGS[$index]}"
      else
        EXTRA_ARGS+=("${arg}")
      fi
      ;;
    --iters=*)
      ITERS="${arg#*=}"
      ;;
    --nsamples)
      if [ $((index + 1)) -lt "${#RAW_ARGS[@]}" ]; then
        index=$((index + 1))
        NSAMPLES="${RAW_ARGS[$index]}"
      else
        EXTRA_ARGS+=("${arg}")
      fi
      ;;
    --nsamples=*)
      NSAMPLES="${arg#*=}"
      ;;
    --batch_size)
      if [ $((index + 1)) -lt "${#RAW_ARGS[@]}" ]; then
        index=$((index + 1))
        BATCH_SIZE="${RAW_ARGS[$index]}"
      else
        EXTRA_ARGS+=("${arg}")
      fi
      ;;
    --batch_size=*)
      BATCH_SIZE="${arg#*=}"
      ;;
    --gradient_accumulate_steps)
      if [ $((index + 1)) -lt "${#RAW_ARGS[@]}" ]; then
        index=$((index + 1))
        GRADIENT_ACCUMULATE_STEPS="${RAW_ARGS[$index]}"
      else
        EXTRA_ARGS+=("${arg}")
      fi
      ;;
    --gradient_accumulate_steps=*)
      GRADIENT_ACCUMULATE_STEPS="${arg#*=}"
      ;;
    *)
      EXTRA_ARGS+=("${arg}")
      ;;
  esac
  index=$((index + 1))
done

RUN_NAME="mode1-llm-only"
if [ -n "${ATTN_BITS}" ]; then
  RUN_NAME+="-attn${ATTN_BITS}"
  QUANT_OVERRIDE_ARGS+=(--attn_bits "${ATTN_BITS}")
fi
if [ -n "${SHARED_EXPERT_BITS}" ]; then
  RUN_NAME+="-shared${SHARED_EXPERT_BITS}"
  QUANT_OVERRIDE_ARGS+=(--shared_expert_bits "${SHARED_EXPERT_BITS}")
fi
if [ "${NSAMPLES}" != "128" ]; then
  RUN_NAME+="-n${NSAMPLES}"
fi
if [ "${ITERS}" != "200" ]; then
  RUN_NAME+="-iter${ITERS}"
fi
if [ "${BATCH_SIZE}" != "8" ]; then
  RUN_NAME+="-bs${BATCH_SIZE}"
fi
if [ "${GRADIENT_ACCUMULATE_STEPS}" != "1" ]; then
  RUN_NAME+="-ga${GRADIENT_ACCUMULATE_STEPS}"
fi

OUTPUT_DIR="${OUTPUT_DIR:-./output/${MODEL_NAME}-${RUN_NAME}}"
LOG_FILE="${LOG_DIR}/${MODEL_NAME}-${RUN_NAME}.log"

mkdir -p "${LOG_DIR}"
export CUDA_VISIBLE_DEVICES="${VISIBLE_GPUS}"

case "${LOW_GPU_MEM_USAGE}" in
  1|true|TRUE|yes|YES|on|ON) LOW_GPU_MEM_ARGS+=(--low_gpu_mem_usage) ;;
esac

echo "=========================================="
echo "Mode 1: MoE LLM-Only Quantization (W4G64)"
echo "Model: ${MODEL_PATH}"
echo "Run tag: ${RUN_NAME}"
echo "Output: ${OUTPUT_DIR}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
echo "AutoRound device_map: ${AUTOROUND_DEVICE_MAP}"
echo "low_gpu_mem_usage: ${LOW_GPU_MEM_USAGE}"
echo "attn_bits override: ${ATTN_BITS:-default}"
echo "shared_expert_bits override: ${SHARED_EXPERT_BITS:-default}"
echo "iters: ${ITERS}"
echo "nsamples: ${NSAMPLES}"
echo "batch_size: ${BATCH_SIZE}"
echo "gradient_accumulate_steps: ${GRADIENT_ACCUMULATE_STEPS}"
echo "=========================================="

"${PYTHON_BIN}" third_party/auto-round/scripts_qwen35moe/quantize_moe.py \
    --model "${MODEL_PATH}" \
    --mode llm-only \
    --output_dir "${OUTPUT_DIR}" \
    --llm_bits 4 \
    --llm_group_size 64 \
    --attn_bits 8 \
    --shared_expert_bits 8 \
    --sym \
    --seed 42 \
    --iters "${ITERS}" \
    --nsamples "${NSAMPLES}" \
    --seqlen 2048 \
    --batch_size "${BATCH_SIZE}" \
    --gradient_accumulate_steps "${GRADIENT_ACCUMULATE_STEPS}" \
    --device_map "${AUTOROUND_DEVICE_MAP}" \
    "${LOW_GPU_MEM_ARGS[@]}" \
    --dataset "NeelNanda/pile-10k" \
    --format auto_round:gptqmodel \
    "${QUANT_OVERRIDE_ARGS[@]}" \
    "${EXTRA_ARGS[@]}" \
    2>&1 | tee "${LOG_FILE}"

echo "Done! Output saved to ${OUTPUT_DIR}"
