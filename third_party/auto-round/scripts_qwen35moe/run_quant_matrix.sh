#!/bin/bash
set -euo pipefail

MODEL_PATH="/data01/nfs_shared/Qwen3.5-35B-A3B"
OUTPUT_ROOT="/data01/home/huxing/hm-auto-round/output"
LOG_DIR="${OUTPUT_ROOT}/logs"
mkdir -p "${LOG_DIR}"

echo "Existing all-4bit rotated checkpoint:"
echo "  ${OUTPUT_ROOT}/qwen35moe-mode3-rotate-full"

run_variant() {
  local gpu="$1"
  local output_dir="$2"
  local log_file="$3"
  shift 3

  echo "Launching ${output_dir} on GPU ${gpu}"
  CUDA_VISIBLE_DEVICES="${gpu}" nohup python scripts_qwen35moe/quantize_moe.py \
    --model "${MODEL_PATH}" \
    --mode rotate-full \
    --output_dir "${output_dir}" \
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
    "$@" \
    > "${log_file}" 2>&1 &
  echo "  pid=$!"
}

run_variant 0 \
  "${OUTPUT_ROOT}/qwen35moe-rotate-attn5-shared5-expert4" \
  "${LOG_DIR}/qwen35moe-rotate-attn5-shared5-expert4.log" \
  --attn_bits 5 \
  --shared_expert_bits 5 \
  --expert_bits 4

run_variant 1 \
  "${OUTPUT_ROOT}/qwen35moe-rotate-attn5-shared5-expert4-down5" \
  "${LOG_DIR}/qwen35moe-rotate-attn5-shared5-expert4-down5.log" \
  --attn_bits 5 \
  --shared_expert_bits 5 \
  --expert_bits 4 \
  --expert_down_bits 5
