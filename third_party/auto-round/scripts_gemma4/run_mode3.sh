#!/bin/bash
# Gemma4-31B Mode 3: Rotate LLM + full vision + projection, quantize LLM
# GPU: cuda:5

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "${SCRIPT_DIR}/.."

MODEL="weights/gemma-4-31B-it"
OUTPUT_DIR="output/gemma4-mode3"
LOG_DIR="output/logs"
mkdir -p "${OUTPUT_DIR}" "${LOG_DIR}"

echo "============================================================"
echo "Gemma4-31B Mode 3: Rotate full (LLM + vision + projection)"
echo "GPU: cuda:7"
echo "Output: ${OUTPUT_DIR}"
echo "============================================================"

python scripts_gemma4/quantize.py \
    --model "${MODEL}" \
    --mode rotate-full \
    --device cuda:7 \
    --output_dir "${OUTPUT_DIR}" \
    --llm_bits 4 \
    --llm_group_size 64 \
    --sym \
    --iters 200 \
    --nsamples 128 \
    --seqlen 2048 \
    --batch_size 8 \
    --dataset NeelNanda/pile-10k \
    --seed 42 \
    --format auto_gptq \
    2>&1 | tee "${LOG_DIR}/gemma4-mode3.log"
