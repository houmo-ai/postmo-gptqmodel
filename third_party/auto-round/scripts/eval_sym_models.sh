#!/bin/bash
# Evaluate all 4 symmetric quantization models: PPL + Vision
# Usage: bash scripts/eval_sym_models.sh [GPU_ID]
set -euo pipefail

GPU=${1:-0}
ORIGINAL_MODEL="${2:-weights/Qwen3.5-27B}"
MODEL_NAME="$(basename "${ORIGINAL_MODEL%/}")"
GPTQ_MODEL="${3:-${ORIGINAL_MODEL}-GPTQModel-self-generated-4bit-hessian-mse}"
SYM_OUTPUT_ROOT="${4:-output}"
LOG_DIR="${5:-./output/eval_logs/${MODEL_NAME}-sym}"
mkdir -p "${LOG_DIR}"

MODELS=(
  "FP16:${ORIGINAL_MODEL}"
  "GPTQ-baseline:${GPTQ_MODEL}"
  "sym-mode1:${SYM_OUTPUT_ROOT}/sym-mode1"
  "sym-mode2:${SYM_OUTPUT_ROOT}/sym-mode2"
  "sym-mode3:${SYM_OUTPUT_ROOT}/sym-mode3"
  "sym-mode4:${SYM_OUTPUT_ROOT}/sym-mode4"
)

echo "=================================="
echo "Symmetric Quantization Evaluation"
echo "GPU: ${GPU}"
echo "Base model: ${MODEL_NAME}"
echo "=================================="

for entry in "${MODELS[@]}"; do
  name="${entry%%:*}"
  path="${entry##*:}"

  if [ ! -d "$path" ]; then
    echo "[SKIP] $name: $path not found"
    continue
  fi

  echo ""
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo "Model: $name ($path)"
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

  # PPL evaluation
  echo "[PPL] Running..."
  CUDA_VISIBLE_DEVICES=${GPU} python /tmp/eval_ppl_quick.py "$path" "cuda:0" \
    2>&1 | tee "${LOG_DIR}/${name}_ppl.txt"

  # Vision evaluation
  echo "[Vision] Running..."
  CUDA_VISIBLE_DEVICES=${GPU} python scripts/test_vision.py \
    --model_path "$path" \
    --device "cuda:0" \
    --output "${LOG_DIR}/${name}_vision.json" \
    2>&1 | tee "${LOG_DIR}/${name}_vision.txt"

  echo "[Done] $name"
done

echo ""
echo "All evaluations complete. Results in ${LOG_DIR}/"
