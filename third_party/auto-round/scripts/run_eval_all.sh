#!/bin/bash
# Batch evaluation of all Qwen3.5 quantized models + baselines.
# Runs: PPL, text inference, chat quality, vision inference.
#
# Results saved as JSON per model in each output directory.
# Usage: bash scripts/run_eval_all.sh

set -euo pipefail

ORIGINAL_MODEL="${1:-weights/Qwen3.5-27B}"
MODEL_NAME="$(basename "${ORIGINAL_MODEL%/}")"
DEVICE="${2:-cuda:5}"
GPTQ_MODEL="${3:-${ORIGINAL_MODEL}-GPTQModel-self-generated-4bit-hessian-mse}"
EVAL_LOG_DIR="${4:-./output/eval_logs/${MODEL_NAME}}"
mkdir -p "${EVAL_LOG_DIR}"

# Models to evaluate (in order)
declare -A MODELS
MODELS["fp16-original"]="${ORIGINAL_MODEL}"
MODELS["gptq-baseline"]="${GPTQ_MODEL}"
MODELS["mode1-llm-only"]="./output/${MODEL_NAME}-mode1-llm-only"
MODELS["mode2-rotate-merger"]="./output/${MODEL_NAME}-mode2-rotate-merger"
MODELS["mode3-rotate-full"]="./output/${MODEL_NAME}-mode3-rotate-full"
MODELS["mode4-rotate-quant-all"]="./output/${MODEL_NAME}-mode4-rotate-quant-all"

# Order matters for display
ORDER=("fp16-original" "gptq-baseline" "mode1-llm-only" "mode2-rotate-merger" "mode3-rotate-full" "mode4-rotate-quant-all")

echo "=========================================="
echo "Batch Evaluation - ${MODEL_NAME}"
echo "Device: ${DEVICE}"
echo "Models: ${#MODELS[@]}"
echo "=========================================="

for name in "${ORDER[@]}"; do
    model_path="${MODELS[$name]}"
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "Evaluating: ${name} → ${model_path}"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    if [ ! -d "${model_path}" ]; then
        echo "⚠️  Path not found, skipping: ${model_path}"
        continue
    fi

    # Run full eval (PPL + text + chat + vision)
    python scripts/test_eval.py \
        --model_path "${model_path}" \
        --original_model "${ORIGINAL_MODEL}" \
        --device "${DEVICE}" \
        --output_report "${EVAL_LOG_DIR}/${name}_eval.json" \
        2>&1 | tee "${EVAL_LOG_DIR}/${name}_eval.log"

    # Run vision-specific eval
    python scripts/test_vision.py \
        --model_path "${model_path}" \
        --original_model "${ORIGINAL_MODEL}" \
        --device "${DEVICE}" \
        --output "${EVAL_LOG_DIR}/${name}_vision.json" \
        2>&1 | tee "${EVAL_LOG_DIR}/${name}_vision.log"

    echo "✅ Done: ${name}"
done

echo ""
echo "=========================================="
echo "All evaluations complete!"
echo "Results in: ${EVAL_LOG_DIR}/"
echo "=========================================="
