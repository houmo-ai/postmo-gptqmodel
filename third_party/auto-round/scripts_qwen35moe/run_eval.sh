#!/bin/bash
# Qwen3.5-MoE (35B-A3B) PPL 评测脚本
# 评测浮点模型和量化模型的 wikitext PPL
#
# 用法:
#   bash scripts_qwen35moe/run_eval.sh [device] [model_path] [quant_dir]
#   bash scripts_qwen35moe/run_eval.sh cuda:1 /data01/datasets/Qwen3.5-35B-A3B ./output/Qwen3.5-35B-A3B-mode1-llm-only

set -euo pipefail

DEVICE="${1:-cuda:0}"
MODEL_PATH="${2:-weights/Qwen3.5-35B-A3B}"
MODEL_NAME="$(basename "${MODEL_PATH%/}")"
QUANT_DIR="${3:-./output/${MODEL_NAME}-mode1-llm-only}"
QUANT_NAME="$(basename "${QUANT_DIR%/}")"
LOG_DIR="${LOG_DIR:-./output/logs}"

mkdir -p "${LOG_DIR}"

echo "=========================================="
echo "Qwen3.5-MoE PPL Evaluation"
echo "Device: ${DEVICE}"
echo "Base model: ${MODEL_PATH}"
echo "Quant model: ${QUANT_DIR}"
echo "=========================================="

# ── Step 1: BF16 浮点模型 PPL (baseline) ──
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Step 1: BF16 Baseline PPL"
echo "Model: ${MODEL_PATH}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

python scripts_qwen35moe/eval/eval_quant_ppl.py \
    --model "${MODEL_PATH}" \
    --device "${DEVICE}" \
    --dtype bfloat16 \
    --eval_bs 4 \
    2>&1 | tee "${LOG_DIR}/${MODEL_NAME}-bf16-ppl.log"

# ── Step 2: 量化模型 PPL ──
if [ -d "${QUANT_DIR}" ]; then
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "Step 2: Quantized Model PPL"
    echo "Model: ${QUANT_DIR}"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    python scripts_qwen35moe/eval/eval_quant_ppl.py \
        --model "${QUANT_DIR}" \
        --device "${DEVICE}" \
        --eval_bs 4 \
        2>&1 | tee "${LOG_DIR}/${QUANT_NAME}-ppl.log"
else
    echo ""
    echo "⚠️  Quantized model not found at ${QUANT_DIR}"
    echo "   Run quantization first: bash scripts_qwen35moe/run_mode1_llm_only.sh \"${MODEL_PATH}\" \"${QUANT_DIR}\""
fi

echo ""
echo "=========================================="
echo "Evaluation complete. Logs in ${LOG_DIR}"
echo "=========================================="
