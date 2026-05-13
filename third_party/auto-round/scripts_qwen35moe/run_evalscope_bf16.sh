#!/bin/bash
# Qwen3.5-35B-A3B BF16 EvalScope 评测脚本（直接使用 evalscope CLI）
#
# 默认数据集选择为 ceval + math_500，避免两个常见坑：
#   1. gpqa / gpqa_diamond 需要先执行 hf auth login
#   2. humaneval 评测代码任务时通常需要 --use-sandbox
#
# 用法:
#   bash scripts_qwen35moe/run_evalscope_bf16.sh
#   bash scripts_qwen35moe/run_evalscope_bf16.sh ceval math_500
#   LIMIT=20 bash scripts_qwen35moe/run_evalscope_bf16.sh
#   USE_SANDBOX=1 bash scripts_qwen35moe/run_evalscope_bf16.sh humaneval
#
# 可覆盖环境变量:
#   MODEL_PATH       模型路径，默认 /data01/nfs_shared/Qwen3.5-35B-A3B
#   MODEL_ID         报告里显示的模型名
#   WORK_DIR         EvalScope 输出根目录
#   LOG_DIR          日志目录
#   DEVICE_MAP       传给 from_pretrained 的 device_map，默认 auto
#   EVAL_BATCH_SIZE  EvalScope batch size，默认 1
#   DATASET_HUB      数据集来源，默认 modelscope
#   MAX_TOKENS       生成最大 token 数，默认 4096
#   LIMIT            每个 subset 的评测样本上限，便于快速 smoke test
#   USE_SANDBOX      设为 1 时追加 --use-sandbox
#   NO_TIMESTAMP     设为 1 时追加 --no-timestamp

set -euo pipefail

MODEL_PATH="${MODEL_PATH:-/data01/nfs_shared/Qwen3.5-35B-A3B}"
MODEL_ID="${MODEL_ID:-Qwen3.5-35B-A3B-BF16}"
WORK_DIR="${WORK_DIR:-./output/evalscope/qwen35moe-bf16}"
LOG_DIR="${LOG_DIR:-./output/logs}"
DEVICE_MAP="${DEVICE_MAP:-auto}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-1}"
DATASET_HUB="${DATASET_HUB:-modelscope}"
MAX_TOKENS="${MAX_TOKENS:-4096}"

mkdir -p "${LOG_DIR}" "${WORK_DIR}"

if [ "$#" -gt 0 ]; then
    DATASETS=("$@")
else
    DATASETS=(ceval math_500)
fi

MODEL_ARGS="precision=bfloat16,device_map=${DEVICE_MAP}"
GENERATION_CONFIG="do_sample=false,max_tokens=${MAX_TOKENS},temperature=0.0"

EXTRA_ARGS=()
if [ -n "${LIMIT:-}" ]; then
    EXTRA_ARGS+=(--limit "${LIMIT}")
fi
if [ "${USE_SANDBOX:-0}" = "1" ]; then
    EXTRA_ARGS+=(--use-sandbox)
fi
if [ "${NO_TIMESTAMP:-0}" = "1" ]; then
    EXTRA_ARGS+=(--no-timestamp)
fi

LOG_FILE="${LOG_DIR}/qwen35moe-bf16-evalscope.log"

echo "=========================================="
echo "Qwen3.5-35B-A3B BF16 EvalScope Evaluation"
echo "Model: ${MODEL_PATH}"
echo "Model ID: ${MODEL_ID}"
echo "Datasets: ${DATASETS[*]}"
echo "Device map: ${DEVICE_MAP}"
echo "Eval batch size: ${EVAL_BATCH_SIZE}"
echo "Dataset hub: ${DATASET_HUB}"
echo "Work dir: ${WORK_DIR}"
echo "Log file: ${LOG_FILE}"
echo "=========================================="

evalscope eval \
    --model "${MODEL_PATH}" \
    --model-id "${MODEL_ID}" \
    --model-args "${MODEL_ARGS}" \
    --datasets "${DATASETS[@]}" \
    --dataset-hub "${DATASET_HUB}" \
    --generation-config "${GENERATION_CONFIG}" \
    --eval-backend Native \
    --eval-batch-size "${EVAL_BATCH_SIZE}" \
    --work-dir "${WORK_DIR}" \
    "${EXTRA_ARGS[@]}" \
    2>&1 | tee "${LOG_FILE}"

echo ""
echo "=========================================="
echo "EvalScope evaluation complete"
echo "Results root: ${WORK_DIR}"
echo "=========================================="