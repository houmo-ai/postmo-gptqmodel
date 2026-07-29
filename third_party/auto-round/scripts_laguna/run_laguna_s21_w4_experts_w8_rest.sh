#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
MODEL_PATH="${1:-/data01/datasets/Laguna-S-2.1}"
OUTPUT_DIR="${2:-${REPO_ROOT}/work_dirs/Laguna-S-2.1-autoround-expert-w4-rest-w8-g64}"
PYTHON_BIN="${AR_PYTHON:-/data01/home/xuzk/anaconda3/envs/gptqmodel/bin/python}"
LOG_DIR="${LOG_DIR:-${REPO_ROOT}/work_dirs/logs}"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/laguna-s21-autoround-expert-w4-rest-w8.log}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Python interpreter is not executable: ${PYTHON_BIN}" >&2
  exit 1
fi

mkdir -p "${LOG_DIR}" "$(dirname "${OUTPUT_DIR}")"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4,5,6,7}"
export PYTHONPATH="${REPO_ROOT}/third_party/auto-round:${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

cd "${REPO_ROOT}"

echo "Model: ${MODEL_PATH}"
echo "Output: ${OUTPUT_DIR}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
echo "AutoRound logical device map: ${AR_DEVICE_MAP:-0,1,2,3}"
echo "Log: ${LOG_FILE}"

"${PYTHON_BIN}" third_party/auto-round/scripts_laguna/quantize_laguna.py \
  --model "${MODEL_PATH}" \
  --output-dir "${OUTPUT_DIR}" \
  --expert-bits 4 \
  --rest-bits 8 \
  --group-size 64 \
  --iters "${AR_ITERS:-200}" \
  --nsamples "${AR_NSAMPLES:-128}" \
  --seqlen "${AR_SEQLEN:-2048}" \
  --batch-size "${AR_BATCH_SIZE:-1}" \
  --gradient-accumulate-steps "${AR_GRADIENT_ACCUMULATE_STEPS:-1}" \
  --device-map "${AR_DEVICE_MAP:-0,1,2,3}" \
  --dataset "${AR_DATASET:-NeelNanda/pile-10k}" \
  --format "${AR_FORMAT:-auto_round}" \
  "${@:3}" \
  2>&1 | tee "${LOG_FILE}"