#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
MODEL_PATH="${1:-/data01/datasets/Laguna-S-2.1}"
OUTPUT_DIR="${2:-${REPO_ROOT}/work_dirs/Laguna-S-2.1-quarot-autoround-expert-w4-rest-w8-g64}"
PYTHON_BIN="${AR_PYTHON:-/data01/home/xuzk/anaconda3/envs/gptqmodel/bin/python}"
LOG_DIR="${LOG_DIR:-${REPO_ROOT}/work_dirs/logs}"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/laguna-s21-quarot-autoround-expert-w4-rest-w8.log}"

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
echo "QuaRot: ${AR_ROTATION:-hadamard} on ${AR_ROTATION_DEVICE:-cuda:0}"
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
  --rotation "${AR_ROTATION:-hadamard}" \
  --rotation-device "${AR_ROTATION_DEVICE:-cuda:0}" \
  --rotation-compute-dtype "${AR_ROTATION_COMPUTE_DTYPE:-float64}" \
  --rotation-chunk-rows "${AR_ROTATION_CHUNK_ROWS:-1024}" \
  --rotation-chunk-columns "${AR_ROTATION_CHUNK_COLUMNS:-1024}" \
  --rotation-matrix-batch "${AR_ROTATION_MATRIX_BATCH:-4}" \
  --device-map "${AR_DEVICE_MAP:-0,1,2,3}" \
  --dataset "${AR_DATASET:-NeelNanda/pile-10k}" \
  --format "${AR_FORMAT:-auto_round}" \
  "${@:3}" \
  2>&1 | tee "${LOG_FILE}"

if [[ "${AR_VERIFY_CHAT:-1}" == "1" ]]; then
  echo "Verifying quantized checkpoint with a real generation..."
  "${PYTHON_BIN}" third_party/auto-round/scripts_laguna/chat_quantized.py \
    --model "${OUTPUT_DIR}" \
    --device-map "${AR_VERIFY_DEVICE_MAP:-auto}" \
    --backend "${AR_INFERENCE_BACKEND:-auto_round:tritonv2_zp}" \
    --prompt "${AR_VERIFY_PROMPT:-请用一句话介绍你自己。}" \
    --max-new-tokens "${AR_VERIFY_MAX_NEW_TOKENS:-128}" \
    --seed "${AR_VERIFY_SEED:-42}"
fi
