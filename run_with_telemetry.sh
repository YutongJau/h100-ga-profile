#!/bin/bash
# Run one portable GPU GA timing profile with nvidia-smi telemetry.
#
# This script is intended for an allocated compute node, either inside an
# interactive job or from a scheduler batch script.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PYTHON="${PYTHON:-python3}"
RUN_TAG="${RUN_TAG:-r01}"
OUT_DIR="${OUT_DIR:-${PWD}/h100_profile_outputs_$(date +%Y%m%d_%H%M%S)}"
STEPS="${STEPS:-100}"
WARMUP="${WARMUP:-20}"
MBS="${MBS:-1}"
SEQ_LEN="${SEQ_LEN:-1024}"
SEED="${SEED:-1234}"
GA_LIST="${GA_LIST:-1 2 4}"

LOG_DIR="${OUT_DIR}/logs"
TELEM_DIR="${OUT_DIR}/telemetry"
mkdir -p "${LOG_DIR}" "${TELEM_DIR}"

OUT_JSON="${OUT_DIR}/tc_profile_${RUN_TAG}.json"
PROFILE_LOG="${LOG_DIR}/profile_${RUN_TAG}.out"
PROFILE_ERR="${LOG_DIR}/profile_${RUN_TAG}.err"
ENV_LOG="${TELEM_DIR}/${RUN_TAG}.environment.txt"
DMON_LOG="${TELEM_DIR}/${RUN_TAG}.dmon.log"
QUERY_LOG="${TELEM_DIR}/${RUN_TAG}.query.csv"
SMI_BEFORE="${TELEM_DIR}/${RUN_TAG}.nvidia_smi_q_before.txt"
SMI_AFTER="${TELEM_DIR}/${RUN_TAG}.nvidia_smi_q_after.txt"

{
    echo "timestamp=$(date -Is)"
    echo "host=$(hostname)"
    echo "pwd=$(pwd)"
    echo "python=${PYTHON}"
    echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
    command -v nvidia-smi || true
    nvidia-smi -L || true
    nvidia-smi || true
    "${PYTHON}" - <<'PY' || true
import sys
print("python", sys.version)
try:
    import torch
    print("torch", torch.__version__)
    print("torch_cuda", torch.version.cuda)
    print("cuda_available", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("gpu", torch.cuda.get_device_name(0))
except Exception as exc:
    print("torch_probe_error", repr(exc))
try:
    import transformers
    print("transformers", transformers.__version__)
except Exception as exc:
    print("transformers_probe_error", repr(exc))
PY
} > "${ENV_LOG}" 2>&1

DMON_PID=""
QUERY_PID=""

cleanup_telemetry() {
    if [ -n "${DMON_PID}" ]; then
        kill "${DMON_PID}" 2>/dev/null || true
        wait "${DMON_PID}" 2>/dev/null || true
    fi
    if [ -n "${QUERY_PID}" ]; then
        kill "${QUERY_PID}" 2>/dev/null || true
        wait "${QUERY_PID}" 2>/dev/null || true
    fi
    if command -v nvidia-smi >/dev/null 2>&1; then
        nvidia-smi -q -d PERFORMANCE,CLOCK,POWER,TEMPERATURE > "${SMI_AFTER}" 2>&1 || true
    fi
}
trap cleanup_telemetry EXIT

if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi -q -d PERFORMANCE,CLOCK,POWER,TEMPERATURE > "${SMI_BEFORE}" 2>&1 || true
    nvidia-smi dmon -s pucvmt -d 1 > "${DMON_LOG}" 2>&1 &
    DMON_PID=$!
    nvidia-smi --query-gpu=timestamp,name,temperature.gpu,power.draw,power.limit,clocks.sm,clocks.mem,pstate,clocks_throttle_reasons.active --format=csv -l 1 > "${QUERY_LOG}" 2>&1 &
    QUERY_PID=$!
fi

echo "Writing profile to ${OUT_JSON}"
echo "Telemetry directory: ${TELEM_DIR}"

"${PYTHON}" "${SCRIPT_DIR}/profile_tc_portable.py" \
    --ga ${GA_LIST} \
    --steps "${STEPS}" \
    --warmup "${WARMUP}" \
    --mbs "${MBS}" \
    --seq_len "${SEQ_LEN}" \
    --seed "${SEED}" \
    --output "${OUT_JSON}" \
    > "${PROFILE_LOG}" 2> "${PROFILE_ERR}"

echo "Profile complete: ${OUT_JSON}"
