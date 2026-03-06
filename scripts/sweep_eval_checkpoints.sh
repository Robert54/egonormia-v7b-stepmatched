#!/bin/bash
# Generic parallel checkpoint sweep for EgoNormia MCQ evaluation.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

CHECKPOINT_ROOT=""
RUN_NAME=""
PYTHON_BIN="python"
EVAL_SCRIPT="${ROOT_DIR}/eval/evaluate_egonormia_vllm.py"
RESULTS_ROOT="${ROOT_DIR}/results"
TEST_PATH=""
VIDEO_BASE=""
TAXONOMY_PATH=""
THINKING=""
GPU_LIST="0,1,2,3,4,5,6,7"
STAGGER_SECS=15
MAX_NEW_TOKENS=512
TENSOR_PARALLEL_SIZE=1
NFRAMES=8
MAX_MODEL_LEN=8192

usage() {
  cat <<'EOF'
Usage:
  bash scripts/sweep_eval_checkpoints.sh \
    --checkpoint-root /path/to/output_or_safetensors \
    --run-name v6b_seed42 \
    --python /path/to/python \
    --test-path /path/to/egonormia_llava_test.json \
    --video-base /path/to/EgoNormia/video \
    --taxonomy-path /path/to/final_data.json \
    [--thinking] \
    [--gpus 0,1,2,3,4,5,6,7] \
    [--results-root /path/to/results] \
    [--max-new-tokens 512] \
    [--tensor-parallel-size 1]
EOF
}

resolve_ckpt_base() {
  local root="$1"
  if [[ -d "${root}" && "$(basename "${root}")" == "safetensors" ]]; then
    printf '%s\n' "${root}"
    return
  fi

  if [[ -d "${root}/safetensors" ]]; then
    printf '%s\n' "${root}/safetensors"
    return
  fi

  local nested
  nested="$(find "${root}" -maxdepth 2 -type d -name safetensors | sort | head -1)"
  if [[ -n "${nested}" ]]; then
    printf '%s\n' "${nested}"
    return
  fi

  return 1
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --checkpoint-root)
      CHECKPOINT_ROOT="$2"
      shift 2
      ;;
    --run-name)
      RUN_NAME="$2"
      shift 2
      ;;
    --python)
      PYTHON_BIN="$2"
      shift 2
      ;;
    --eval-script)
      EVAL_SCRIPT="$2"
      shift 2
      ;;
    --results-root)
      RESULTS_ROOT="$2"
      shift 2
      ;;
    --test-path)
      TEST_PATH="$2"
      shift 2
      ;;
    --video-base)
      VIDEO_BASE="$2"
      shift 2
      ;;
    --taxonomy-path)
      TAXONOMY_PATH="$2"
      shift 2
      ;;
    --thinking)
      THINKING="--enable_thinking"
      shift
      ;;
    --gpus)
      GPU_LIST="$2"
      shift 2
      ;;
    --stagger-secs)
      STAGGER_SECS="$2"
      shift 2
      ;;
    --max-new-tokens)
      MAX_NEW_TOKENS="$2"
      shift 2
      ;;
    --tensor-parallel-size)
      TENSOR_PARALLEL_SIZE="$2"
      shift 2
      ;;
    --nframes)
      NFRAMES="$2"
      shift 2
      ;;
    --max-model-len)
      MAX_MODEL_LEN="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown arg: $1"
      usage
      exit 1
      ;;
  esac
done

if [[ -z "${CHECKPOINT_ROOT}" || -z "${RUN_NAME}" || -z "${TEST_PATH}" || -z "${VIDEO_BASE}" || -z "${TAXONOMY_PATH}" ]]; then
  echo "ERROR: missing required arguments"
  usage
  exit 1
fi

if [[ ! -x "${PYTHON_BIN}" && ! "$(command -v "${PYTHON_BIN}" 2>/dev/null)" ]]; then
  echo "ERROR: python executable not found: ${PYTHON_BIN}"
  exit 1
fi

if [[ ! -f "${EVAL_SCRIPT}" ]]; then
  echo "ERROR: evaluator not found: ${EVAL_SCRIPT}"
  exit 1
fi

if [[ ! -f "${TEST_PATH}" ]]; then
  echo "ERROR: test json not found: ${TEST_PATH}"
  exit 1
fi

if [[ ! -d "${VIDEO_BASE}" ]]; then
  echo "ERROR: video base not found: ${VIDEO_BASE}"
  exit 1
fi

if [[ ! -f "${TAXONOMY_PATH}" ]]; then
  echo "ERROR: taxonomy json not found: ${TAXONOMY_PATH}"
  exit 1
fi

CKPT_BASE="$(resolve_ckpt_base "${CHECKPOINT_ROOT}" || true)"
if [[ -z "${CKPT_BASE}" || ! -d "${CKPT_BASE}" ]]; then
  echo "ERROR: could not resolve a safetensors directory under ${CHECKPOINT_ROOT}"
  exit 1
fi

IFS=',' read -r -a GPUS <<< "${GPU_LIST}"
NUM_GPUS="${#GPUS[@]}"
if [[ "${NUM_GPUS}" -eq 0 ]]; then
  echo "ERROR: no GPUs specified"
  exit 1
fi

THINK_SUFFIX=""
if [[ -n "${THINKING}" ]]; then
  THINK_SUFFIX="_think"
fi
RESULT_PREFIX="${RUN_NAME}${THINK_SUFFIX}"

STEPS=()
for d in "${CKPT_BASE}"/step_*; do
  if [[ -d "${d}" ]]; then
    STEP="$(basename "${d}" | sed 's/step_//')"
    STEPS+=("${STEP}")
  fi
done

if [[ "${#STEPS[@]}" -eq 0 ]]; then
  echo "ERROR: no checkpoint directories found under ${CKPT_BASE}"
  exit 1
fi

IFS=$'\n' STEPS=($(sort -n <<< "${STEPS[*]}"))
unset IFS

TODO=()
mkdir -p "${RESULTS_ROOT}"

for STEP in "${STEPS[@]}"; do
  CKPT="${CKPT_BASE}/step_${STEP}"
  OUT_DIR="${RESULTS_ROOT}/${RESULT_PREFIX}_step${STEP}"
  if [[ -f "${OUT_DIR}/summary.json" ]]; then
    echo "step_${STEP}: already done, skipping"
    continue
  fi
  TODO+=("${STEP}")
done

echo "Checkpoint base: ${CKPT_BASE}"
echo "Found ${#STEPS[@]} checkpoints"
echo "Need to evaluate ${#TODO[@]} checkpoints across ${NUM_GPUS} GPUs"
echo "Result prefix: ${RESULT_PREFIX}"
if [[ -n "${THINKING}" ]]; then
  echo "Mode: with <think> reasoning enabled"
fi
echo ""

if [[ "${#TODO[@]}" -eq 0 ]]; then
  echo "Nothing to do!"
  exit 0
fi

WAVE=0
WAVE_NUM=1
TOTAL_WAVES=$(( (${#TODO[@]} + NUM_GPUS - 1) / NUM_GPUS ))

while [[ "${WAVE}" -lt "${#TODO[@]}" ]]; do
  PIDS=()
  WAVE_END=$(( WAVE + NUM_GPUS ))
  if [[ "${WAVE_END}" -gt "${#TODO[@]}" ]]; then
    WAVE_END=${#TODO[@]}
  fi

  echo "=== Wave ${WAVE_NUM}/${TOTAL_WAVES}: steps ${TODO[@]:$WAVE:$NUM_GPUS} ==="

  GPU_IDX=0
  for (( i=WAVE; i<WAVE_END; i++ )); do
    STEP="${TODO[$i]}"
    GPU="${GPUS[$GPU_IDX]}"
    CKPT="${CKPT_BASE}/step_${STEP}"
    OUT_DIR="${RESULTS_ROOT}/${RESULT_PREFIX}_step${STEP}"
    LOG="${RESULTS_ROOT}/${RESULT_PREFIX}_step${STEP}.log"

    echo "  Launching step_${STEP} on GPU ${GPU} ..."
    CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON_BIN}" "${EVAL_SCRIPT}" \
      --model "${CKPT}" \
      --output_dir "${OUT_DIR}" \
      --test-path "${TEST_PATH}" \
      --video-base "${VIDEO_BASE}" \
      --taxonomy-path "${TAXONOMY_PATH}" \
      --nframes "${NFRAMES}" \
      --max-new-tokens "${MAX_NEW_TOKENS}" \
      --max-model-len "${MAX_MODEL_LEN}" \
      --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}" \
      ${THINKING} \
      > "${LOG}" 2>&1 &
    PIDS+=($!)
    GPU_IDX=$(( GPU_IDX + 1 ))

    if [[ "${i}" -lt $(( WAVE_END - 1 )) ]]; then
      sleep "${STAGGER_SECS}"
    fi
  done

  echo "  Waiting for wave to finish (${#PIDS[@]} jobs)..."
  wait "${PIDS[@]}"

  for (( i=WAVE; i<WAVE_END; i++ )); do
    STEP="${TODO[$i]}"
    SUMMARY="${RESULTS_ROOT}/${RESULT_PREFIX}_step${STEP}/summary.json"
    if [[ -f "${SUMMARY}" ]]; then
      "${PYTHON_BIN}" -c "
import json
s=json.load(open('${SUMMARY}'))
print(f'  step_${STEP}: action={s[\"action_accuracy\"]*100:.1f}%  both={s[\"both_accuracy\"]*100:.1f}%  s-iou={s[\"sensibility_iou\"]:.4f}  parse={s[\"parseable_rate\"]*100:.1f}%')
"
    else
      echo "  step_${STEP}: FAILED"
    fi
  done

  echo ""
  WAVE=${WAVE_END}
  WAVE_NUM=$(( WAVE_NUM + 1 ))
done

echo "=== All checkpoints evaluated! ==="
echo ""
echo "=== Leaderboard (sorted by both_accuracy) ==="
"${PYTHON_BIN}" -c "
import glob
import json
import re

results = []
for p in glob.glob('${RESULTS_ROOT}/${RESULT_PREFIX}_step*/summary.json'):
    step = int(re.search(r'step(\\d+)', p).group(1))
    s = json.load(open(p))
    results.append((step, s))

results.sort(key=lambda x: -x[1]['both_accuracy'])
print(f\"{'Step':>6s}  {'Action':>8s}  {'Justif':>8s}  {'Both':>8s}  {'S-IoU':>8s}  {'Parse':>7s}\")
print('-' * 55)
for step, s in results:
    print(f'{step:>6d}  {s[\"action_accuracy\"]*100:>7.1f}%  {s[\"justification_accuracy\"]*100:>7.1f}%  {s[\"both_accuracy\"]*100:>7.1f}%  {s[\"sensibility_iou\"]:>8.4f}  {s[\"parseable_rate\"]*100:>6.1f}%')

print()
best = results[0]
print(f'Best checkpoint: step_{best[0]} (both_accuracy={best[1][\"both_accuracy\"]*100:.1f}%)')
"
