#!/bin/bash
# Portable launcher for the primary v7b MCQ-only step-matched SFT run.

set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
TEMPLATE="${ROOT_DIR}/configs/egonormia_sft_v7b_cot_mcq3_stepmatched.toml.template"

COSMOS_RL_DIR=""
DATA_JSON=""
MEDIA_DIR=""
OUTPUT_ROOT="${ROOT_DIR}/outputs"
SEED=42

usage() {
  cat <<'EOF'
Usage:
  bash scripts/run_sft_v7b_cot_mcq3_stepmatched.sh \
    --cosmos-rl-dir /path/to/cosmos-reason2/examples/cosmos_rl \
    --data-json /path/to/egonormia_llava_v7_cot_mcq3_train.json \
    --media-dir /path/to/EgoNormia/video \
    [--output-root /path/to/outputs] \
    [--seed 42|1234]
EOF
}

escape_sed() {
  printf '%s' "$1" | sed -e 's/[\/&]/\\&/g'
}

load_hf_token() {
  if [[ -n "${HF_TOKEN:-}" ]]; then
    export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN}"
    return
  fi

  HF_TOKEN="$(
    grep -E '^[[:space:]]*(export[[:space:]]+)?HF_TOKEN=' ~/.bashrc 2>/dev/null \
      | head -1 \
      | sed -E "s/^[[:space:]]*(export[[:space:]]+)?HF_TOKEN=[[:space:]]*//; s/^['\\\"]//; s/['\\\"]$//"
  )"

  if [[ -z "${HF_TOKEN:-}" ]]; then
    echo "ERROR: HF_TOKEN is not set and was not found in ~/.bashrc"
    exit 1
  fi

  export HF_TOKEN
  export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN}"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --cosmos-rl-dir)
      COSMOS_RL_DIR="$2"
      shift 2
      ;;
    --data-json)
      DATA_JSON="$2"
      shift 2
      ;;
    --media-dir)
      MEDIA_DIR="$2"
      shift 2
      ;;
    --output-root)
      OUTPUT_ROOT="$2"
      shift 2
      ;;
    --seed)
      SEED="$2"
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

if [[ -z "${COSMOS_RL_DIR}" || -z "${DATA_JSON}" || -z "${MEDIA_DIR}" ]]; then
  echo "ERROR: --cosmos-rl-dir, --data-json, and --media-dir are required"
  usage
  exit 1
fi

if [[ ! -d "${COSMOS_RL_DIR}" ]]; then
  echo "ERROR: cosmos_rl directory not found: ${COSMOS_RL_DIR}"
  exit 1
fi

if [[ ! -f "${DATA_JSON}" ]]; then
  echo "ERROR: data json not found: ${DATA_JSON}"
  exit 1
fi

if [[ ! -d "${MEDIA_DIR}" ]]; then
  echo "ERROR: media directory not found: ${MEDIA_DIR}"
  exit 1
fi

if [[ ! -f "${TEMPLATE}" ]]; then
  echo "ERROR: config template not found: ${TEMPLATE}"
  exit 1
fi

if ! command -v uv >/dev/null 2>&1; then
  echo "ERROR: uv is not installed or not on PATH"
  exit 1
fi

load_hf_token

RUN_NAME="egonormia_sft_v7b_cot_mcq3_stepmatched_seed${SEED}"
OUTPUT_DIR="${OUTPUT_ROOT}/${RUN_NAME}"
mkdir -p "${OUTPUT_ROOT}"

TMP_CONFIG="$(mktemp /tmp/${RUN_NAME}_XXXXXX.toml)"
trap 'rm -f "${TMP_CONFIG}"' EXIT

sed \
  -e "s/__ANNOTATION_PATH__/$(escape_sed "${DATA_JSON}")/" \
  -e "s/__MEDIA_PATH__/$(escape_sed "${MEDIA_DIR}")/" \
  -e "s/__OUTPUT_DIR__/$(escape_sed "${OUTPUT_DIR}")/" \
  -e "s/__EXPERIMENT_NAME__/$(escape_sed "${RUN_NAME}")/" \
  -e "s/__SEED__/${SEED}/" \
  "${TEMPLATE}" > "${TMP_CONFIG}"

echo "Starting v7b step-matched SFT..."
echo "cosmos_rl: ${COSMOS_RL_DIR}"
echo "data_json:  ${DATA_JSON}"
echo "media_dir:  ${MEDIA_DIR}"
echo "output_dir: ${OUTPUT_DIR}"
echo "seed:       ${SEED}"

uv run --no-sync --directory "${COSMOS_RL_DIR}" \
  cosmos-rl \
  --config "${TMP_CONFIG}" \
  scripts/llava_sft.py
