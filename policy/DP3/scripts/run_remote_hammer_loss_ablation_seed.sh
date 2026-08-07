#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 SEED OUTPUT_ROOT" >&2
  exit 2
fi

SEED="$1"
OUTPUT_ROOT="$2"
[[ "${SEED}" =~ ^[0-9]+$ ]] || { echo "invalid seed: ${SEED}" >&2; exit 2; }

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUNNER="${SCRIPT_DIR}/run_hammer_semantic_loss_ablation_matrix.sh"
PYTHON_BIN="${PYTHON_BIN:-/workspace/venvs/geo-utonia/bin/python}"
DATASET_ROOT="${DATASET_ROOT:-/workspace/data/PartNext_mesh}"
UTONIA_ROOT="${UTONIA_ROOT:-/workspace/Utonia}"
UTONIA_CHECKPOINT="${UTONIA_CHECKPOINT:-/workspace/checkpoints/utonia/utonia.pth}"
SEMANTIC_FIELD_RELEASE_ROOT="${SEMANTIC_FIELD_RELEASE_ROOT:-/workspace/RoboTwin_geo/include/3d_semantic_train/semantic_field_release}"
EPOCHS="${EPOCHS:-4000}"
RUN_NAME_EPOCH_TAG="${RUN_NAME_EPOCH_TAG:-${EPOCHS}}"
BATCH_SIZE="${BATCH_SIZE:-6}"
NUM_WORKERS="${NUM_WORKERS:-8}"
DEVICE="${DEVICE:-cuda:0}"
VARIANTS="${VARIANTS:-full_fixed no_ce no_supcon no_consistency}"
FAST_PATH="${FAST_PATH:-0}"

mkdir -p "${OUTPUT_ROOT}/completed" "${OUTPUT_ROOT}/manifests" "${OUTPUT_ROOT}/logs"
read -r -a VARIANT_ARRAY <<< "${VARIANTS}"

for variant in "${VARIANT_ARRAY[@]}"; do
  run_name="hammer_${variant}_e${RUN_NAME_EPOCH_TAG}_split42_seed${SEED}"
  run_dir="${OUTPUT_ROOT}/${run_name}"
  completion="${OUTPUT_ROOT}/completed/${run_name}.complete"
  log_path="${OUTPUT_ROOT}/logs/${run_name}.log"
  manifest="${OUTPUT_ROOT}/manifests/${run_name}.sha256"

  if [[ -f "${completion}" ]]; then
    echo "[remote-seed] already complete: ${run_name}"
    continue
  fi

  echo "[remote-seed] starting: ${run_name}"
  env \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 \
    PYTHON_BIN="${PYTHON_BIN}" \
    DATASET_ROOT="${DATASET_ROOT}" \
    UTONIA_ROOT="${UTONIA_ROOT}" \
    UTONIA_CHECKPOINT="${UTONIA_CHECKPOINT}" \
    SEMANTIC_FIELD_RELEASE_ROOT="${SEMANTIC_FIELD_RELEASE_ROOT}" \
    OUTPUT_ROOT="${OUTPUT_ROOT}" \
    SEEDS="${SEED}" \
    VARIANTS="${variant}" \
    EPOCHS="${EPOCHS}" \
    RUN_NAME_EPOCH_TAG="${RUN_NAME_EPOCH_TAG}" \
    BATCH_SIZE="${BATCH_SIZE}" \
    NUM_WORKERS="${NUM_WORKERS}" \
    SAVE_EVERY="${EPOCHS}" \
    DEVICE="${DEVICE}" \
    FAST_PATH="${FAST_PATH}" \
    bash "${RUNNER}" 2>&1 | tee "${log_path}"

  manifest_tmp="${manifest}.tmp-${BASHPID}"
  completion_tmp="${completion}.tmp-${BASHPID}"
  trap 'rm -f "${manifest_tmp}" "${completion_tmp}"' EXIT
  find "${run_dir}" -type f -print0 | sort -z | xargs -0 sha256sum > "${manifest_tmp}"
  mv "${manifest_tmp}" "${manifest}"
  {
    date -Is
    echo "seed=${SEED}"
    echo "variant=${variant}"
    echo "epochs=${EPOCHS}"
    echo "artifact_epoch_tag=${RUN_NAME_EPOCH_TAG}"
    sha256sum "${manifest}"
  } > "${completion_tmp}"
  mv "${completion_tmp}" "${completion}"
  trap - EXIT
  echo "[remote-seed] complete: ${run_name}"
done

all_complete=1
for variant in full_fixed no_ce no_supcon no_consistency; do
  run_name="hammer_${variant}_e${RUN_NAME_EPOCH_TAG}_split42_seed${SEED}"
  [[ -f "${OUTPUT_ROOT}/completed/${run_name}.complete" ]] || all_complete=0
done
if [[ "${all_complete}" == "1" ]]; then
  seed_completion="${OUTPUT_ROOT}/completed/seed_${SEED}.complete"
  seed_completion_tmp="${seed_completion}.tmp-${BASHPID}"
  date -Is > "${seed_completion_tmp}"
  mv "${seed_completion_tmp}" "${seed_completion}"
  echo "[remote-seed] all variants complete: seed=${SEED}"
else
  echo "[remote-seed] subset complete; seed marker deferred: seed=${SEED}"
fi
