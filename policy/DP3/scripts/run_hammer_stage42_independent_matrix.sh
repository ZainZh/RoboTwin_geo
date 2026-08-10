#!/usr/bin/env bash
# Frozen launcher for the independent held-out Hammer stage42 matrix.
#
# Default behavior is command-only dry-run.  Training can start only when the
# caller explicitly sets RUN_STAGE42=1.  Scientific arguments are assigned
# below rather than inherited, so stale SPLIT_SEED/TEST_RATIO variables cannot
# silently turn this back into the legacy validation matrix.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
MATRIX_LAUNCHER="${SCRIPT_DIR}/run_hammer_semantic_loss_ablation_matrix.sh"

ROLE="${STAGE42_ROLE:-}"
RUN_GATE="${RUN_STAGE42:-0}"
SELECTED_SEEDS="${STAGE42_SEEDS:-}"
SELECTED_VARIANTS="${STAGE42_VARIANTS:-full_fixed no_ce no_supcon no_consistency ce_only}"

case "${RUN_GATE}" in
  0) DRY_RUN_VALUE=1 ;;
  1) DRY_RUN_VALUE=0 ;;
  *) echo "RUN_STAGE42 must be exactly 0 or 1" >&2; exit 2 ;;
esac

case "${ROLE}" in
  local)
    ALLOWED_SEEDS="20260805 20260807"
    DEFAULT_SEEDS="${ALLOWED_SEEDS}"
    OUTPUT_TEMPLATE="${REPO_ROOT}/outputs/paper_revision/hammer_stage42_independent_local4090_seed%s_e4000_split424242_20260806"
    PYTHON_VALUE="/home/zheng/miniforge3/envs/RoboTwin/bin/python"
    DATASET_VALUE="/home/zheng/Datasets/PartNext_mesh"
    UTONIA_ROOT_VALUE="${REPO_ROOT}/include/3d_semantic_train/include/Utonia"
    UTONIA_CHECKPOINT_VALUE="/home/zheng/.cache/utonia/ckpt/utonia.pth"
    RELEASE_VALUE="${REPO_ROOT}/include/3d_semantic_train/semantic_field_release"
    WORKERS_VALUE="${STAGE42_NUM_WORKERS:-4}"
    ;;
  remote)
    ALLOWED_SEEDS="20260806"
    DEFAULT_SEEDS="${ALLOWED_SEEDS}"
    OUTPUT_TEMPLATE="/workspace/RoboTwin_geo/outputs/paper_revision/hammer_stage42_independent_remote6000ada_seed%s_e4000_split424242_20260806"
    PYTHON_VALUE="/workspace/venvs/geo-utonia/bin/python"
    DATASET_VALUE="/workspace/data/PartNext_mesh"
    UTONIA_ROOT_VALUE="/workspace/Utonia"
    UTONIA_CHECKPOINT_VALUE="/workspace/checkpoints/utonia/utonia.pth"
    RELEASE_VALUE="/workspace/RoboTwin_geo/include/3d_semantic_train/semantic_field_release"
    WORKERS_VALUE="${STAGE42_NUM_WORKERS:-8}"
    ;;
  *)
    echo "STAGE42_ROLE must be explicitly set to local or remote" >&2
    exit 2
    ;;
esac

[[ "${WORKERS_VALUE}" =~ ^[0-9]+$ ]] || { echo "invalid STAGE42_NUM_WORKERS" >&2; exit 2; }
if [[ -z "${SELECTED_SEEDS}" ]]; then
  SELECTED_SEEDS="${DEFAULT_SEEDS}"
fi

read -r -a seed_array <<< "${SELECTED_SEEDS}"
read -r -a variant_array <<< "${SELECTED_VARIANTS}"
read -r -a allowed_seed_array <<< "${ALLOWED_SEEDS}"

for seed in "${seed_array[@]}"; do
  allowed=0
  for candidate in "${allowed_seed_array[@]}"; do
    [[ "${seed}" == "${candidate}" ]] && allowed=1
  done
  [[ "${allowed}" == "1" ]] || {
    echo "seed ${seed} is not assigned to STAGE42_ROLE=${ROLE}" >&2
    exit 2
  }
done
for variant in "${variant_array[@]}"; do
  case "${variant}" in
    full_fixed|no_ce|no_supcon|no_consistency|ce_only) ;;
    *) echo "unknown stage42 variant: ${variant}" >&2; exit 2 ;;
  esac
done

for seed in "${seed_array[@]}"; do
  printf -v output_root "${OUTPUT_TEMPLATE}" "${seed}"
  env \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 \
    PYTHON_BIN="${PYTHON_VALUE}" \
    DATASET_ROOT="${DATASET_VALUE}" \
    UTONIA_ROOT="${UTONIA_ROOT_VALUE}" \
    UTONIA_CHECKPOINT="${UTONIA_CHECKPOINT_VALUE}" \
    SEMANTIC_FIELD_RELEASE_ROOT="${RELEASE_VALUE}" \
    OUTPUT_ROOT="${output_root}" \
    SEEDS="${seed}" \
    VARIANTS="${SELECTED_VARIANTS}" \
    SPLIT_SEED=424242 \
    VAL_RATIO=0.1 \
    TEST_RATIO=0.2 \
    EPOCHS=1750 \
    RUN_NAME_EPOCH_TAG=4000 \
    BATCH_SIZE=6 \
    NUM_WORKERS="${WORKERS_VALUE}" \
    SAVE_EVERY=1750 \
    DEVICE=cuda:0 \
    FAST_PATH=1 \
    SEMANTIC_FAST_PATH=1 \
    AUTO_RESUME=1 \
    DRY_RUN="${DRY_RUN_VALUE}" \
    bash "${MATRIX_LAUNCHER}"
done
