#!/usr/bin/env bash
set -euo pipefail

VARIANT="${1:-}"
case "${VARIANT}" in
  full_fixed|no_ce|no_supcon|no_consistency) ;;
  *)
    echo "usage: $0 {full_fixed|no_ce|no_supcon|no_consistency}" >&2
    exit 2 ;;
esac

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/outputs/paper_revision/hammer_semantic_loss_ablation_local4090_seed20260807_e4000_fastw4_resumable_restart20260806}"
RUN_NAME="hammer_${VARIANT}_e4000_split42_seed20260807"
RESUME_PATH="${OUTPUT_ROOT}/${RUN_NAME}/resume.pt"

if [[ ! -f "${RESUME_PATH}" && ! -f "${OUTPUT_ROOT}/${RUN_NAME}/completion.json" ]]; then
  echo "neither resumable nor complete run exists: ${OUTPUT_ROOT}/${RUN_NAME}" >&2
  exit 3
fi

export PYTHON_BIN="${PYTHON_BIN:-/home/zheng/miniforge3/envs/RoboTwin/bin/python}"
export DATASET_ROOT="${DATASET_ROOT:-/home/zheng/Datasets/PartNext_mesh}"
export UTONIA_CHECKPOINT="${UTONIA_CHECKPOINT:-/home/zheng/.cache/utonia/ckpt/utonia.pth}"
export UTONIA_ROOT="${UTONIA_ROOT:-${REPO_ROOT}/include/3d_semantic_train/include/Utonia}"
export OUTPUT_ROOT
export SEEDS=20260807
export VARIANTS="${VARIANT}"
export FAST_PATH=1
export EPOCHS=1750
export RUN_NAME_EPOCH_TAG=4000
export SAVE_EVERY=1750
export AUTO_RESUME=1
export NUM_WORKERS=0
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1

exec /bin/bash "${SCRIPT_DIR}/run_hammer_semantic_loss_ablation_matrix.sh"
