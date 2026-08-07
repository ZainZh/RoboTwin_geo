#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
RELEASE_ROOT="${SEMANTIC_FIELD_RELEASE_ROOT:-${REPO_ROOT}/include/3d_semantic_train/semantic_field_release}"
FAST_PATH="${FAST_PATH:-0}"
if [[ "${FAST_PATH}" == "1" ]]; then
  TRAIN_ENTRY="${SCRIPT_DIR}/train_semantic_field_loss_ablation_fast.py"
else
  TRAIN_ENTRY="${SCRIPT_DIR}/train_semantic_field_loss_ablation_safe.py"
fi

PYTHON_BIN="${PYTHON_BIN:-python}"
CONDA_ENV="${CONDA_ENV:-}"
DATASET_ROOT="${DATASET_ROOT:-/home/zheng/Datasets/PartNext_mesh}"
UTONIA_CHECKPOINT="${UTONIA_CHECKPOINT:-auto}"
ALIAS_CONFIG="${ALIAS_CONFIG:-${RELEASE_ROOT}/configs/hammer.json}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/outputs/paper_revision/hammer_semantic_loss_ablation}"
DEVICE="${DEVICE:-cuda:0}"
SEEDS="${SEEDS:-20260805 20260806 20260807}"
VARIANTS="${VARIANTS:-full_fixed no_ce no_supcon no_consistency}"

# These are the actual Hammer recipe values found in the checkpoint/release
# config.  In particular CE=1.0 (not the stale manuscript value 0.5).
SPLIT_SEED="${SPLIT_SEED:-42}"
VAL_RATIO="${VAL_RATIO:-0.1}"
TEST_RATIO="${TEST_RATIO:-0.0}"
EPOCHS="${EPOCHS:-4000}"
RUN_NAME_EPOCH_TAG="${RUN_NAME_EPOCH_TAG:-${EPOCHS}}"
BATCH_SIZE="${BATCH_SIZE:-6}"
NUM_WORKERS="${NUM_WORKERS:-8}"
SAVE_EVERY="${SAVE_EVERY:-${EPOCHS}}"
DRY_RUN="${DRY_RUN:-0}"
AUTO_RESUME="${AUTO_RESUME:-1}"

read -r -a SEED_ARRAY <<< "${SEEDS}"
read -r -a VARIANT_ARRAY <<< "${VARIANTS}"
if [[ -n "${CONDA_ENV}" ]]; then
  PYTHON_COMMAND=(conda run --no-capture-output -n "${CONDA_ENV}" python)
else
  if [[ "${PYTHON_BIN}" =~ [[:space:]] ]]; then
    echo "PYTHON_BIN must be one executable path; use CONDA_ENV for conda run" >&2
    exit 2
  fi
  PYTHON_COMMAND=("${PYTHON_BIN}")
fi



if [[ "${DRY_RUN}" != "1" ]]; then
  [[ -f "${TRAIN_ENTRY}" ]] || { echo "missing safe training entry: ${TRAIN_ENTRY}" >&2; exit 2; }
  [[ -f "${RELEASE_ROOT}/train/train_utonia_universal_field.py" ]] || { echo "missing release trainer under ${RELEASE_ROOT}" >&2; exit 2; }
  [[ -d "${DATASET_ROOT}/Hammer" ]] || { echo "missing Hammer dataset: ${DATASET_ROOT}/Hammer" >&2; exit 2; }
  [[ -f "${ALIAS_CONFIG}" ]] || { echo "missing Hammer alias config: ${ALIAS_CONFIG}" >&2; exit 2; }
  if [[ "${UTONIA_CHECKPOINT}" != "auto" ]]; then
    [[ -f "${UTONIA_CHECKPOINT}" ]] || { echo "missing Utonia checkpoint: ${UTONIA_CHECKPOINT}" >&2; exit 2; }
  fi
fi

declare -a COMMANDS=()
declare -a RUN_DIRS=()
for seed in "${SEED_ARRAY[@]}"; do
  [[ "${seed}" =~ ^[0-9]+$ ]] || { echo "invalid seed: ${seed}" >&2; exit 2; }
  for variant in "${VARIANT_ARRAY[@]}"; do
    case "${variant}" in
      full_fixed)
        ce_weight="1.0"; supcon_weight="0.2"; consistency_weight="0.1" ;;
      no_ce)
        ce_weight="0.0"; supcon_weight="0.2"; consistency_weight="0.1" ;;
      no_supcon)
        ce_weight="1.0"; supcon_weight="0.0"; consistency_weight="0.1" ;;
      no_consistency)
        ce_weight="1.0"; supcon_weight="0.2"; consistency_weight="0.0" ;;
      *)
        echo "unknown variant: ${variant}" >&2; exit 2 ;;
    esac

    run_name="hammer_${variant}_e${RUN_NAME_EPOCH_TAG}_split${SPLIT_SEED}_seed${seed}"
    run_dir="${OUTPUT_ROOT}/${run_name}"
    RUN_DIRS+=("${run_dir}")
    command=(
      "${PYTHON_COMMAND[@]}" "${TRAIN_ENTRY}"
      --train-mode semantic
      --dataset-root "${DATASET_ROOT}"
      --categories Hammer
      --split-seed "${SPLIT_SEED}"
      --val-ratio "${VAL_RATIO}"
      --test-ratio "${TEST_RATIO}"
      --label-level config
      --alias-config "${ALIAS_CONFIG}"
      --utonia-checkpoint "${UTONIA_CHECKPOINT}"
      --device "${DEVICE}"
      --batch-size "${BATCH_SIZE}"
      --num-workers "${NUM_WORKERS}"
      --epochs "${EPOCHS}"
      --num-support-points 5000
      --num-query-surface-points 2048
      --balanced-sampling-ratio 0
      --query-sampling-ratio 0.75
      --rotation-mode so3
      --jitter-std 0.005
      --triplane-resolution 64
      --sem-ce-weight "${ce_weight}"
      --sem-contrastive-weight "${supcon_weight}"
      --sem-consistency-weight "${consistency_weight}"
      --geo-dense-correspondence-weight 0.0
      --geo-metric-weight 0.0
      --geo-consistency-weight 0.0
      --occ-weight 0.0
      --amp
      --output-dir "${OUTPUT_ROOT}"
      --run-name "${run_name}"
      --save-every "${SAVE_EVERY}"
      --seed "${seed}"
      --wandb-mode disabled
    )
    if [[ "${AUTO_RESUME}" == "1" ]]; then
      command+=(--auto-resume)
    fi
    printf -v rendered '%q ' "${command[@]}"
    COMMANDS+=("${rendered% }")
  done
done

# Per-run Python locks validate complete targets and reject incomplete or
# mismatched targets.  This also makes already-running matrix processes safely
# skip variants completed by concurrently launched services.
if [[ "${DRY_RUN}" != "1" ]]; then
  mkdir -p "${OUTPUT_ROOT}"
fi

for command in "${COMMANDS[@]}"; do
  if [[ "${DRY_RUN}" == "1" ]]; then
    printf '%s\n' "${command}"
  else
    echo "[hammer-loss-ablation] ${command}"
    # Non-login shell preserves an already activated conda environment.
    SEMANTIC_FIELD_RELEASE_ROOT="${RELEASE_ROOT}" bash -c "${command}"
  fi
done
