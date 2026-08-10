#!/usr/bin/env bash
set -euo pipefail

POLL_SECONDS="${POLL_SECONDS:-60}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/zheng/miniforge3/envs/RoboTwin/bin/python}"
OUTPUTS="${REPO_ROOT}/outputs/paper_revision"
ROOT05="${OUTPUTS}/hammer_stage42_independent_local4090_seed20260805_e4000_split424242_20260806"
ROOT06="${OUTPUTS}/hammer_stage42_independent_remote6000ada_seed20260806_e4000_split424242_20260806"
ROOT07="${OUTPUTS}/hammer_stage42_independent_local4090_seed20260807_e4000_split424242_20260806"
EVAL_ROOT="${OUTPUTS}/hammer_stage42_independent_test_evaluation_e1750_v1"
MANIFEST="${OUTPUTS}/hammer_stage42_e1750_15run_manifest_v1/manifest.json"
SUMMARY_ROOT="${OUTPUTS}/hammer_stage42_independent_test_summary_e1750_v1"
RUN05="hammer_ce_only_e4000_split424242_seed20260805"
RUN06="hammer_ce_only_e4000_split424242_seed20260806"
RUN07="hammer_ce_only_e4000_split424242_seed20260807"

if ! [[ "${POLL_SECONDS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "POLL_SECONDS must be a positive integer" >&2
  exit 2
fi

required=(
  "${ROOT05}/${RUN05}/completion.json"
  "${ROOT06}/${RUN06}/completion.json"
  "${ROOT07}/${RUN07}/completion.json"
  "${EVAL_ROOT}/seed20260805_ce_only/r1_test_r5_fp32/results.json"
  "${EVAL_ROOT}/seed20260805_ce_only/r2_test_t5_fp32/results.json"
  "${EVAL_ROOT}/seed20260806_ce_only/r1_test_r5_fp32/results.json"
  "${EVAL_ROOT}/seed20260806_ce_only/r2_test_t5_fp32/results.json"
  "${EVAL_ROOT}/seed20260807_ce_only/r1_test_r5_fp32/results.json"
  "${EVAL_ROOT}/seed20260807_ce_only/r2_test_t5_fp32/results.json"
)
while true; do
  missing=0
  for path in "${required[@]}"; do
    [[ -f "${path}" ]] || missing=1
  done
  [[ "${missing}" == "0" ]] && break
  sleep "${POLL_SECONDS}"
done

export CUDA_VISIBLE_DEVICES=""
"${PYTHON_BIN}" "${SCRIPT_DIR}/prepare_hammer_loss_ablation_evaluation.py" \
  --seed-root "20260805=${ROOT05}" \
  --seed-root "20260806=${ROOT06}" \
  --seed-root "20260807=${ROOT07}" \
  --mode validate --epochs 1750 --artifact-epoch-tag 4000 \
  --split-seed 424242 --evaluation-protocol independent_test \
  --expected-val-ratio 0.1 --expected-test-ratio 0.2 \
  --manifest "${MANIFEST}" --python-bin "${PYTHON_BIN}" \
  --dataset-root /home/zheng/Datasets/PartNext_mesh \
  --alias-config "${REPO_ROOT}/include/3d_semantic_train/semantic_field_release/configs/hammer.json" \
  --utonia-checkpoint /home/zheng/.cache/utonia/ckpt/utonia.pth \
  --evaluation-output-root "${EVAL_ROOT}" --device cuda:0 \
  --evaluation-seed 20260805

if [[ -f "${SUMMARY_ROOT}/summary.json" ]]; then
  echo "[stage42-finalize] summary already exists after successful validation: ${SUMMARY_ROOT}/summary.json"
  exit 0
fi
if [[ -e "${SUMMARY_ROOT}" ]] && [[ -n "$(find "${SUMMARY_ROOT}" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
  echo "refusing non-empty incomplete summary directory: ${SUMMARY_ROOT}" >&2
  exit 1
fi
"${PYTHON_BIN}" "${SCRIPT_DIR}/summarize_hammer_loss_ablation_evaluation.py" \
  --manifest "${MANIFEST}" --evaluation-root "${EVAL_ROOT}" \
  --output-dir "${SUMMARY_ROOT}"
echo "[stage42-finalize] validated and summarized 15 runs / 30 evaluations"
