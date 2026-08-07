#!/usr/bin/env bash
set -euo pipefail

# Reproduce the dense NDF functional-frame route on object-disjoint folds 1--4.
#
# The script deliberately writes to a versioned root and never passes
# --overwrite to a Python producer.  A complete unit is skipped on a repeated
# invocation; a partially populated unit is treated as an error so that a
# damaged result cannot silently be mixed with a new run.

dataset="${1:-outputs/geometry_flow/remote4090/task_flow_remote50_xyz128_stride1.npz}"
queue_root="${2:-outputs/geometry_flow/remote4090/dense_ndf_crossfold_v1}"
data_dir="${3:-data/place_shoe_geometry_marker/demo_clean_3d_object_pc_geometry_marker_remote50/data}"
ndf_checkpoint="${4:-/home/zheng/model/ndf/shoe.pth}"

python_bin="${ROBOTWIN_PYTHON:-/home/zheng/miniforge3/envs/RoboTwin/bin/python}"
gpu="${DENSE_NDF_GPU:-0}"
max_parallel="${DENSE_NDF_MAX_PARALLEL:-2}"
cuda_memory_fraction="${DENSE_NDF_CUDA_MEMORY_FRACTION:-0.48}"
existing_output_root="${DENSE_NDF_EXISTING_OUTPUT_ROOT:-outputs/geometry_flow/remote4090}"
fold_spec="${DENSE_NDF_FOLDS:-1 2 3 4}"
read -r -a folds <<< "${fold_spec}"
seeds=(0 1 2)

die() {
  printf 'error: %s\n' "$*" >&2
  exit 1
}

[[ -x "${python_bin}" ]] || die "Python is not executable: ${python_bin}"
[[ -f "${dataset}" ]] || die "dataset does not exist: ${dataset}"
[[ -d "${data_dir}" ]] || die "raw data directory does not exist: ${data_dir}"
[[ -f "${ndf_checkpoint}" ]] || die "NDF checkpoint does not exist: ${ndf_checkpoint}"
[[ "${max_parallel}" =~ ^[12]$ ]] || die "DENSE_NDF_MAX_PARALLEL must be 1 or 2"
[[ ${#folds[@]} -gt 0 ]] || die "DENSE_NDF_FOLDS must contain at least one fold"
for fold in "${folds[@]}"; do
  [[ "${fold}" =~ ^[1-4]$ ]] || die "folds must be selected from 1 2 3 4, got ${fold}"
done

mkdir -p "${queue_root}"
exec 9>"${queue_root}/.queue.lock"
flock -n 9 || die "another dense NDF cross-fold queue owns ${queue_root}"

# Fail before allocating a GPU if a dataset is row-misaligned or lacks one of
# the fields consumed by the frame trainer and the two policy augmenters.
"${python_bin}" - "${dataset}" "${data_dir}" <<'PY'
import json
from pathlib import Path
import sys

import h5py
import numpy as np

dataset = Path(sys.argv[1])
data_dir = Path(sys.argv[2])
sample_fields = {
    "points_a",
    "points_b",
    "state",
    "action",
    "current_anchors",
    "current_object_pose9",
    "active_arm_right",
    "episode_index",
    "episode_id",
    "shoe_id",
    "relation_phase",
    "goal_translation_error_xyz_m",
    "goal_rotation_error_rotvec",
}
episode_fields = {"episode_flow", "episode_pose", "episode_shoe_id"}
with np.load(dataset, allow_pickle=False) as archive:
    missing = sorted((sample_fields | episode_fields) - set(archive.files))
    if missing:
        raise KeyError(f"dataset is missing required fields: {missing}")
    sample_count = len(archive["shoe_id"])
    bad_rows = {
        key: list(archive[key].shape)
        for key in sample_fields
        if len(archive[key]) != sample_count
    }
    if bad_rows:
        raise ValueError(f"sample-row alignment mismatch: {bad_rows}")
    if archive["points_a"].ndim != 3 or archive["points_a"].shape[-1] < 3:
        raise ValueError(f"points_a must be [samples,points,C>=3], got {archive['points_a'].shape}")
    if archive["current_object_pose9"].shape != (sample_count, 9):
        raise ValueError("current_object_pose9 must have shape [samples,9]")
    for key in ("goal_translation_error_xyz_m", "goal_rotation_error_rotvec"):
        if archive[key].shape != (sample_count, 3):
            raise ValueError(f"{key} must have shape [samples,3]")
    episode_shoe = np.asarray(archive["episode_shoe_id"], dtype=np.int64)
    episode_index = np.asarray(archive["episode_index"], dtype=np.int64)
    episode_count = len(episode_shoe)
    if not np.array_equal(np.unique(episode_index), np.arange(episode_count)):
        raise ValueError("episode_index must densely cover [0, episode_count)")
    for key in ("episode_flow", "episode_pose"):
        if len(archive[key]) != episode_count:
            raise ValueError(f"{key} is not aligned with episode_shoe_id")

marker_frames = 0
for episode in range(episode_count):
    path = data_dir / f"episode{episode}.hdf5"
    if not path.is_file():
        raise FileNotFoundError(path)
    with h5py.File(path, "r") as archive:
        key = "object_pointcloud/{B}"
        if key not in archive or len(archive[key]) == 0:
            raise KeyError(f"{path} lacks a nonempty {key}")
        marker_frames += len(archive[key])

print(json.dumps({
    "preflight": "ok",
    "dataset": str(dataset),
    "samples": sample_count,
    "episodes": episode_count,
    "raw_B_frames": marker_frames,
}))
PY

if [[ "${DENSE_NDF_PREFLIGHT_ONLY:-0}" == "1" ]]; then
  printf 'dense NDF cross-fold preflight complete\n'
  exit 0
fi

while pgrep -f '[p]olicy.GeometryFlow.train_ndf_functional_frame_head' >/dev/null; do
  if [[ "${DENSE_NDF_WAIT_FOR_EXISTING:-1}" != "1" ]]; then
    die "an NDF functional-frame trainer is already running"
  fi
  printf 'wait for existing NDF functional-frame trainers before entering max-2 queue\n'
  sleep 30
done

legacy_run_dir() {
  local fold="$1"
  printf '%s/ndf_functional_frame_dense_fold%s_seeds012' \
    "${existing_output_root}" "${fold}"
}

legacy_run_complete() {
  local fold="$1"
  local seed="$2"
  local run_dir
  run_dir="$(legacy_run_dir "${fold}")"
  local stem="fold${fold}_correct_frame_seed${seed}"
  [[ -f "${run_dir}/${stem}_checkpoint.pth" \
    && -f "${run_dir}/${stem}_frames.npz" \
    && -f "${run_dir}/${stem}.json" ]]
}

train_one() {
  local fold="$1"
  local seed="$2"
  local fold_root="${queue_root}/fold${fold}"
  local run_dir="${fold_root}/ndf_seed${seed}"
  local stem="fold${fold}_correct_frame_seed${seed}"
  local checkpoint_path="${run_dir}/${stem}_checkpoint.pth"
  local frames_path="${run_dir}/${stem}_frames.npz"
  local result_path="${run_dir}/${stem}.json"

  if legacy_run_complete "${fold}" "${seed}"; then
    printf 'reuse complete existing NDF frame run fold=%s seed=%s path=%s\n' \
      "${fold}" "${seed}" "$(legacy_run_dir "${fold}")"
    return 0
  fi
  if [[ -f "${checkpoint_path}" && -f "${frames_path}" && -f "${result_path}" ]]; then
    printf 'skip complete NDF frame run fold=%s seed=%s\n' "${fold}" "${seed}"
    return 0
  fi
  if [[ -e "${run_dir}" ]]; then
    printf 'refuse partial NDF frame run fold=%s seed=%s path=%s\n' \
      "${fold}" "${seed}" "${run_dir}" >&2
    return 2
  fi

  mkdir -p "${run_dir}"
  printf 'start NDF frame run fold=%s seed=%s\n' "${fold}" "${seed}"
  CUDA_VISIBLE_DEVICES="${gpu}" \
    OMP_NUM_THREADS=4 \
    MKL_NUM_THREADS=4 \
    PYTHONUNBUFFERED=1 \
    "${python_bin}" -m policy.GeometryFlow.train_ndf_functional_frame_head \
    --dataset "${dataset}" \
    --checkpoint "${ndf_checkpoint}" \
    --output-dir "${run_dir}" \
    --folds "${fold}" \
    --seeds "${seed}" \
    --conditions correct_frame \
    --epochs 200 \
    --patience 30 \
    --batch-size 8 \
    --learning-rate 0.0003 \
    --weight-decay 0.0001 \
    --axis-loss-weight 0.5 \
    --orthogonality-loss-weight 0.05 \
    --trainable-scope full \
    --device cuda:0 \
    --cuda-memory-fraction "${cuda_memory_fraction}" \
    > "${run_dir}/train.log" 2>&1

  [[ -f "${checkpoint_path}" ]] || return 3
  [[ -f "${frames_path}" ]] || return 3
  [[ -f "${result_path}" ]] || return 3
  printf 'complete NDF frame run fold=%s seed=%s\n' "${fold}" "${seed}"
}

# Launch fixed-size batches. This is intentionally capped at two simultaneous
# trainers even if the caller requests a larger value.
pids=()
labels=()
wait_batch() {
  local failed=0
  local index
  for index in "${!pids[@]}"; do
    if ! wait "${pids[${index}]}"; then
      printf 'training unit failed: %s\n' "${labels[${index}]}" >&2
      failed=1
    fi
  done
  pids=()
  labels=()
  [[ "${failed}" -eq 0 ]]
}

for fold in "${folds[@]}"; do
  for seed in "${seeds[@]}"; do
    train_one "${fold}" "${seed}" &
    pids+=("$!")
    labels+=("fold${fold}/seed${seed}")
    if [[ ${#pids[@]} -ge ${max_parallel} ]]; then
      wait_batch || die "one or more NDF frame runs failed"
    fi
  done
done
if [[ ${#pids[@]} -gt 0 ]]; then
  wait_batch || die "one or more NDF frame runs failed"
fi

produce_pair() {
  local output="$1"
  shift
  local metadata="${output%.npz}.json"
  if [[ -f "${output}" && -f "${metadata}" ]]; then
    printf 'skip complete dataset %s\n' "${output}"
    return 0
  fi
  if [[ -e "${output}" || -e "${metadata}" ]]; then
    die "refuse partial output pair: ${output} ${metadata}"
  fi
  "$@"
  [[ -f "${output}" && -f "${metadata}" ]] \
    || die "producer did not create output pair: ${output} ${metadata}"
}

for fold in "${folds[@]}"; do
  fold_root="${queue_root}/fold${fold}"
  ensemble="${fold_root}/fold${fold}_correct_frame_ensemble_p95.npz"
  partial="${fold_root}/task_flow_fold${fold}_ndf_ensemble_p95_partial.npz"
  camera="${fold_root}/task_flow_fold${fold}_ndf_ensemble_p95_camera_only.npz"
  predictions=()
  for seed in "${seeds[@]}"; do
    prediction="${fold_root}/ndf_seed${seed}/fold${fold}_correct_frame_seed${seed}_frames.npz"
    if legacy_run_complete "${fold}" "${seed}"; then
      prediction="$(legacy_run_dir "${fold}")/fold${fold}_correct_frame_seed${seed}_frames.npz"
    fi
    [[ -f "${prediction}" ]] || die "missing frame predictions: ${prediction}"
    predictions+=(
      --frame-predictions
      "${prediction}"
    )
  done

  produce_pair "${ensemble}" \
    "${python_bin}" -m policy.GeometryFlow.ensemble_ndf_functional_frames \
    --dataset "${dataset}" \
    "${predictions[@]}" \
    --output "${ensemble}" \
    --fold "${fold}" \
    --confidence-percentile 95

  produce_pair "${partial}" \
    "${python_bin}" -m policy.GeometryFlow.augment_task_flow_with_ndf_full_frame \
    --dataset "${dataset}" \
    --frame-predictions "${ensemble}" \
    --output "${partial}" \
    --fold "${fold}"

  produce_pair "${camera}" \
    "${python_bin}" -m policy.GeometryFlow.augment_task_flow_with_camera_ndf_frame \
    --dataset "${partial}" \
    --data-dir "${data_dir}" \
    --output "${camera}" \
    --fold "${fold}"
done

printf 'dense NDF cross-fold queue complete: %s\n' "${queue_root}"
