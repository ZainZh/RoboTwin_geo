#!/usr/bin/env bash
set -euo pipefail

# Materialize one pre-registered camera-only policy dataset after the three
# full-frame NDF candidates for a fold have completed.  Model selection reads
# validation identities only; the selected heads are then rerun on the common
# nominal+recovery policy corpus before training-identity-only calibration.

if [[ $# -ne 1 ]]; then
  echo "usage: bash $0 FOLD" >&2
  exit 2
fi

fold="$1"
if [[ "${fold}" != "1" && "${fold}" != "2" ]]; then
  echo "FOLD must be 1 or 2" >&2
  exit 2
fi

python_bin="${PYTHON_BIN:-/root/miniconda3/envs/RoboTwin/bin/python}"
root="outputs/geometry_flow/goal_policy_frame_crossfold_frozen_v1"
head_dir="${root}/ndf_fold${fold}"
source_dataset="outputs/geometry_flow/task_flow_functional_frame128_cachedB_stride1.npz"
nominal_dataset="outputs/geometry_flow/goal_policy_frame_fold0_v1/task_flow_xyz128_cachedB_stride1_goal_labels.npz"
recovery_dataset="outputs/geometry_flow/goal_policy_frame_fold0_v1/task_flow_xyz128_nominal_plus_realrecovery_collapsed_train72_val8_test8_cachedAB_v4.npz"
ndf_checkpoint="/shared2/sz/model/ndf/shoe.pth"
fold_dir="${root}/fold${fold}"
mkdir -p "${fold_dir}"

selection="${fold_dir}/fold${fold}_correct_frame_valtop2_p99.npz"
"${python_bin}" -m policy.GeometryFlow.select_ndf_functional_frame_ensemble \
  --dataset "${source_dataset}" \
  --candidate "${head_dir}/fold${fold}_correct_frame_seed0.json" \
              "${head_dir}/fold${fold}_correct_frame_seed0_frames.npz" \
  --candidate "${head_dir}/fold${fold}_correct_frame_seed1.json" \
              "${head_dir}/fold${fold}_correct_frame_seed1_frames.npz" \
  --candidate "${head_dir}/fold${fold}_correct_frame_seed2.json" \
              "${head_dir}/fold${fold}_correct_frame_seed2_frames.npz" \
  --output "${selection}" \
  --fold "${fold}" \
  --confidence-percentile 99

selection_json="${selection%.npz}.json"
mapfile -t selected_seeds < <(
  sed -n '/"selected_seeds"/,/]/p' "${selection_json}" \
    | grep -E '^[[:space:]]*[0-9]+,?[[:space:]]*$' \
    | tr -d ' ,'
)
if [[ ${#selected_seeds[@]} -ne 2 ]]; then
  echo "could not read exactly two selected seeds from ${selection_json}" >&2
  exit 1
fi

materialize_camera_dataset() {
  local dataset="$1"
  local tag="$2"
  shift 2
  local prediction_args=()
  local camera_args=()
  local seed prediction ensemble partial camera raw_dir

  for seed in "${selected_seeds[@]}"; do
    prediction="${fold_dir}/fold${fold}_${tag}_ndf_seed${seed}_frames.npz"
    "${python_bin}" -m policy.GeometryFlow.predict_ndf_functional_frames \
      --dataset "${dataset}" \
      --ndf-checkpoint "${ndf_checkpoint}" \
      --frame-checkpoint "${head_dir}/fold${fold}_correct_frame_seed${seed}_checkpoint.pth" \
      --output "${prediction}" \
      --batch-size 64 \
      --device cuda:0
    prediction_args+=(--frame-predictions "${prediction}")
  done

  ensemble="${fold_dir}/fold${fold}_${tag}_ndf_valtop2_p99.npz"
  "${python_bin}" -m policy.GeometryFlow.ensemble_ndf_functional_frames \
    --dataset "${dataset}" \
    "${prediction_args[@]}" \
    --output "${ensemble}" \
    --fold "${fold}" \
    --confidence-percentile 99 \
    --aggregation projected_mean

  partial="${fold_dir}/task_flow_fold${fold}_${tag}_ndf_valtop2_p99_partial.npz"
  "${python_bin}" -m policy.GeometryFlow.augment_task_flow_with_ndf_full_frame \
    --dataset "${dataset}" \
    --frame-predictions "${ensemble}" \
    --output "${partial}" \
    --fold "${fold}"

  for raw_dir in "$@"; do
    camera_args+=(--data-dir "${raw_dir}")
  done
  camera="${fold_dir}/task_flow_fold${fold}_${tag}_ndf_valtop2_p99_camera.npz"
  "${python_bin}" -m policy.GeometryFlow.augment_task_flow_with_camera_ndf_frame \
    --dataset "${partial}" \
    "${camera_args[@]}" \
    --output "${camera}" \
    --fold "${fold}"
  echo "completed frozen fold ${fold} ${tag} camera dataset: ${camera}"
}

nominal_raw=/shared2/sz/robotwin_data/data/place_shoe_geometry_marker/demo_clean_3d_object_pc_geometry_marker/data
materialize_camera_dataset "${nominal_dataset}" nominal "${nominal_raw}"
materialize_camera_dataset \
  "${recovery_dataset}" recovery_v4 \
  "${nominal_raw}" \
  outputs/geometry_flow/goal_policy_frame_fold0_v1/pose_correction_real_train24_v2/data \
  outputs/geometry_flow/goal_policy_frame_fold0_v1/pose_correction_real_train48_v3/data \
  outputs/geometry_flow/goal_policy_frame_fold0_v1/pose_correction_real_validation8_v1/data \
  outputs/geometry_flow/goal_policy_frame_fold0_v1/pose_correction_real_test8_v1/data
