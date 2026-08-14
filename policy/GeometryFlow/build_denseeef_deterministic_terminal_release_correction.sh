#!/usr/bin/env bash
set -euo pipefail

python_bin="${PYTHON_BIN:-/root/miniconda3/envs/RoboTwin/bin/python}"
artifact_root="${TAGRT_ARTIFACT_ROOT:-/shared2/sz/TAGRT-v1}"
source_root="${TAGRT_SOURCE_ARTIFACT_ROOT:-/shared2/sz/TAGRT-v1}"
confidence_threshold_deg="${NDF_CONFIDENCE_THRESHOLD_DEG:-4.383880233764648}"
control_stride="${CONTROL_STRIDE:-15}"
run_name="${RUN_NAME:-denseeef_left60_pi_deterministic_terminal_seed173_fixedconf_v1}"
root="${artifact_root}/processed_datasets/fulltask_tagrt_v2/${run_name}"
sim_root="${SIM_ROOT:-${artifact_root}/simulator_data/place_shoe_geometry_marker/fulltask_tagrt_pi_deterministic_gripper_terminal_seed173_v2}"
data_dir="${sim_root}/data"
ndf_checkpoint="${source_root}/models/ndf/shoe.pth"
frame_root="${source_root}/models/ndf_functional_frames/fold0_local50_seeds012"
calibration="${source_root}/processed_datasets/fulltask_tagrt_v2/gate2_left60/task_flow_ndf_camera.json"
mkdir -p "${root}"

base="${root}/task_flow_base.npz"
"${python_bin}" -m policy.GeometryFlow.build_task_flow_dataset \
  --data-dir "${data_dir}" --episodes 1 --observation-points 128 \
  --point-channels 6 --target-color-priority --cache-target-observation \
  --cache-operated-observation --target-frame-token --anchors 32 \
  --flow-steps 16 --action-horizon 6 --frame-stride 1 \
  --output "${base}" --overwrite

prediction_args=()
for seed in 0 1 2; do
  prediction="${root}/ndf_seed${seed}_frames.npz"
  "${python_bin}" -m policy.GeometryFlow.predict_ndf_functional_frames \
    --dataset "${base}" --ndf-checkpoint "${ndf_checkpoint}" \
    --frame-checkpoint "${frame_root}/fold0_correct_frame_seed${seed}_checkpoint.pth" \
    --output "${prediction}" --batch-size 64 --device cuda:0 --overwrite
  prediction_args+=(--frame-predictions "${prediction}")
done

ensemble="${root}/ndf_medoid_fixedconf_frames.npz"
"${python_bin}" -m policy.GeometryFlow.ensemble_ndf_functional_frames \
  --dataset "${base}" "${prediction_args[@]}" --output "${ensemble}" \
  --fold 0 --aggregation medoid \
  --confidence-threshold-deg "${confidence_threshold_deg}" --overwrite

partial="${root}/task_flow_ndf_partial.npz"
"${python_bin}" -m policy.GeometryFlow.augment_task_flow_with_ndf_full_frame \
  --dataset "${base}" --frame-predictions "${ensemble}" \
  --output "${partial}" --fold 0 --overwrite

camera="${root}/task_flow_ndf_camera.npz"
"${python_bin}" -m policy.GeometryFlow.augment_task_flow_with_camera_ndf_frame \
  --dataset "${partial}" --data-dir "${data_dir}" --output "${camera}" \
  --fold 0 --calibration-metadata "${calibration}" --overwrite

full_archive="${root}/fulltask_history3_eefdelta6_stride${control_stride}.npz"
"${python_bin}" -m policy.GeometryFlow.prepare_fulltask_dense_eef_archive \
  --data-dir "${data_dir}" --episodes 1 --relation-archive "${camera}" \
  --output "${full_archive}" --history 3 --history-stride 1 \
  --action-horizon 6 --control-stride "${control_stride}" \
  --drop-unmatched-terminal --overwrite

correction="${root}/fulltask_history3_eefdelta6_stride${control_stride}_terminal_release.npz"
"${python_bin}" -m policy.GeometryFlow.select_mixed_correction_windows \
  --input "${full_archive}" --data-dir "${data_dir}" \
  --manifest "${sim_root}/correction_manifest.json" \
  --output "${correction}" --post-transition-frames 15 --overwrite

"${python_bin}" -m policy.GeometryFlow.merge_fulltask_dense_archives \
  --nominal "${artifact_root}/processed_datasets/fulltask_tagrt_v2/denseeef_left60_v1/fulltask_history3_eefdelta6_stride15.npz" \
  --correction "${correction}" \
  --output "${root}/fulltask_history3_eefdelta6_stride${control_stride}_terminal_release_merged.npz" \
  --overwrite
