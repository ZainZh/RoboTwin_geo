#!/usr/bin/env bash
set -euo pipefail

python_bin="${PYTHON_BIN:-/root/miniconda3/envs/RoboTwin/bin/python}"
episode_count="${EPISODES:-10}"
run_name="${RUN_NAME:-densecontrol_pilot10}"
root="outputs/geometry_flow/fulltask_tagrt_v2/${run_name}"
data_dir="/shared2/sz/robotwin_data/data/place_shoe_geometry_marker/demo_clean_3d_object_pc_geometry_marker_densecontrol_pilot/data"
ndf_checkpoint="/shared2/sz/model/ndf/shoe.pth"
frame_root="outputs/geometry_flow/ndf_pretraining_ablation_local50_fold0_full_seeds012"
mkdir -p "${root}"

base="${root}/task_flow_base.npz"
"${python_bin}" -m policy.GeometryFlow.build_task_flow_dataset \
  --data-dir "${data_dir}" \
  --episodes "${episode_count}" \
  --observation-points 128 \
  --point-channels 6 \
  --target-color-priority \
  --cache-target-observation \
  --cache-operated-observation \
  --target-frame-token \
  --anchors 32 \
  --flow-steps 16 \
  --action-horizon 6 \
  --frame-stride 1 \
  --output "${base}" \
  --overwrite

prediction_args=()
for seed in 0 1 2; do
  prediction="${root}/ndf_seed${seed}_frames.npz"
  "${python_bin}" -m policy.GeometryFlow.predict_ndf_functional_frames \
    --dataset "${base}" \
    --ndf-checkpoint "${ndf_checkpoint}" \
    --frame-checkpoint "${frame_root}/fold0_correct_frame_seed${seed}_checkpoint.pth" \
    --output "${prediction}" \
    --batch-size 64 \
    --device cuda:0 \
    --overwrite
  prediction_args+=(--frame-predictions "${prediction}")
done

ensemble="${root}/ndf_medoid_p95_frames.npz"
"${python_bin}" -m policy.GeometryFlow.ensemble_ndf_functional_frames \
  --dataset "${base}" \
  "${prediction_args[@]}" \
  --output "${ensemble}" \
  --fold 0 \
  --confidence-percentile 95 \
  --aggregation medoid \
  --overwrite

partial="${root}/task_flow_ndf_partial.npz"
"${python_bin}" -m policy.GeometryFlow.augment_task_flow_with_ndf_full_frame \
  --dataset "${base}" \
  --frame-predictions "${ensemble}" \
  --output "${partial}" \
  --fold 0 \
  --overwrite

camera="${root}/task_flow_ndf_camera.npz"
"${python_bin}" -m policy.GeometryFlow.augment_task_flow_with_camera_ndf_frame \
  --dataset "${partial}" \
  --data-dir "${data_dir}" \
  --output "${camera}" \
  --fold 0 \
  --overwrite

"${python_bin}" -m policy.GeometryFlow.prepare_fulltask_dense_control_archive \
  --data-dir "${data_dir}" \
  --episodes "${episode_count}" \
  --relation-archive "${camera}" \
  --output "${root}/fulltask_history3_dense15.npz" \
  --history 3 \
  --history-stride 2 \
  --action-horizon 15 \
  --overwrite

echo "dense-control TAGRT archive: ${root}/fulltask_history3_dense15.npz"
