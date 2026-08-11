#!/usr/bin/env bash
set -euo pipefail

artifact_root="${TAGRT_ARTIFACT_ROOT:-/shared2/sz/TAGRT-v1}"
python_bin="${PYTHON_BIN:-/root/miniconda3/envs/RoboTwin/bin/python}"
gpu_id="${GPU_ID:-5}"
data_root="${artifact_root}/simulator_data/place_shoe_geometry_marker/demo_clean_3d_object_pc_geometry_marker_densecontrol_gate2_train30"
audit_root="${artifact_root}/experiment_outputs/fulltask_tagrt_v2/gate2_train30/final_audit"
log_root="${artifact_root}/experiment_outputs/fulltask_tagrt_v2/gate2_train30"

mkdir -p "${audit_root}" "${log_root}"
export CUDA_VISIBLE_DEVICES="${gpu_id}"
export TAGRT_ARTIFACT_ROOT="${artifact_root}"
export PYTHONUNBUFFERED=1

"${python_bin}" -m policy.GeometryFlow.audit_densecontrol_batch \
  --data-root "${data_root}" \
  --output-dir "${audit_root}" \
  --expected-episodes 30 \
  --require-complete \
  --require-all-valid

"${python_bin}" script/eval_policy.py \
  --config policy/GeometryFlow/deploy_fulltask_action_replay_densecontrol_gate2_train30.yml \
  2>&1 | tee "${log_root}/dense_replay.log"
