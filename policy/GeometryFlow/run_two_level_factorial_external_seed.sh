#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: bash $0 SEED" >&2
  exit 2
fi

seed="$1"
if [[ "${seed}" != "0" && "${seed}" != "1" && "${seed}" != "2" ]]; then
  echo "SEED must be 0, 1, or 2" >&2
  exit 2
fi

python_bin="${PYTHON_BIN:-/root/miniconda3/envs/RoboTwin/bin/python}"
factorial_root="outputs/geometry_flow/two_level_factorial_v1/fold1"
followup_root="outputs/geometry_flow/goal_policy_frame_joint_followup_v1/fold1"
frozen_root="outputs/geometry_flow/goal_policy_frame_crossfold_frozen_v1/fold1"
manifest="outputs/geometry_flow/external_shoe_blind_v2/blind_policy_eval/pose_correction_snapshots_32.json"
output="${factorial_root}/external_factorial_seed${seed}_32.json"
resume_args=()
if [[ -f "${output}" ]]; then
  resume_args+=(--resume-existing)
fi

"${python_bin}" script/evaluate_pose_correction_closedloop.py \
  --manifest "${manifest}" \
  --checkpoint point128 "${factorial_root}/point128_joint_seed${seed}/fold1_point_tokens_seed${seed}.pt" \
  --checkpoint pointcap148 "${factorial_root}/pointcap148_joint_seed${seed}/fold1_point_tokens_seed${seed}.pt" \
  --checkpoint local_relation "${factorial_root}/local_relation_joint_seed${seed}/fold1_interaction_tokens_seed${seed}.pt" \
  --checkpoint local_flow "${factorial_root}/local_flow_joint_seed${seed}/fold1_interaction_flow_seed${seed}.pt" \
  --checkpoint local_flow_zero_global "${followup_root}/policy_joint_zero_seed${seed}/fold1_interaction_functional_direct_flow_seed${seed}.pt" \
  --checkpoint full_two_level "${followup_root}/policy_joint_clean_seed${seed}/fold1_interaction_functional_direct_flow_seed${seed}.pt" \
  --dense-frame-checkpoint "outputs/geometry_flow/goal_policy_frame_crossfold_frozen_v1/ndf_fold1/fold1_correct_frame_seed0_checkpoint.pth" \
  --dense-frame-checkpoint "outputs/geometry_flow/goal_policy_frame_crossfold_frozen_v1/ndf_fold1/fold1_correct_frame_seed1_checkpoint.pth" \
  --camera-metadata "${frozen_root}/task_flow_fold1_recovery_v4_ndf_valtop2_p99_camera.json" \
  --ensemble-metadata "${frozen_root}/fold1_recovery_v4_ndf_valtop2_p99.json" \
  --output "${output}" \
  "${resume_args[@]}" \
  --device cuda:0 \
  --variant-dense-frame-mode local_flow_zero_global zero \
  --levels 0 \
  --perturbation-seed 20260807 \
  --policy-calls 6 \
  --execute-steps 1 \
  --target-latch-frames 3 \
  --success-translation-cm 4 \
  --success-rotation-deg 15 \
  --paired-snapshot \
  --snapshot-settle-steps 5

echo "completed external two-level factorial evaluation: seed=${seed}"
