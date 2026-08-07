#!/usr/bin/env bash
set -euo pipefail

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
frozen_root="outputs/geometry_flow/goal_policy_frame_crossfold_frozen_v1/fold${fold}"
followup_root="outputs/geometry_flow/goal_policy_frame_joint_followup_v1/fold${fold}"
selection_json="${frozen_root}/fold${fold}_correct_frame_valtop2_p99.json"
manifest="${frozen_root}/pose_correction_snapshots_valid24.json"
output="${followup_root}/closedloop_joint_clean_vs_trained_zero_balanced6.json"

if [[ "${fold}" == "1" ]]; then
  episodes="0,1,2,3,4,7"
else
  episodes="0,1,2,3,6,11"
fi

mapfile -t selected_seeds < <(
  sed -n '/"selected_seeds"/,/]/p' "${selection_json}" \
    | grep -E '^[[:space:]]*[0-9]+,?[[:space:]]*$' \
    | tr -d ' ,'
)
if [[ ${#selected_seeds[@]} -ne 2 ]]; then
  echo "could not read exactly two selected NDF seeds" >&2
  exit 1
fi

dense_args=()
for ndf_seed in "${selected_seeds[@]}"; do
  dense_args+=(
    --dense-frame-checkpoint
    "outputs/geometry_flow/goal_policy_frame_crossfold_frozen_v1/ndf_fold${fold}/fold${fold}_correct_frame_seed${ndf_seed}_checkpoint.pth"
  )
done

checkpoint_args=()
variant_args=()
for seed in 0 1 2; do
  if [[ "${fold}" == "2" ]]; then
    clean="${frozen_root}/policy_joint_diagnostic_seed${seed}/fold2_interaction_functional_direct_flow_seed${seed}.pt"
  else
    clean="${followup_root}/policy_joint_clean_seed${seed}/fold1_interaction_functional_direct_flow_seed${seed}.pt"
  fi
  zero="${followup_root}/policy_joint_zero_seed${seed}/fold${fold}_interaction_functional_direct_flow_seed${seed}.pt"
  checkpoint_args+=(--checkpoint "joint_clean_seed${seed}" "${clean}")
  checkpoint_args+=(--checkpoint "joint_zero_seed${seed}" "${zero}")
  variant_args+=(--variant-dense-frame-mode "joint_zero_seed${seed}" zero)
done

"${python_bin}" script/evaluate_pose_correction_closedloop.py \
  --manifest "${manifest}" \
  "${checkpoint_args[@]}" \
  "${dense_args[@]}" \
  --camera-metadata "${frozen_root}/task_flow_fold${fold}_recovery_v4_ndf_valtop2_p99_camera.json" \
  --ensemble-metadata "${frozen_root}/fold${fold}_recovery_v4_ndf_valtop2_p99.json" \
  --output "${output}" \
  --device cuda:0 \
  "${variant_args[@]}" \
  --episodes "${episodes}" \
  --levels 0 \
  --perturbation-seed 20260807 \
  --policy-calls 6 \
  --execute-steps 1 \
  --target-latch-frames 3 \
  --success-translation-cm 4 \
  --success-rotation-deg 15 \
  --paired-snapshot \
  --snapshot-settle-steps 5

echo "completed joint geometry follow-up closed loop: fold=${fold} output=${output}"
