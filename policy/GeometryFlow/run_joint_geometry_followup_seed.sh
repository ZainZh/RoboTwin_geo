#!/usr/bin/env bash
set -euo pipefail

# Post-hoc architecture diagnostic locked in
# JOINT_GEOMETRY_FOLLOWUP_PROTOCOL_20260806.md.

if [[ $# -ne 3 ]]; then
  echo "usage: bash $0 FOLD SEED MODE" >&2
  exit 2
fi

fold="$1"
seed="$2"
mode="$3"
if [[ "${fold}" != "1" && "${fold}" != "2" ]]; then
  echo "FOLD must be 1 or 2" >&2
  exit 2
fi
if [[ "${seed}" != "0" && "${seed}" != "1" && "${seed}" != "2" ]]; then
  echo "SEED must be 0, 1, or 2" >&2
  exit 2
fi
if [[ "${mode}" != "clean" && "${mode}" != "zero" ]]; then
  echo "MODE must be clean or zero" >&2
  exit 2
fi

python_bin="${PYTHON_BIN:-/root/miniconda3/envs/RoboTwin/bin/python}"
frozen_root="outputs/geometry_flow/goal_policy_frame_crossfold_frozen_v1/fold${fold}"
followup_root="outputs/geometry_flow/goal_policy_frame_joint_followup_v1/fold${fold}"
nominal_dataset="${frozen_root}/task_flow_fold${fold}_nominal_ndf_valtop2_p99_camera.npz"
recovery_dataset="${frozen_root}/task_flow_fold${fold}_recovery_v4_ndf_valtop2_p99_camera.npz"

common_args=(
  --conditions interaction_functional_direct_flow
  --folds "${fold}"
  --seed "${seed}"
  --learning-rate 0.0003
  --weight-decay 0.00001
  --feature-dim 128
  --layers 2
  --num-anchors 16
  --flow-steps 4
  --diffusion-steps 100
  --diffusion-inference-steps 10
  --diffusion-prediction-type sample
  --geometry-loss-weight 0.1
  --rigidity-loss-weight 0.02
  --rotation-loss-weight 0
  --flow-loss-mode unordered
  --motion-loss-scale 10
  --flow-parameterization free
  --target-horizon-mode terminal
  --geometry-phase all
  --active-arm-loss-weight 1
  --relation-loss-weight 1
  --rotation-action-loss-weight 1
  --endpoint-rotation-action-loss-weight 0
  --action-frame world
  --policy-frame world
  --functional-frame-source dataset_relative
  --sample-mode all
  --policy-phase relation
  --device cuda:0
  --num-workers 0
  --episode-balanced-sampling
)

if [[ "${mode}" == "zero" ]]; then
  nominal_dir="${followup_root}/policy_zero_nominal_seed${seed}"
  "${python_bin}" -m policy.GeometryFlow.train_interaction_flow_tokens \
    --dataset "${nominal_dataset}" \
    --output-dir "${nominal_dir}" \
    "${common_args[@]}" \
    --epochs 200 \
    --patience 30 \
    --batch-size 32 \
    --target-frame-mode zero
  initial_checkpoint="${nominal_dir}/fold${fold}_interaction_functional_direct_flow_seed${seed}.pt"
else
  initial_checkpoint="${frozen_root}/policy_nominal_seed${seed}/fold${fold}_interaction_functional_direct_flow_seed${seed}.pt"
fi

joint_dir="${followup_root}/policy_joint_${mode}_seed${seed}"
"${python_bin}" -m policy.GeometryFlow.train_interaction_flow_tokens \
  --dataset "${recovery_dataset}" \
  --output-dir "${joint_dir}" \
  "${common_args[@]}" \
  --epochs 160 \
  --patience 35 \
  --batch-size 128 \
  --target-frame-mode "${mode/clean/normal}" \
  --initialize-from-checkpoint "${initial_checkpoint}" \
  --recovery-target-fraction 0.25 \
  --minimum-recovery-train-target-confidence 0.5 \
  --recovery-validation-weight 0.25

echo "completed joint geometry follow-up: fold=${fold} seed=${seed} mode=${mode}"
