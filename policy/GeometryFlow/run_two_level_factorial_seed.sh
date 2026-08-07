#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: bash $0 SEED CONDITION" >&2
  exit 2
fi

seed="$1"
label="$2"
if [[ "${seed}" != "0" && "${seed}" != "1" && "${seed}" != "2" ]]; then
  echo "SEED must be 0, 1, or 2" >&2
  exit 2
fi

case "${label}" in
  point128)
    condition="point_tokens"
    feature_dim=128
    ;;
  pointcap148)
    condition="point_tokens"
    feature_dim=148
    ;;
  local_relation)
    condition="interaction_tokens"
    feature_dim=128
    ;;
  local_flow)
    condition="interaction_flow"
    feature_dim=128
    ;;
  *)
    echo "CONDITION must be point128, pointcap148, local_relation, or local_flow" >&2
    exit 2
    ;;
esac

python_bin="${PYTHON_BIN:-/root/miniconda3/envs/RoboTwin/bin/python}"
fold=1
frozen_root="outputs/geometry_flow/goal_policy_frame_crossfold_frozen_v1/fold1"
root="outputs/geometry_flow/two_level_factorial_v1/fold1"
nominal_dataset="${frozen_root}/task_flow_fold1_nominal_ndf_valtop2_p99_camera.npz"
recovery_dataset="${frozen_root}/task_flow_fold1_recovery_v4_ndf_valtop2_p99_camera.npz"
nominal_dir="${root}/${label}_nominal_seed${seed}"
joint_dir="${root}/${label}_joint_seed${seed}"
nominal_checkpoint="${nominal_dir}/fold1_${condition}_seed${seed}.pt"
joint_checkpoint="${joint_dir}/fold1_${condition}_seed${seed}.pt"

common_args=(
  --conditions "${condition}"
  --folds "${fold}"
  --seed "${seed}"
  --learning-rate 0.0003
  --weight-decay 0.00001
  --feature-dim "${feature_dim}"
  --layers 2
  --num-anchors 16
  --flow-steps 4
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
  --target-frame-mode normal
  --sample-mode all
  --policy-phase relation
  --device cuda:0
  --num-workers 0
  --episode-balanced-sampling
)

if [[ ! -f "${nominal_checkpoint}" ]]; then
  "${python_bin}" -m policy.GeometryFlow.train_interaction_flow_tokens \
    --dataset "${nominal_dataset}" \
    --output-dir "${nominal_dir}" \
    "${common_args[@]}" \
    --epochs 200 \
    --patience 30 \
    --batch-size 32
fi

if [[ ! -f "${joint_checkpoint}" ]]; then
  "${python_bin}" -m policy.GeometryFlow.train_interaction_flow_tokens \
    --dataset "${recovery_dataset}" \
    --output-dir "${joint_dir}" \
    "${common_args[@]}" \
    --epochs 160 \
    --patience 35 \
    --batch-size 128 \
    --initialize-from-checkpoint "${nominal_checkpoint}" \
    --recovery-target-fraction 0.25 \
    --minimum-recovery-train-target-confidence 0.5 \
    --recovery-validation-weight 0.25
fi

echo "completed two-level factorial training: seed=${seed} label=${label}"
