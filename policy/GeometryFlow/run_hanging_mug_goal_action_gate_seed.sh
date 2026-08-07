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
root="outputs/geometry_flow/hanging_mug_cross_task_v1"
dataset="${root}/dev24_task_flow_camera_rotation_temporal.npz"

common_args=(
  --dataset "${dataset}"
  --folds 0
  --seed "${seed}"
  --epochs 200
  --patience 35
  --batch-size 32
  --learning-rate 0.0003
  --weight-decay 0.00001
  --feature-dim 128
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
  --policy-frame goal_action
  --functional-frame-source dataset_relative
  --sample-mode all
  --policy-phase relation
  --device cuda:0
  --num-workers 0
  --episode-balanced-sampling
)

train_variant() {
  local name="$1"
  local frame_mode="$2"
  local output_dir="${root}/camera_rotation_temporal_goal_action_${name}_seed${seed}"
  local result="${output_dir}/fold0_interaction_functional_direct_flow_seed${seed}.json"
  if [[ -f "${result}" ]]; then
    echo "skip completed ${result}"
    return
  fi
  "${python_bin}" -m policy.GeometryFlow.train_interaction_flow_tokens \
    "${common_args[@]}" \
    --output-dir "${output_dir}" \
    --conditions interaction_functional_direct_flow \
    --target-frame-mode "${frame_mode}"
}

train_variant zero zero
train_variant predicted normal
