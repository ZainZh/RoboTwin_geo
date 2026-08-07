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
data_dir="${root}/dev24_nominal/data"
dataset="${root}/dev24_task_flow.npz"
oracle_dataset="${root}/dev24_task_flow_oracle.npz"

if [[ ! -f "${data_dir}/episode23.hdf5" ]]; then
  echo "development collection is incomplete: ${data_dir}/episode23.hdf5 missing" >&2
  exit 1
fi

if [[ ! -f "${dataset}" ]]; then
  "${python_bin}" -m policy.GeometryFlow.build_task_flow_dataset \
    --data-dir "${data_dir}" \
    --episodes 24 \
    --observation-points 128 \
    --point-channels 3 \
    --cache-target-observation \
    --cache-operated-observation \
    --anchors 16 \
    --flow-steps 4 \
    --action-horizon 6 \
    --frame-stride 1 \
    --collapse-static-eef-frames \
    --output "${dataset}"
fi

if [[ ! -f "${oracle_dataset}" ]]; then
  "${python_bin}" -m policy.GeometryFlow.augment_task_flow_with_oracle_frame \
    --dataset "${dataset}" \
    --output "${oracle_dataset}"
fi

common_args=(
  --dataset "${oracle_dataset}"
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
  --policy-frame world
  --functional-frame-source dataset_relative
  --sample-mode all
  --policy-phase relation
  --device cuda:0
  --num-workers 0
  --episode-balanced-sampling
)

train_condition() {
  local label="$1"
  local condition="$2"
  local frame_mode="$3"
  local output_dir="${root}/oracle_gate_${label}_seed${seed}"
  local result="${output_dir}/fold0_${condition}_seed${seed}.json"
  if [[ -f "${result}" ]]; then
    echo "skip completed ${result}"
    return
  fi
  "${python_bin}" -m policy.GeometryFlow.train_interaction_flow_tokens \
    "${common_args[@]}" \
    --output-dir "${output_dir}" \
    --conditions "${condition}" \
    --target-frame-mode "${frame_mode}"
}

train_condition raw point_tokens normal
train_condition zero_global interaction_functional_direct_flow zero
train_condition full_oracle interaction_functional_direct_flow normal

echo "completed hanging-mug oracle gate: seed=${seed}"
