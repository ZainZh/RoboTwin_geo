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
source_dataset="${root}/dev24_task_flow.npz"
data_dir="${root}/dev24_nominal/data"
frame_variant="${HANGING_MUG_FRAME_VARIANT:-pretrained}"
case "${frame_variant}" in
  pretrained)
    frame_predictions="${root}/ndf_frame_pretrained_fold0/fold0_correct_frame_valtop2_p95.npz"
    dataset="${root}/dev24_task_flow_ndf_current_oracle_goal.npz"
    output_prefix="ndf_partial"
    augment_mode="partial"
    ;;
  random)
    frame_predictions="${root}/ndf_frame_random_fold0/fold0_correct_frame_valtop2_p95.npz"
    dataset="${root}/dev24_task_flow_random_equivariant_current_oracle_goal.npz"
    output_prefix="ndf_random_partial"
    augment_mode="partial"
    ;;
  camera_rotation)
    frame_predictions="${root}/ndf_frame_random_fold0/fold0_correct_frame_valtop2_p95.npz"
    dataset="${root}/dev24_task_flow_camera_rotation.npz"
    output_prefix="camera_rotation"
    augment_mode="camera_rotation"
    ;;
  camera_rotation_temporal)
    frame_predictions="${root}/ndf_frame_random_fold0/fold0_correct_frame_valtop2_p95.npz"
    dataset="${root}/dev24_task_flow_camera_rotation_temporal.npz"
    output_prefix="camera_rotation_temporal"
    augment_mode="camera_rotation_temporal"
    ;;
  *)
    echo "HANGING_MUG_FRAME_VARIANT must be pretrained, random, camera_rotation, or camera_rotation_temporal" >&2
    exit 2
    ;;
esac

if [[ ! -f "${frame_predictions}" ]]; then
  echo "selected NDF frame predictions are missing: ${frame_predictions}" >&2
  exit 1
fi
if [[ ! -f "${dataset}" ]]; then
  if [[ "${augment_mode}" == camera_rotation* ]]; then
    temporal_args=()
    if [[ "${augment_mode}" == "camera_rotation_temporal" ]]; then
      temporal_args+=(--temporal-symmetry-stabilization)
    fi
    "${python_bin}" -m policy.GeometryFlow.augment_hanging_mug_with_camera_rotation \
      --dataset "${source_dataset}" \
      --data-dir "${data_dir}" \
      --current-frame-predictions "${frame_predictions}" \
      --output "${dataset}" \
      --reference-episode 15 \
      --train-ids 2 3 4 7 \
      --validation-ids 1 6 \
      --test-ids 0 5 \
      --registration-points 512 \
      "${temporal_args[@]}"
  else
    "${python_bin}" -m policy.GeometryFlow.augment_task_flow_with_ndf_full_frame \
      --dataset "${source_dataset}" \
      --frame-predictions "${frame_predictions}" \
      --output "${dataset}" \
      --fold 0
  fi
fi

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
  local output_dir="${root}/${output_prefix}_${label}_seed${seed}"
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
train_condition shuffled interaction_functional_direct_flow shuffled
train_condition predicted interaction_functional_direct_flow normal

echo "completed hanging-mug ${frame_variant} partial-perception gate: seed=${seed}"
