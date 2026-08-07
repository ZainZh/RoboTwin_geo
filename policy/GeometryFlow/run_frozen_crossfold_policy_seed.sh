#!/usr/bin/env bash
set -euo pipefail

# Train one matched nominal/recovery policy pair from the frozen camera-only
# datasets.  Run multiple seeds concurrently on one selected GPU if desired.

if [[ $# -ne 2 ]]; then
  echo "usage: bash $0 FOLD SEED" >&2
  exit 2
fi

fold="$1"
seed="$2"
if [[ "${fold}" != "1" && "${fold}" != "2" ]]; then
  echo "FOLD must be 1 or 2" >&2
  exit 2
fi
if [[ "${seed}" != "0" && "${seed}" != "1" && "${seed}" != "2" ]]; then
  echo "SEED must be 0, 1, or 2" >&2
  exit 2
fi

python_bin="${PYTHON_BIN:-/root/miniconda3/envs/RoboTwin/bin/python}"
root="outputs/geometry_flow/goal_policy_frame_crossfold_frozen_v1/fold${fold}"
nominal_dataset="${root}/task_flow_fold${fold}_nominal_ndf_valtop2_p99_camera.npz"
recovery_dataset="${root}/task_flow_fold${fold}_recovery_v4_ndf_valtop2_p99_camera.npz"
nominal_dir="${root}/policy_nominal_seed${seed}"
recovery_dir="${root}/policy_recovery_seed${seed}"

"${python_bin}" -m policy.GeometryFlow.train_interaction_flow_tokens \
  --dataset "${nominal_dataset}" \
  --output-dir "${nominal_dir}" \
  --conditions interaction_functional_direct_flow \
  --folds "${fold}" \
  --seed "${seed}" \
  --epochs 200 \
  --patience 30 \
  --batch-size 32 \
  --learning-rate 0.0003 \
  --weight-decay 0.00001 \
  --feature-dim 128 \
  --layers 2 \
  --num-anchors 16 \
  --flow-steps 4 \
  --diffusion-steps 100 \
  --diffusion-inference-steps 10 \
  --diffusion-prediction-type sample \
  --geometry-loss-weight 0.1 \
  --rigidity-loss-weight 0.02 \
  --rotation-loss-weight 0 \
  --flow-loss-mode unordered \
  --motion-loss-scale 10 \
  --flow-parameterization free \
  --target-horizon-mode terminal \
  --geometry-phase all \
  --active-arm-loss-weight 1 \
  --relation-loss-weight 1 \
  --rotation-action-loss-weight 1 \
  --endpoint-rotation-action-loss-weight 0 \
  --action-frame world \
  --policy-frame world \
  --functional-frame-source dataset_relative \
  --target-frame-mode normal \
  --sample-mode all \
  --policy-phase relation \
  --device cuda:0 \
  --num-workers 0 \
  --episode-balanced-sampling

nominal_checkpoint="${nominal_dir}/fold${fold}_interaction_functional_direct_flow_seed${seed}.pt"
"${python_bin}" -m policy.GeometryFlow.train_interaction_flow_tokens \
  --dataset "${recovery_dataset}" \
  --output-dir "${recovery_dir}" \
  --conditions interaction_functional_correction_adapter_flow \
  --folds "${fold}" \
  --seed "${seed}" \
  --epochs 160 \
  --patience 35 \
  --batch-size 128 \
  --learning-rate 0.0003 \
  --weight-decay 0.00001 \
  --feature-dim 128 \
  --layers 2 \
  --num-anchors 16 \
  --flow-steps 4 \
  --diffusion-steps 100 \
  --diffusion-inference-steps 10 \
  --diffusion-prediction-type sample \
  --geometry-loss-weight 0.1 \
  --rigidity-loss-weight 0.02 \
  --rotation-loss-weight 0 \
  --flow-loss-mode unordered \
  --motion-loss-scale 10 \
  --flow-parameterization free \
  --target-horizon-mode terminal \
  --geometry-phase all \
  --active-arm-loss-weight 1 \
  --relation-loss-weight 1 \
  --rotation-action-loss-weight 1 \
  --endpoint-rotation-action-loss-weight 0 \
  --action-frame world \
  --policy-frame world \
  --functional-frame-source dataset_relative \
  --target-frame-mode normal \
  --sample-mode all \
  --policy-phase relation \
  --initialize-from-checkpoint "${nominal_checkpoint}" \
  --initialize-recovery-adapter-from-decoder \
  --train-recovery-action-adapter-only \
  --device cuda:0 \
  --num-workers 2 \
  --episode-balanced-sampling \
  --recovery-target-fraction 0.25 \
  --minimum-recovery-train-target-confidence 0.5 \
  --recovery-validation-weight 0.25 \
  --maximum-nominal-validation-relative-degradation 0.02 \
  --nominal-adapter-distillation-weight 5

echo "completed frozen matched policy pair: fold=${fold} seed=${seed}"
