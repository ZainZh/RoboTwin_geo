#!/usr/bin/env bash
set -euo pipefail

# Capacity-matched Raw/TAGRT flow-motion adaptation on admitted on-policy
# expert continuations. Observation encoders and gripper heads stay frozen.
artifact_root="${TAGRT_ARTIFACT_ROOT:-/shared2/sz/TAGRT-v1}"
python_bin="${PYTHON_BIN:-/root/miniconda3/envs/RoboTwin/bin/python}"
gpu_id="${GPU_ID:-0}"
seed="${SEED:-0}"
variant="${VARIANT:-motion_adapter}"
initial_variant="${INITIAL_VARIANT:-deterministic_gripper}"
initial_checkpoint_suffix="${INITIAL_CHECKPOINT_SUFFIX:-}"
dataset="${DATASET:?set DATASET to a paired nominal/correction archive}"
episode_split_file="${EPISODE_SPLIT_FILE:?set EPISODE_SPLIT_FILE}"
epochs="${EPOCHS:-20}"
patience="${PATIENCE:-6}"
learning_rate="${LEARNING_RATE:-1e-5}"
correction_weight="${CORRECTION_WEIGHT:-8}"
ema_decay="${EMA_DECAY:-0.995}"
validation_every="${VALIDATION_EVERY:-2}"
initial_root="${artifact_root}/models/fulltask_tagrt_v2/denseeef_left60_pi_flow_${initial_variant}_seed${seed}"
model_dir="${artifact_root}/models/fulltask_tagrt_v2/denseeef_left60_pi_flow_${variant}_seed${seed}"
metrics_dir="${artifact_root}/experiment_outputs/fulltask_tagrt_v2/denseeef_left60_pi_flow_${variant}_seed${seed}"

export CUDA_VISIBLE_DEVICES="${gpu_id}"
export TAGRT_ARTIFACT_ROOT="${artifact_root}"
export PYTHONUNBUFFERED=1
mkdir -p "${model_dir}" "${metrics_dir}"

train_one() {
  local condition="$1"
  "${python_bin}" -m policy.GeometryFlow.train_fulltask_tagrt_v2 \
    --dataset "${dataset}" --output-dir "${model_dir}" \
    --metrics-output-dir "${metrics_dir}" --conditions "${condition}" \
    --seed "${seed}" --epochs "${epochs}" --patience "${patience}" --batch-size 64 \
    --learning-rate "${learning_rate}" --weight-decay 1e-4 --lr-scheduler cosine \
    --lr-warmup-steps 25 --ema-decay "${ema_decay}" --validation-every "${validation_every}" \
    --validation-selection-metric first_action_normalized_motion_mse \
    --feature-dim 192 --heads 6 --memory-layers 4 --decoder-layers 4 \
    --dropout 0.05 --flow-steps 10 --flow-time-alpha 1.5 \
    --deterministic-gripper-query --gripper-weight 0 \
    --correction-sample-weight "${correction_weight}" \
    --initialize-from "${initial_root}/${condition}_seed${seed}${initial_checkpoint_suffix}.pt" \
    --reuse-initial-statistics --train-flow-motion-decoder-only \
    --save-final-checkpoint --episode-split-file "${episode_split_file}" \
    --device cuda:0 >"${metrics_dir}/${condition}.train.log" 2>&1
}

train_one raw_pi_flow &
raw_pid=$!
train_one tagrt_pi_flow &
tagrt_pid=$!
status=0
wait "${raw_pid}" || status=$?
wait "${tagrt_pid}" || status=$?
exit "${status}"
