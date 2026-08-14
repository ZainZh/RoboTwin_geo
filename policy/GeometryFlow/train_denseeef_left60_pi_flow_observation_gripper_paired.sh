#!/usr/bin/env bash
set -euo pipefail

artifact_root="${TAGRT_ARTIFACT_ROOT:-/shared2/sz/TAGRT-v1}"
python_bin="${PYTHON_BIN:-/root/miniconda3/envs/RoboTwin/bin/python}"
gpu_id="${GPU_ID:-0}"
seed="${SEED:-0}"
variant="${VARIANT:-observation_gripper}"
initial_variant="${INITIAL_VARIANT:-placement_episode0}"
initial_checkpoint_suffix="${INITIAL_CHECKPOINT_SUFFIX:-}"
dataset="${DATASET:-${artifact_root}/processed_datasets/fulltask_tagrt_v2/denseeef_left60_pi_postgrasp140_mixed_correction_v1/fulltask_history3_eefdelta6_stride15_placement_episode0_merged.npz}"
epochs="${EPOCHS:-30}"
patience="${PATIENCE:-8}"
learning_rate="${LEARNING_RATE:-3e-4}"
ema_decay="${EMA_DECAY:-0.995}"
validation_every="${VALIDATION_EVERY:-2}"
correction_weight="${CORRECTION_WEIGHT:-8}"
gripper_transition_weight="${GRIPPER_TRANSITION_WEIGHT:-6}"
correction_open_gripper_weight="${CORRECTION_OPEN_GRIPPER_WEIGHT:-1}"
initial_root="${artifact_root}/models/fulltask_tagrt_v2/denseeef_left60_pi_flow_${initial_variant}_seed${seed}"
model_dir="${artifact_root}/models/fulltask_tagrt_v2/denseeef_left60_pi_flow_${variant}_seed${seed}"
metrics_dir="${artifact_root}/experiment_outputs/fulltask_tagrt_v2/denseeef_left60_pi_flow_${variant}_seed${seed}"
episode_split_file="${EPISODE_SPLIT_FILE:-policy/GeometryFlow/episode_split_denseeef_left60_placement_episode0.json}"

export CUDA_VISIBLE_DEVICES="${gpu_id}"
export TAGRT_ARTIFACT_ROOT="${artifact_root}"
export PYTHONUNBUFFERED=1
mkdir -p "${model_dir}" "${metrics_dir}"

deterministic_args=()
if [[ "${DETERMINISTIC_GRIPPER_QUERY:-0}" == "1" ]]; then
  deterministic_args+=(--deterministic-gripper-query)
fi

train_one() {
  local condition="$1"
  "${python_bin}" -m policy.GeometryFlow.train_fulltask_tagrt_v2 \
    --dataset "${dataset}" --output-dir "${model_dir}" \
    --metrics-output-dir "${metrics_dir}" --conditions "${condition}" \
    --seed "${seed}" --epochs "${epochs}" --patience "${patience}" --batch-size 64 \
    --learning-rate "${learning_rate}" --weight-decay 1e-4 --lr-scheduler cosine \
    --lr-warmup-steps 25 --ema-decay "${ema_decay}" --validation-every "${validation_every}" \
    --feature-dim 192 --heads 6 --memory-layers 4 --decoder-layers 4 \
    --dropout 0.05 --flow-steps 10 --flow-time-alpha 1.5 \
    --gripper-weight 1 --gripper-transition-weight "${gripper_transition_weight}" \
    --correction-sample-weight "${correction_weight}" --gripper-selection-delay 0.15 \
    --correction-open-gripper-weight "${correction_open_gripper_weight}" \
    --observation-gripper-head \
    "${deterministic_args[@]}" \
    --initialize-from "${initial_root}/${condition}_seed${seed}${initial_checkpoint_suffix}.pt" \
    --reuse-initial-statistics --train-observation-gripper-head-only \
    --save-final-checkpoint \
    --episode-split-file "${episode_split_file}" \
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
