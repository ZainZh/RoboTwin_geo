#!/usr/bin/env bash
set -euo pipefail

artifact_root="${TAGRT_ARTIFACT_ROOT:-/shared2/sz/TAGRT-v1}"
python_bin="${PYTHON_BIN:-/root/miniconda3/envs/RoboTwin/bin/python}"
gpu_id="${GPU_ID:-0}"
seed="${SEED:-0}"
variant="${VARIANT:-terminal_release_geometry_adapter}"
distillation_weight="${DISTILLATION_WEIGHT:-10}"
correction_weight="${CORRECTION_WEIGHT:-256}"
gripper_transition_weight="${GRIPPER_TRANSITION_WEIGHT:-6}"
correction_open_gripper_weight="${CORRECTION_OPEN_GRIPPER_WEIGHT:-1}"
epochs="${EPOCHS:-20}"
patience="${PATIENCE:-8}"
validation_every="${VALIDATION_EVERY:-2}"
learning_rate="${LEARNING_RATE:-3e-4}"
ema_decay="${EMA_DECAY:-0.995}"
dataset="${DATASET:-${artifact_root}/processed_datasets/fulltask_tagrt_v2/denseeef_left60_pi_deterministic_terminal_seed173_fixedconf_v1/fulltask_history3_eefdelta6_stride15_terminal_release_merged.npz}"
initial_root="${artifact_root}/models/fulltask_tagrt_v2/denseeef_left60_pi_flow_deterministic_gripper_seed${seed}"
model_dir="${artifact_root}/models/fulltask_tagrt_v2/denseeef_left60_pi_flow_${variant}_seed${seed}"
metrics_dir="${artifact_root}/experiment_outputs/fulltask_tagrt_v2/denseeef_left60_pi_flow_${variant}_seed${seed}"

export CUDA_VISIBLE_DEVICES="${gpu_id}"
export TAGRT_ARTIFACT_ROOT="${artifact_root}"
export PYTHONUNBUFFERED=1
mkdir -p "${model_dir}" "${metrics_dir}"

correction_only_args=()
if [[ "${TRAIN_CORRECTIONS_ONLY:-0}" == "1" ]]; then
  correction_only_args+=(--train-corrections-only)
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
    --deterministic-gripper-query --gripper-geometry-adapter \
    --gripper-weight 1 --gripper-transition-weight "${gripper_transition_weight}" \
    --correction-sample-weight "${correction_weight}" --gripper-selection-delay 0.15 \
    --correction-open-gripper-weight "${correction_open_gripper_weight}" \
    --gripper-distillation-weight "${distillation_weight}" \
    --initialize-from "${initial_root}/${condition}_seed${seed}.pt" \
    --reuse-initial-statistics --train-gripper-geometry-adapter-only \
    "${correction_only_args[@]}" \
    --save-final-checkpoint \
    --episode-split-file policy/GeometryFlow/episode_split_denseeef_left60_placement_episode0.json \
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
