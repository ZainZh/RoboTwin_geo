#!/usr/bin/env bash
set -euo pipefail
if [[ $# -ne 1 || ( "$1" != "tagrt" && "$1" != "raw" ) ]]; then
  echo "usage: $0 {tagrt|raw}" >&2; exit 2
fi
condition="$1"
artifact_root="${TAGRT_ARTIFACT_ROOT:-/shared/sz/TAGRT-v1}"
python_bin="${PYTHON_BIN:-/root/miniconda3/envs/RoboTwin/bin/python}"
gpu_id="${GPU_ID:-0}"
variant="${MODEL_VARIANT:-placement_episode0}"
checkpoint_suffix="${CHECKPOINT_SUFFIX:-}"
max_policy_steps="${MAX_POLICY_STEPS:-220}"
evaluation_tag="${EVALUATION_TAG:-}"
evaluation_seed_file="${EVALUATION_SEED_FILE:-policy/GeometryFlow/eval_seeds_gate2_left60_train6.json}"
evaluation_start_episode_index="${EVALUATION_START_EPISODE_INDEX:-49}"
if [[ -n "${evaluation_tag}" ]]; then
  evaluation_tag="-${evaluation_tag//_/-}"
fi
config="policy/GeometryFlow/deploy_fulltask_${condition}_pi_flow_placement_episode0_video.yml"
checkpoint="${artifact_root}/models/fulltask_tagrt_v2/denseeef_left60_pi_flow_${variant}_seed0/${condition}_pi_flow_seed0${checkpoint_suffix}.pt"
log_root="${artifact_root}/experiment_outputs/fulltask_tagrt_v2/denseeef_left60_pi_flow_${variant}_online"
[[ -s "${checkpoint}" ]] || { echo "missing checkpoint: ${checkpoint}" >&2; exit 1; }
mkdir -p "${log_root}"
export CUDA_VISIBLE_DEVICES="${gpu_id}"
export TAGRT_ARTIFACT_ROOT="${artifact_root}"
export PYTHONUNBUFFERED=1
"${python_bin}" script/eval_policy.py --config "${config}" --overrides \
  --fulltask_tagrt_checkpoint "${checkpoint}" \
  --fulltask_tagrt_max_policy_steps "${max_policy_steps}" \
  --evaluation_seed_file "${evaluation_seed_file}" \
  --evaluation_start_episode_index "${evaluation_start_episode_index}" \
  --ckpt_setting "fulltask-${condition}-pi-flow-${variant//_/-}-seed0${checkpoint_suffix//_/-}${evaluation_tag}-video" \
  2>&1 | tee "${log_root}/${condition}.log"
