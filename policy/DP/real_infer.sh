#!/bin/bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

task_name=${1:-grasp_mug}
task_config=${2:-demo_real_zed_sam2_objpc}
expert_data_num=${3:-32}
seed=${4:-0}
gpu_id=${5:-0}
camera_labels=${6:-left}
checkpoint_num=${7:-600}
shift $(( $# < 7 ? $# : 7 ))

camera_setting=${camera_labels//,/_}
ckpt_setting="${task_config}-dp-${camera_setting}"

if [[ "${camera_labels}" == *","* ]]; then
    dp_camera_map="head_cam:global,left_cam:left,right_cam:right"
else
    dp_camera_map="head_cam:${camera_labels}"
fi

export CUDA_VISIBLE_DEVICES=${gpu_id}
export HYDRA_FULL_ERROR=1
echo -e "\033[33mgpu id (to use): ${gpu_id}\033[0m"
echo -e "\033[33mReal-ZED DP inference cameras: ${camera_labels}\033[0m"

cd "${repo_root}"

python script/real_zed_inference/real_dp_inference.py \
    --task_name "${task_name}" \
    --task_config "${task_config}" \
    --ckpt_setting "${ckpt_setting}" \
    --expert_data_num "${expert_data_num}" \
    --seed "${seed}" \
    --checkpoint_num "${checkpoint_num}" \
    --gpu_id "${gpu_id}" \
    --camera_labels "${camera_labels}" \
    --dp_camera_map "${dp_camera_map}" \
    --profile_timing \
    --execute \
    --async_control \
    --async_control_hz 25 \
    --control_hz 5 \
    --servo_j_t 0.10 \
    --servo_j_gain 200 \
    --max_executed_joint_delta 0.012 \
    --max_executed_joint_delta_change 0.003 \
    --max_executed_gripper_delta 0.02 \
    --max_executed_gripper_delta_change 0.06 \
    --action_diagnostics_csv outputs/real_zed_inference/action_diag_async.csv\
    "$@"
