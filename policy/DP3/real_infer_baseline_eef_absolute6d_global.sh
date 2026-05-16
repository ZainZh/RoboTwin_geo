#!/bin/bash

task_name=${1:-grasp_mug}
task_config=${2:-demo_real_zed_sam2_objpc_global}
ckpt_setting="${task_config}-eef-absolute6d-global"
expert_data_num=${3:-32}
seed=${4:-0}
gpu_id=${5:-0}
checkpoint_num=${6:-3000}

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd "${script_dir}/../.." && pwd)

output_frame=${7:-source}
eef_frame_mode=${8:-reference_camera}
extra_flags=("${@:9}")
frame_overrides=(--output_frame "${output_frame}" --eef_frame_mode "${eef_frame_mode}")

export CUDA_VISIBLE_DEVICES=${gpu_id}
export HYDRA_FULL_ERROR=1
echo -e "\033[33mgpu id (to use): ${gpu_id}\033[0m"
echo -e "\033[33mreal zed output_frame: ${output_frame}\033[0m"
echo -e "\033[33mreal zed eef_frame_mode: ${eef_frame_mode}\033[0m"

cd "${repo_root}"

PYTHONWARNINGS=ignore::UserWarning \
python script/real_zed_inference/real_dp3_inference.py \
    --mode baseline \
    --action_mode eef_absolute6d \
    --config_name robot_dp3_eef_absolute6d \
    --task_name "${task_name}" \
    --task_config "${task_config}" \
    --ckpt_setting "${ckpt_setting}" \
    --expert_data_num "${expert_data_num}" \
    --seed "${seed}" \
    --gpu_id "${gpu_id}" \
    --checkpoint_num "${checkpoint_num}" \
    --profile_timing \
    --execute \
    --async_control \
    --async_control_hz 25 \
    --control_hz 5 \
    --servo_j_t 0.10 \
    --servo_j_gain 200 \
    --max_executed_joint_delta 0.015 \
    --max_executed_joint_delta_change 0.004 \
    --max_executed_gripper_delta 0.02 \
    --max_executed_gripper_delta_change 0.06 \
    --action_diagnostics_csv outputs/real_zed_inference/action_diag_async_baseline_eef_global.csv \
    "${frame_overrides[@]}" \
    "${extra_flags[@]}"
