#!/bin/bash

task_name=${1:-grasp_mug_new}
task_config=${2:-demo_real_zed_sam2_objpc_rightbase}
expert_data_num=${3:-32}
seed=${4:-0}
gpu_id=${5:-0}
semantic_ckpt_A=${6:-${SEMANTIC_CKPT_A:-${HOME}/DataModel/semantic/mug.pt}}
semantic_ckpt_B=${7:-none}
semantic_device=${8:-cuda:0}
object_placeholders=${9:-\{A\},\{B\}}
checkpoint_num=${10:-3000}
semantic_point_num=${11:-128}

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd "${script_dir}/../.." && pwd)
source "${script_dir}/real_infer_arg_utils.sh"

if [[ "${12:-}" == --* ]]; then
    output_frame_arg=auto
    robot_camera_calibration_path_arg=auto
    extra_flags=("${@:12}")
else
    output_frame_arg=${12:-auto}
    robot_camera_calibration_path_arg=${13:-auto}
    extra_flags=("${@:14}")
fi

output_frame=$(resolve_real_zed_output_frame "${repo_root}" "${task_name}" "${task_config}" "${output_frame_arg}")
robot_camera_calibration_path=$(
    resolve_real_zed_robot_camera_calibration_path \
        "${repo_root}" \
        "${output_frame}" \
        "${robot_camera_calibration_path_arg}"
)

frame_overrides=(--output_frame "${output_frame}")
if [ -n "${robot_camera_calibration_path}" ]; then
    frame_overrides+=(--robot_camera_calibration_path "${robot_camera_calibration_path}")
fi

export CUDA_VISIBLE_DEVICES=${gpu_id}
export HYDRA_FULL_ERROR=1
echo -e "\033[33mgpu id (to use): ${gpu_id}\033[0m"
echo -e "\033[33mreal zed output_frame: ${output_frame}\033[0m"
if [ -n "${robot_camera_calibration_path}" ]; then
    echo -e "\033[33mrobot-camera calibration: ${robot_camera_calibration_path}\033[0m"
fi

cd "${repo_root}"

PYTHONWARNINGS=ignore::UserWarning \
python script/real_zed_inference/real_dp3_inference.py \
    --mode semantic_pointwise_hybrid \
    --task_name "${task_name}" \
    --task_config "${task_config}" \
    --ckpt_setting "${task_config}-objpc-semantic-pointwise-hybrid" \
    --expert_data_num "${expert_data_num}" \
    --seed "${seed}" \
    --gpu_id "${gpu_id}" \
    --checkpoint_num "${checkpoint_num}" \
    --semantic_ckpt_A "${semantic_ckpt_A}" \
    --semantic_ckpt_B "${semantic_ckpt_B}" \
    --semantic_device "${semantic_device}" \
    --semantic_point_num "${semantic_point_num}" \
    --object_placeholders "${object_placeholders}" \
    --enable_sam2_objpc \
    --profile_timing \
    --execute \
    --async_control \
    --async_control_hz 20 \
    --control_hz 5 \
    --servo_j_t 0.10 \
    --servo_j_gain 200 \
    --max_executed_joint_delta 0.015 \
    --max_executed_joint_delta_change 0.004 \
    --max_executed_gripper_delta 0.02 \
    --max_executed_gripper_delta_change 0.06 \
    --action_diagnostics_csv outputs/real_zed_inference/action_diag_async.csv\
    "${frame_overrides[@]}" \
    "${extra_flags[@]}"
