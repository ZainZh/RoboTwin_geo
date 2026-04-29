#!/bin/bash

task_name=${1:-grasp_mug}
task_config=${2:-demo_real_zed_sam2_objpc}
expert_data_num=${3:-57}
seed=${4:-0}
gpu_id=${5:-0}
checkpoint_num=${6:-3000}

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd "${script_dir}/../.." && pwd)
source "${script_dir}/real_infer_arg_utils.sh"

if [[ "${7:-}" == --* ]]; then
    output_frame_arg=auto
    robot_camera_calibration_path_arg=auto
    extra_flags=("${@:7}")
else
    output_frame_arg=${7:-auto}
    robot_camera_calibration_path_arg=${8:-auto}
    extra_flags=("${@:9}")
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
    --mode baseline \
    --task_name "${task_name}" \
    --task_config "${task_config}" \
    --ckpt_setting "${task_config}" \
    --expert_data_num "${expert_data_num}" \
    --seed "${seed}" \
    --gpu_id "${gpu_id}" \
    --checkpoint_num "${checkpoint_num}" \
    --profile_timing \
    --execute \
    "${frame_overrides[@]}" \
    "${extra_flags[@]}"
