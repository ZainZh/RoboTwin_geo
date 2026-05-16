#!/bin/bash

task_name=${1}
task_config=${2}
expert_data_num=${3}
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd "${script_dir}/../.." && pwd)
eef_calibration_path=${4:-${repo_root}/script/real_zed_collection/calibration/three_camera_workspace_extrinsics.yaml}
left_robot_camera_calibration_path=${5:-${repo_root}/script/real_zed_collection/calibration/robot_camera_apriltag_left_global.yaml}
right_robot_camera_calibration_path=${6:-${repo_root}/script/real_zed_collection/calibration/robot_camera_apriltag_right_global.yaml}
eef_frame_mode=${7:-right_base}

python scripts/process_data_eef_absolute6d.py \
    "${task_name}" \
    "${task_config}" \
    "${expert_data_num}" \
    --eef_calibration_path "${eef_calibration_path}" \
    --eef_frame_mode "${eef_frame_mode}" \
    --left_robot_camera_calibration_path "${left_robot_camera_calibration_path}" \
    --right_robot_camera_calibration_path "${right_robot_camera_calibration_path}"
