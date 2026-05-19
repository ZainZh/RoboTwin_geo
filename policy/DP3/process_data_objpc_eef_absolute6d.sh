#!/bin/bash

task_name=${1}
task_config=${2}
expert_data_num=${3}
object_placeholders=${4:-\{A\},\{B\}}
target_num_points=${5:-1024}
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd "${script_dir}/../.." && pwd)
eef_calibration_path=${6:-${repo_root}/script/real_zed_collection/calibration/three_camera_workspace_extrinsics.yaml}
left_robot_camera_calibration_path=${7:-${repo_root}/script/real_zed_collection/calibration/robot_camera_apriltag_left_global.yaml}
right_robot_camera_calibration_path=${8:-${repo_root}/script/real_zed_collection/calibration/robot_camera_apriltag_right_global.yaml}
eef_frame_mode=${9:-right_base}
output_suffix=${10:--objpc-eef-absolute6d-rightbase}

python scripts/process_data_objpc_eef_absolute6d.py \
    "${task_name}" \
    "${task_config}" \
    "${expert_data_num}" \
    --object_placeholders "${object_placeholders}" \
    --target_num_points "${target_num_points}" \
    --output_suffix="${output_suffix}" \
    --eef_calibration_path "${eef_calibration_path}" \
    --eef_frame_mode "${eef_frame_mode}" \
    --left_robot_camera_calibration_path "${left_robot_camera_calibration_path}" \
    --right_robot_camera_calibration_path "${right_robot_camera_calibration_path}"
