#!/bin/bash

task_name=${1}
task_config=${2}
expert_data_num=${3}
semantic_ckpt_A=${4:-none}
semantic_ckpt_B=${5:-none}
semantic_device=${6:-cuda:0}
object_placeholders=${7:-\{A\},\{B\}}
semantic_point_num=${8:-1024}
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd "${script_dir}/../.." && pwd)
eef_calibration_path=${9:-${repo_root}/script/real_zed_collection/calibration/three_camera_workspace_extrinsics.yaml}
left_robot_camera_calibration_path=${10:-${repo_root}/script/real_zed_collection/calibration/robot_camera_apriltag_left_global.yaml}
right_robot_camera_calibration_path=${11:-${repo_root}/script/real_zed_collection/calibration/robot_camera_apriltag_right_global.yaml}
eef_frame_mode=${12:-reference_camera}

extra_args=()
if [ "${semantic_ckpt_A}" != "none" ] && [ -n "${semantic_ckpt_A}" ]; then
    extra_args+=(--semantic_model "{A}=${semantic_ckpt_A}")
fi
if [ "${semantic_ckpt_B}" != "none" ] && [ -n "${semantic_ckpt_B}" ]; then
    extra_args+=(--semantic_model "{B}=${semantic_ckpt_B}")
fi

python scripts/process_data_semantic_pointwise_hybrid_eef_absolute6d_global.py \
    "${task_name}" \
    "${task_config}" \
    "${expert_data_num}" \
    --object_placeholders "${object_placeholders}" \
    --semantic_device "${semantic_device}" \
    --semantic_num_points "${semantic_point_num}" \
    --eef_calibration_path "${eef_calibration_path}" \
    --eef_frame_mode "${eef_frame_mode}" \
    --left_robot_camera_calibration_path "${left_robot_camera_calibration_path}" \
    --right_robot_camera_calibration_path "${right_robot_camera_calibration_path}" \
    "${extra_args[@]}"
