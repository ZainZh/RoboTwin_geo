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
eef_frame_mode=${12:-right_base}
target_num_points=${13:-1024}
output_suffix=${14:--objpc-semantic-pointwise-hybrid-semdebugref-eef-absolute6d-rightbase}
semantic_input_color_mode=${15:-debug_placeholder}
semantic_forward_mode=${16:-reference}

extra_args=()
if [ "${semantic_ckpt_A}" != "none" ] && [ -n "${semantic_ckpt_A}" ]; then
    extra_args+=(--semantic_model "{A}=${semantic_ckpt_A}")
fi
if [ "${semantic_ckpt_B}" != "none" ] && [ -n "${semantic_ckpt_B}" ]; then
    extra_args+=(--semantic_model "{B}=${semantic_ckpt_B}")
fi

python scripts/process_data_semantic_pointwise_hybrid_eef_absolute6d.py \
    "${task_name}" \
    "${task_config}" \
    "${expert_data_num}" \
    --object_placeholders "${object_placeholders}" \
    --semantic_device "${semantic_device}" \
    --semantic_num_points "${semantic_point_num}" \
    --semantic_input_color_mode "${semantic_input_color_mode}" \
    --semantic_forward_mode "${semantic_forward_mode}" \
    --target_num_points "${target_num_points}" \
    --output_suffix="${output_suffix}" \
    --eef_calibration_path "${eef_calibration_path}" \
    --eef_frame_mode "${eef_frame_mode}" \
    --left_robot_camera_calibration_path "${left_robot_camera_calibration_path}" \
    --right_robot_camera_calibration_path "${right_robot_camera_calibration_path}" \
    "${extra_args[@]}"
