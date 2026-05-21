#!/bin/bash
set -euo pipefail

task_name=${1}
task_config=${2}
expert_data_num=${3}
seed=${4}
gpu_id=${5}
camera_label=${6:-global}
head_camera_type=${7:-Large_L515}
resume=${8:-true}
meta_path=${9:-}
batch_size=${10:-128}
val_batch_size=${11:-${batch_size}}
gradient_accumulate_every=${12:-1}

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd "${script_dir}/../.." && pwd)
eef_calibration_path=${13:-${repo_root}/script/real_zed_collection/calibration/three_camera_workspace_extrinsics.yaml}
left_robot_camera_calibration_path=${14:-${repo_root}/script/real_zed_collection/calibration/robot_camera_apriltag_left_global.yaml}
right_robot_camera_calibration_path=${15:-${repo_root}/script/real_zed_collection/calibration/robot_camera_apriltag_right_global.yaml}
eef_frame_mode=${16:-reference_camera}

camera_resize_hw_for_type() {
    case "$1" in
        L515) echo "180,320" ;;
        Large_L515) echo "360,640" ;;
        D435) echo "240,320" ;;
        Large_D435) echo "480,640" ;;
        *)
            echo "Unknown head_camera_type '$1'. Add it to camera_resize_hw_for_type in $0." >&2
            return 1
            ;;
    esac
}

DEBUG=False
wandb_mode=online
train_setting="${task_config}-dp-${camera_label}-eef-absolute6d-global"
exp_name="${task_name}-robot_dp-real_zed_eef_train"
zarr_path="data/${task_name}-${train_setting}-${expert_data_num}.zarr"

echo -e "\033[33mgpu id (to use): ${gpu_id}\033[0m"
echo -e "\033[33mReal-ZED DP EEF camera: ${camera_label}, eef_frame_mode: ${eef_frame_mode}\033[0m"

export HYDRA_FULL_ERROR=1
export CUDA_VISIBLE_DEVICES=${gpu_id}

if [ ! -d "./${zarr_path}" ]; then
    resize_hw="$(camera_resize_hw_for_type "${head_camera_type}")"
    bash process_data_real_zed.sh \
        "${task_name}" \
        "${task_config}" \
        "${expert_data_num}" \
        "${camera_label}" \
        "${meta_path}" \
        "${resize_hw}" \
        --output_zarr "${zarr_path}" \
        --action_mode eef_absolute6d \
        --eef_calibration_path "${eef_calibration_path}" \
        --eef_frame_mode "${eef_frame_mode}" \
        --left_robot_camera_calibration_path "${left_robot_camera_calibration_path}" \
        --right_robot_camera_calibration_path "${right_robot_camera_calibration_path}"
fi

python train.py --config-name=robot_dp_20.yaml \
    task.name=${task_name} \
    task.dataset.zarr_path="${zarr_path}" \
    training.debug=${DEBUG} \
    training.resume=${resume} \
    training.seed=${seed} \
    training.device="cuda:0" \
    training.gradient_accumulate_every=${gradient_accumulate_every} \
    dataloader.batch_size=${batch_size} \
    val_dataloader.batch_size=${val_batch_size} \
    exp_name=${exp_name} \
    logging.mode=${wandb_mode} \
    setting=${train_setting} \
    expert_data_num=${expert_data_num} \
    head_camera_type=${head_camera_type}
