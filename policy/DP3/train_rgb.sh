#!/bin/bash

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/data_paths.sh"

task_name=${1}
task_config=${2}
expert_data_num=${3}
seed=${4}
gpu_id=${5}
dataloader_num_workers=${6:-4}
val_dataloader_num_workers=${7:-2}
pin_memory=${8:-true}
val_pin_memory=${9:-false}
max_val_steps=${10:-2}
point_cloud_num=${11:-1024}
point_cloud_suffix=""
if [ "${point_cloud_num}" != "1024" ]; then
    point_cloud_suffix="-pc${point_cloud_num}"
fi

if [ ! -d "${ROBOTWIN_DP3_DATA_ROOT}/${task_name}-${task_config}-${expert_data_num}${point_cloud_suffix}.zarr" ]; then
    bash process_data.sh "${task_name}" "${task_config}" "${expert_data_num}" "${point_cloud_num}" "${point_cloud_suffix}"
fi

bash scripts/train_policy_rgb.sh robot_dp3 "${task_name}" "${task_config}${point_cloud_suffix}" "${expert_data_num}" train "${seed}" "${gpu_id}" "${dataloader_num_workers}" "${val_dataloader_num_workers}" "${pin_memory}" "${val_pin_memory}" "${max_val_steps}" "${point_cloud_num}" "${ROBOTWIN_DP3_DATA_ROOT}/${task_name}-${task_config}-${expert_data_num}${point_cloud_suffix}.zarr"
