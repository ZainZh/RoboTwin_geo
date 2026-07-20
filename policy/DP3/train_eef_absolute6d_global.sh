#!/bin/bash

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/data_paths.sh"
set -euo pipefail

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
output_suffix="-eef-absolute6d-global${point_cloud_suffix}"
zarr_path="${ROBOTWIN_DP3_DATA_ROOT}/${task_name}-${task_config}-${expert_data_num}${output_suffix}.zarr"

if [ ! -d "${ROBOTWIN_DP3_DATA_ROOT}/${task_name}-${task_config}-${expert_data_num}${output_suffix}.zarr" ]; then
    bash process_data_eef_absolute6d_global.sh "${task_name}" "${task_config}" "${expert_data_num}" "" "" "" reference_camera "${point_cloud_num}" "${output_suffix}"
fi

bash scripts/train_policy.sh robot_dp3_eef_absolute6d "${task_name}" "${task_config}${output_suffix}" "${expert_data_num}" train "${seed}" "${gpu_id}" "${dataloader_num_workers}" "${val_dataloader_num_workers}" "${pin_memory}" "${val_pin_memory}" "${max_val_steps}" "${point_cloud_num}" "${zarr_path}"
