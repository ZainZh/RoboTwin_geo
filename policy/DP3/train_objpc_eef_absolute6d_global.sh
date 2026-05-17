#!/bin/bash
set -euo pipefail

task_name=${1}
task_config=${2}
expert_data_num=${3}
seed=${4}
gpu_id=${5}
object_placeholders=${6:-\{A\},\{B\}}
point_cloud_num=${7:-1024}
dataloader_num_workers=${8:-4}
val_dataloader_num_workers=${9:-2}
pin_memory=${10:-true}
val_pin_memory=${11:-false}
max_val_steps=${12:-2}
point_cloud_suffix=""
if [ "${point_cloud_num}" != "1024" ]; then
    point_cloud_suffix="-pc${point_cloud_num}"
fi
output_suffix="-objpc-eef-absolute6d-global${point_cloud_suffix}"
zarr_path="../../../data/${task_name}-${task_config}-${expert_data_num}${output_suffix}.zarr"

if [ ! -d "./data/${task_name}-${task_config}-${expert_data_num}${output_suffix}.zarr" ]; then
    bash process_data_objpc_eef_absolute6d_global.sh "${task_name}" "${task_config}" "${expert_data_num}" "${object_placeholders}" "${point_cloud_num}" "" "" "" reference_camera "${output_suffix}"
fi

bash scripts/train_policy.sh robot_dp3_objpc_eef_absolute6d "${task_name}" "${task_config}${output_suffix}" "${expert_data_num}" train "${seed}" "${gpu_id}" "${dataloader_num_workers}" "${val_dataloader_num_workers}" "${pin_memory}" "${val_pin_memory}" "${max_val_steps}" "${point_cloud_num}" "${zarr_path}"
