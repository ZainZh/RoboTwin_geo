#!/bin/bash

task_name=${1}
task_config=${2}
expert_data_num=${3}
seed=${4}
gpu_id=${5}
object_placeholders=${6:-\{A\},\{B\}}
target_num_points=${7:-1024}
dataloader_num_workers=${8:-4}
val_dataloader_num_workers=${9:-2}
pin_memory=${10:-true}
val_pin_memory=${11:-false}
max_val_steps=${12:-2}

if [ ! -d "./data/${task_name}-${task_config}-${expert_data_num}-objpc-eef-absolute6d-global.zarr" ]; then
    bash process_data_objpc_eef_absolute6d_global.sh "${task_name}" "${task_config}" "${expert_data_num}" "${object_placeholders}" "${target_num_points}"
fi

bash scripts/train_policy.sh robot_dp3_objpc_eef_absolute6d "${task_name}" "${task_config}-objpc-eef-absolute6d-global" "${expert_data_num}" train "${seed}" "${gpu_id}" "${dataloader_num_workers}" "${val_dataloader_num_workers}" "${pin_memory}" "${val_pin_memory}" "${max_val_steps}"
