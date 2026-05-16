#!/bin/bash

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

if [ ! -d "./data/${task_name}-${task_config}-${expert_data_num}-eef-absolute6d-global.zarr" ]; then
    bash process_data_eef_absolute6d_global.sh "${task_name}" "${task_config}" "${expert_data_num}"
fi

bash scripts/train_policy.sh robot_dp3_eef_absolute6d "${task_name}" "${task_config}-eef-absolute6d-global" "${expert_data_num}" train "${seed}" "${gpu_id}" "${dataloader_num_workers}" "${val_dataloader_num_workers}" "${pin_memory}" "${val_pin_memory}" "${max_val_steps}"
