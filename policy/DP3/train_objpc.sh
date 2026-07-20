#!/bin/bash

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/data_paths.sh"

set -euo pipefail

task_name=${1}
task_config=${2}
expert_data_num=${3}
seed=${4}
gpu_id=${5}
resume=${6:-true}
object_placeholders=${7:-\{A\},\{B\}}
dataloader_num_workers=${8:-4}
val_dataloader_num_workers=${9:-2}
pin_memory=${10:-true}
val_pin_memory=${11:-false}
max_val_steps=${12:-2}
point_cloud_num=${13:-1024}
point_cloud_suffix=""
if [ "${point_cloud_num}" != "1024" ]; then
    point_cloud_suffix="-pc${point_cloud_num}"
fi
output_suffix="-objpc${point_cloud_suffix}"

if [ ! -d "${ROBOTWIN_DP3_DATA_ROOT}/${task_name}-${task_config}-${expert_data_num}${output_suffix}.zarr" ]; then
    bash process_data_objpc.sh "${task_name}" "${task_config}" "${expert_data_num}" "${object_placeholders}" "${point_cloud_num}" "${output_suffix}"
fi

DEBUG=False
save_ckpt=True
wandb_mode=online
train_setting="${task_config}${output_suffix}"
exp_name="${task_name}-robot_dp3_objpc-train_objpc"
run_dir="data/outputs/${exp_name}_seed${seed}"
zarr_path="${ROBOTWIN_DP3_DATA_ROOT}/${task_name}-${task_config}-${expert_data_num}${output_suffix}.zarr"

cd 3D-Diffusion-Policy

export HYDRA_FULL_ERROR=1
export CUDA_VISIBLE_DEVICES=${gpu_id}
python train_dp3.py --config-name=robot_dp3_objpc.yaml \
    task_name=${task_name} \
    hydra.run.dir=${run_dir} \
    training.debug=${DEBUG} \
    training.resume=${resume} \
    training.seed=${seed} \
    training.device="cuda:0" \
    exp_name=${exp_name} \
    logging.mode=${wandb_mode} \
    checkpoint.save_ckpt=${save_ckpt} \
    expert_data_num=${expert_data_num} \
    setting=${train_setting} \
    dataloader.num_workers=${dataloader_num_workers} \
    dataloader.pin_memory=${pin_memory} \
    val_dataloader.num_workers=${val_dataloader_num_workers} \
    val_dataloader.pin_memory=${val_pin_memory} \
    training.max_val_steps=${max_val_steps} \
    task.shape_meta.obs.point_cloud.shape=[${point_cloud_num},6] \
    task.dataset.zarr_path=${zarr_path}
