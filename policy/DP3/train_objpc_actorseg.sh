#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

task_name=${1}
task_config=${2}
expert_data_num=${3}
seed=${4}
gpu_id=${5}
object_placeholders=${6:-\{A\},\{B\}}
actorseg_camera_names=${7:-head_camera,front_camera}
dataloader_num_workers=${8:-4}
val_dataloader_num_workers=${9:-2}
pin_memory=${10:-true}
val_pin_memory=${11:-false}
max_val_steps=${12:-2}

zarr_dir="./data/${task_name}-${task_config}-${expert_data_num}-objpc-actorseg.zarr"

zarr_complete() {
    local root="$1"
    [ -d "${root}" ] && \
    [ -e "${root}/data/point_cloud" ] && \
    [ -e "${root}/data/state" ] && \
    [ -e "${root}/data/action" ] && \
    [ -e "${root}/meta/episode_ends" ]
}

if zarr_complete "${zarr_dir}"; then
    echo "Found complete actor-segmentation zarr at ${zarr_dir}, skipping preprocessing."
else
    echo "Building or resuming actor-segmentation zarr at ${zarr_dir} ..."
    bash "${SCRIPT_DIR}/process_data_objpc_actorseg.sh" \
        "${task_name}" \
        "${task_config}" \
        "${expert_data_num}" \
        "${object_placeholders}" \
        "${actorseg_camera_names}"
fi

DEBUG=False
save_ckpt=True
wandb_mode=online
train_setting="${task_config}-objpc-actorseg"
exp_name="${task_name}-robot_dp3_objpc-train_objpc_actorseg"
run_dir="data/outputs/${exp_name}_seed${seed}"
zarr_path="../../../data/${task_name}-${task_config}-${expert_data_num}-objpc-actorseg.zarr"

cd 3D-Diffusion-Policy

export HYDRA_FULL_ERROR=1
export CUDA_VISIBLE_DEVICES=${gpu_id}
python train_dp3.py --config-name=robot_dp3_objpc_actorseg.yaml \
    task_name=${task_name} \
    hydra.run.dir=${run_dir} \
    training.debug=${DEBUG} \
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
    task.dataset.zarr_path=${zarr_path}
