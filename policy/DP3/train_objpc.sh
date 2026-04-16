#!/bin/bash

task_name=${1}
task_config=${2}
expert_data_num=${3}
seed=${4}
gpu_id=${5}
resume=${6:-true}
object_placeholders=${7:-\{A\},\{B\}}

if [ ! -d "./data/${task_name}-${task_config}-${expert_data_num}-objpc.zarr" ]; then
    bash process_data_objpc.sh "${task_name}" "${task_config}" "${expert_data_num}" "${object_placeholders}"
fi

DEBUG=False
save_ckpt=True
wandb_mode=online
train_setting="${task_config}-objpc"
exp_name="${task_name}-robot_dp3_objpc-train_objpc"
run_dir="data/outputs/${exp_name}_seed${seed}"
zarr_path="../../../data/${task_name}-${task_config}-${expert_data_num}-objpc.zarr"

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
    task.dataset.zarr_path=${zarr_path}
