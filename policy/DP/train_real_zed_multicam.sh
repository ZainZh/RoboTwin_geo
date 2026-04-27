#!/bin/bash

task_name=${1}
task_config=${2}
expert_data_num=${3}
seed=${4}
action_dim=${5:-14}
gpu_id=${6}
camera_labels=${7:-global,left,right}
head_camera_type=${8:-Large_L515}
resume=${9:-true}
meta_path=${10:-}
batch_size=${11:-128}
val_batch_size=${12:-${batch_size}}
gradient_accumulate_every=${13:-1}

DEBUG=False
save_ckpt=True
wandb_mode=online
camera_setting=${camera_labels//,/_}
train_setting="${task_config}-dp-${camera_setting}"
exp_name="${task_name}-robot_dp-real_zed_multicam_train"
run_dir="data/outputs/${exp_name}_seed${seed}"
zarr_path="data/${task_name}-${train_setting}-${expert_data_num}.zarr"

echo -e "\033[33mgpu id (to use): ${gpu_id}\033[0m"
echo -e "\033[33mReal-ZED DP multicam: ${camera_labels}, head_camera_type: ${head_camera_type}\033[0m"

export HYDRA_FULL_ERROR=1
export CUDA_VISIBLE_DEVICES=${gpu_id}

if [ ! -d "./${zarr_path}" ]; then
    bash process_data_real_zed_multicam.sh "${task_name}" "${task_config}" "${expert_data_num}" "${camera_labels}" "${meta_path}"
fi

python train.py --config-name=robot_dp_${action_dim}.yaml \
    task=default_task_${action_dim}_multicam \
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
