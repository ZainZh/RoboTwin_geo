#!/bin/bash

task_name=${1}
task_config=${2}
expert_data_num=${3}
seed=${4}
gpu_id=${5}
ndf_ckpt_A=${6:-none}
ndf_ckpt_B=${7:-none}
ndf_device=${8:-cuda:0}
ndf_dgcnn_placeholders=${9:-}
object_placeholders=${10:-\{A\},\{B\}}
dataloader_num_workers=${11:-4}
val_dataloader_num_workers=${12:-2}
pin_memory=${13:-true}
val_pin_memory=${14:-false}
max_val_steps=${15:-2}

if [ ! -d "./data/${task_name}-${task_config}-${expert_data_num}-objpc-ndf.zarr" ]; then
    bash process_data_ndf.sh "${task_name}" "${task_config}" "${expert_data_num}" "${ndf_ckpt_A}" "${ndf_ckpt_B}" "${ndf_device}" "${ndf_dgcnn_placeholders}" "${object_placeholders}"
fi

DEBUG=False
save_ckpt=True
wandb_mode=online
train_setting="${task_config}-objpc-ndf"
exp_name="${task_name}-robot_dp3_ndf-train_ndf"
run_dir="data/outputs/${exp_name}_seed${seed}"
zarr_path="../../../data/${task_name}-${task_config}-${expert_data_num}-objpc-ndf.zarr"

dataset_extra_keys=()
shape_overrides=()
if [ "${ndf_ckpt_A}" != "none" ] && [ -n "${ndf_ckpt_A}" ]; then
    dataset_extra_keys+=(ndf_feat_A)
    shape_overrides+=("+task.shape_meta.obs.ndf_feat_A.shape=[256]")
    shape_overrides+=("+task.shape_meta.obs.ndf_feat_A.type=low_dim")
fi
if [ "${ndf_ckpt_B}" != "none" ] && [ -n "${ndf_ckpt_B}" ]; then
    dataset_extra_keys+=(ndf_feat_B)
    shape_overrides+=("+task.shape_meta.obs.ndf_feat_B.shape=[256]")
    shape_overrides+=("+task.shape_meta.obs.ndf_feat_B.type=low_dim")
fi

dataset_override=()
if [ ${#dataset_extra_keys[@]} -gt 0 ]; then
    joined_keys=$(IFS=,; echo "${dataset_extra_keys[*]}")
    dataset_override=("+task.dataset.extra_obs_keys=[${joined_keys}]")
fi

cd 3D-Diffusion-Policy

export HYDRA_FULL_ERROR=1
export CUDA_VISIBLE_DEVICES=${gpu_id}
python train_dp3.py --config-name=robot_dp3_ndf.yaml \
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
    task.dataset.zarr_path=${zarr_path} \
    "${dataset_override[@]}" \
    "${shape_overrides[@]}"
