#!/bin/bash

DEFAULT_OBJECT_PLACEHOLDERS="{A},{B}"

task_name=${1}
task_config=${2}
expert_data_num=${3}
seed=${4}
gpu_id=${5}
ndf_ckpt_A=${6:-none}
ndf_ckpt_B=${7:-none}
ndf_device=${8:-cuda:0}
ndf_dgcnn_placeholders=${9:-}
object_placeholders=${10:-${DEFAULT_OBJECT_PLACEHOLDERS}}
ndf_point_num=${11:-128}
batch_size=${12:-256}
val_batch_size=${13:-${batch_size}}
gradient_accumulate_every=${14:-1}
output_suffix="-objpc-ndf-pointwise-hybrid-interact"
zarr_dir="./data/${task_name}-${task_config}-${expert_data_num}${output_suffix}.zarr"

if [ ! -d "${zarr_dir}" ]; then
    bash process_data_ndf_pointwise_hybrid_interact.sh \
        "${task_name}" \
        "${task_config}" \
        "${expert_data_num}" \
        "${ndf_ckpt_A}" \
        "${ndf_ckpt_B}" \
        "${ndf_device}" \
        "${ndf_dgcnn_placeholders}" \
        "${object_placeholders}" \
        "${ndf_point_num}"
fi

DEBUG=False
save_ckpt=True
wandb_mode=online
train_setting="${task_config}${output_suffix}"
exp_name="${task_name}-robot_dp3_ndf_pointwise_hybrid_interact-train_ndf"
run_dir="data/outputs/${exp_name}_seed${seed}"
zarr_path="../../../data/${task_name}-${task_config}-${expert_data_num}${output_suffix}.zarr"

dataset_extra_keys=()
shape_overrides=()
if [ "${ndf_ckpt_A}" != "none" ] && [ -n "${ndf_ckpt_A}" ]; then
    dataset_extra_keys+=(ndf_point_cloud_A)
    dataset_extra_keys+=(ndf_interact_point_cloud_B_from_A)
    shape_overrides+=("+task.shape_meta.obs.ndf_point_cloud_A.shape=[${ndf_point_num},259]")
    shape_overrides+=("+task.shape_meta.obs.ndf_point_cloud_A.type=point_cloud")
    shape_overrides+=("+task.shape_meta.obs.ndf_interact_point_cloud_B_from_A.shape=[${ndf_point_num},259]")
    shape_overrides+=("+task.shape_meta.obs.ndf_interact_point_cloud_B_from_A.type=point_cloud")
fi
if [ "${ndf_ckpt_B}" != "none" ] && [ -n "${ndf_ckpt_B}" ]; then
    dataset_extra_keys+=(ndf_point_cloud_B)
    dataset_extra_keys+=(ndf_interact_point_cloud_A_from_B)
    shape_overrides+=("+task.shape_meta.obs.ndf_point_cloud_B.shape=[${ndf_point_num},259]")
    shape_overrides+=("+task.shape_meta.obs.ndf_point_cloud_B.type=point_cloud")
    shape_overrides+=("+task.shape_meta.obs.ndf_interact_point_cloud_A_from_B.shape=[${ndf_point_num},259]")
    shape_overrides+=("+task.shape_meta.obs.ndf_interact_point_cloud_A_from_B.type=point_cloud")
fi

dataset_override=()
if [ ${#dataset_extra_keys[@]} -gt 0 ]; then
    joined_keys=$(IFS=,; echo "${dataset_extra_keys[*]}")
    dataset_override=("+task.dataset.extra_obs_keys=[${joined_keys}]")
fi

cd 3D-Diffusion-Policy

export HYDRA_FULL_ERROR=1
export CUDA_VISIBLE_DEVICES=${gpu_id}
python train_dp3.py --config-name=robot_dp3_ndf_pointwise_hybrid_interact.yaml \
    task_name=${task_name} \
    hydra.run.dir=${run_dir} \
    training.debug=${DEBUG} \
    training.seed=${seed} \
    training.device="cuda:0" \
    training.gradient_accumulate_every=${gradient_accumulate_every} \
    exp_name=${exp_name} \
    logging.mode=${wandb_mode} \
    checkpoint.save_ckpt=${save_ckpt} \
    expert_data_num=${expert_data_num} \
    setting=${train_setting} \
    dataloader.batch_size=${batch_size} \
    val_dataloader.batch_size=${val_batch_size} \
    task.dataset.zarr_path=${zarr_path} \
    "${dataset_override[@]}" \
    "${shape_overrides[@]}"
