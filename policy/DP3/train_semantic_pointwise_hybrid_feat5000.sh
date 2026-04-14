#!/bin/bash

task_name=${1}
task_config=${2}
expert_data_num=${3}
seed=${4}
gpu_id=${5}
semantic_ckpt_A=${6:-none}
semantic_ckpt_B=${7:-none}
semantic_device=${8:-cuda:0}
object_placeholders=${9:-\{A\},\{B\}}
semantic_point_num=${10:-5000}
semantic_feat_dim=${11:-128}
batch_size=${12:-256}
val_batch_size=${13:-${batch_size}}
use_ema=${14:-true}
gradient_accumulate_every=${15:-1}
encoder_output_dim=${16:-128}
output_suffix="-objpc-semantic-pointwise-hybrid-feat5000"
zarr_dir="./data/${task_name}-${task_config}-${expert_data_num}${output_suffix}.zarr"
meta_path="./data/${task_name}-${task_config}-${expert_data_num}${output_suffix}_meta.json"

if [ ! -d "${zarr_dir}" ]; then
    bash process_data_semantic_pointwise_hybrid_feat5000.sh "${task_name}" "${task_config}" "${expert_data_num}" "${semantic_ckpt_A}" "${semantic_ckpt_B}" "${semantic_device}" "${object_placeholders}" "${semantic_point_num}"
fi

if [ -f "${meta_path}" ]; then
    mapfile -t semantic_meta < <(python -c 'import json,sys; m=json.load(open(sys.argv[1], "r", encoding="utf-8")); print(int(m["semantic_num_points"])); print(int(m["semantic_feat_dim"]))' "${meta_path}")
    if [ ${#semantic_meta[@]} -ge 2 ]; then
        semantic_point_num=${semantic_meta[0]}
        semantic_feat_dim=${semantic_meta[1]}
    fi
fi

DEBUG=False
save_ckpt=True
wandb_mode=online
train_setting="${task_config}-objpc-semantic-pointwise-hybrid-feat5000"
exp_name="${task_name}-robot_dp3_semantic_pointwise_hybrid_feat5000-train"
run_dir="data/outputs/${exp_name}_seed${seed}"
zarr_path="../../../data/${task_name}-${task_config}-${expert_data_num}${output_suffix}.zarr"

dataset_extra_keys=()
shape_overrides=()
if [ "${semantic_ckpt_A}" != "none" ] && [ -n "${semantic_ckpt_A}" ]; then
    dataset_extra_keys+=(semantic_point_cloud_A)
    shape_overrides+=("+task.shape_meta.obs.semantic_point_cloud_A.shape=[${semantic_point_num},$((3 + semantic_feat_dim))]")
    shape_overrides+=("+task.shape_meta.obs.semantic_point_cloud_A.type=point_cloud")
fi
if [ "${semantic_ckpt_B}" != "none" ] && [ -n "${semantic_ckpt_B}" ]; then
    dataset_extra_keys+=(semantic_point_cloud_B)
    shape_overrides+=("+task.shape_meta.obs.semantic_point_cloud_B.shape=[${semantic_point_num},$((3 + semantic_feat_dim))]")
    shape_overrides+=("+task.shape_meta.obs.semantic_point_cloud_B.type=point_cloud")
fi

dataset_override=()
if [ ${#dataset_extra_keys[@]} -gt 0 ]; then
    joined_keys=$(IFS=,; echo "${dataset_extra_keys[*]}")
    dataset_override=("+task.dataset.extra_obs_keys=[${joined_keys}]")
fi

cd 3D-Diffusion-Policy

export HYDRA_FULL_ERROR=1
export CUDA_VISIBLE_DEVICES=${gpu_id}
python train_dp3.py --config-name=robot_dp3_semantic_pointwise_hybrid_feat5000.yaml \
    task_name=${task_name} \
    hydra.run.dir=${run_dir} \
    training.debug=${DEBUG} \
    training.seed=${seed} \
    training.device="cuda:0" \
    training.use_ema=${use_ema} \
    training.gradient_accumulate_every=${gradient_accumulate_every} \
    exp_name=${exp_name} \
    logging.mode=${wandb_mode} \
    checkpoint.save_ckpt=${save_ckpt} \
    expert_data_num=${expert_data_num} \
    setting=${train_setting} \
    dataloader.batch_size=${batch_size} \
    val_dataloader.batch_size=${val_batch_size} \
    policy.encoder_output_dim=${encoder_output_dim} \
    task.dataset.zarr_path=${zarr_path} \
    "${dataset_override[@]}" \
    "${shape_overrides[@]}"
