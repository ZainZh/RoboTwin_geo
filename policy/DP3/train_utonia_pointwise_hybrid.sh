#!/bin/bash

task_name=${1}
task_config=${2}
expert_data_num=${3}
seed=${4}
gpu_id=${5}
utonia_device=${6:-cuda:0}
object_placeholders=${7:-\{A\},\{B\}}
utonia_feature_placeholders=${8:-\{A\}}
utonia_point_num=${9:-128}
batch_size=${10:-256}
val_batch_size=${11:-${batch_size}}
use_ema=${12:-true}
gradient_accumulate_every=${13:-1}
encoder_output_dim=${14:-128}
output_suffix="-objpc-utonia-pointwise-hybrid"
zarr_dir="./data/${task_name}-${task_config}-${expert_data_num}${output_suffix}.zarr"
meta_path="./data/${task_name}-${task_config}-${expert_data_num}${output_suffix}_meta.json"

if [ ! -d "${zarr_dir}" ]; then
    bash process_data_utonia_pointwise_hybrid.sh \
        "${task_name}" \
        "${task_config}" \
        "${expert_data_num}" \
        "${utonia_device}" \
        "${object_placeholders}" \
        "${utonia_feature_placeholders}" \
        "${utonia_point_num}"
fi

if [ -f "${meta_path}" ]; then
    mapfile -t utonia_meta < <(python -c 'import json,sys; m=json.load(open(sys.argv[1], "r", encoding="utf-8")); print(int(m["utonia_num_points"])); print(int(m["utonia_feat_dim"]))' "${meta_path}")
    if [ ${#utonia_meta[@]} -ge 2 ]; then
        utonia_point_num=${utonia_meta[0]}
        utonia_feat_dim=${utonia_meta[1]}
    else
        utonia_feat_dim=96
    fi
else
    utonia_feat_dim=96
fi

DEBUG=False
save_ckpt=True
wandb_mode=online
train_setting="${task_config}${output_suffix}"
exp_name="${task_name}-robot_dp3_utonia_pointwise_hybrid-train"
run_dir="data/outputs/${exp_name}_seed${seed}"
zarr_path="../../../data/${task_name}-${task_config}-${expert_data_num}${output_suffix}.zarr"

mapfile -t selected_utonia_placeholders < <(python -c 'import sys; print("\n".join([p.strip() for p in sys.argv[1].split(",") if p.strip()]))' "${utonia_feature_placeholders}")

dataset_extra_keys=()
shape_overrides=()
for placeholder in "${selected_utonia_placeholders[@]}"; do
    key="utonia_point_cloud_$(echo "${placeholder}" | tr -d '{}')"
    dataset_extra_keys+=("${key}")
    shape_overrides+=("+task.shape_meta.obs.${key}.shape=[${utonia_point_num},$((3 + utonia_feat_dim))]")
    shape_overrides+=("+task.shape_meta.obs.${key}.type=point_cloud")
done

dataset_override=()
if [ ${#dataset_extra_keys[@]} -gt 0 ]; then
    joined_keys=$(IFS=,; echo "${dataset_extra_keys[*]}")
    dataset_override=("+task.dataset.extra_obs_keys=[${joined_keys}]")
fi

cd 3D-Diffusion-Policy

export HYDRA_FULL_ERROR=1
export CUDA_VISIBLE_DEVICES=${gpu_id}
python train_dp3.py --config-name=robot_dp3_utonia_pointwise_hybrid.yaml \
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
