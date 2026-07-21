#!/bin/bash

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/data_paths.sh"
set -euo pipefail

task_name=${1}
task_config=${2}
expert_data_num=${3}
seed=${4}
gpu_id=${5}
route=${6:-baseline}
goal_table=${7:-}
object_placeholders=${8:-\{A\},\{B\}}
point_cloud_num=${9:-1024}
batch_size=${10:-256}
val_batch_size=${11:-${batch_size}}
use_ema=${12:-true}
gradient_accumulate_every=${13:-1}
encoder_output_dim=${14:-128}
dataloader_num_workers=${15:-4}
val_dataloader_num_workers=${16:-2}
pin_memory=${17:-true}
val_pin_memory=${18:-false}
max_val_steps=${19:-2}
resume=${20:-true}

dependency_route=baseline
if [ "${route}" = "ndf_no_direction" ] || [ "${route}" = "ndf_direction" ]; then
    dependency_route=ndf
fi
python scripts/check_shoe_se3_dependencies.py --route "${dependency_route}" --training

if [ "${route}" = "baseline" ]; then
    route_suffix="baseline"
else
    route_suffix="se3-relation-${route//_/-}"
fi
point_cloud_suffix=""
if [ "${point_cloud_num}" != "1024" ]; then
    point_cloud_suffix="-pc${point_cloud_num}"
fi
output_suffix="-objpc-placement-only-${route_suffix}${point_cloud_suffix}"
zarr_dir="${ROBOTWIN_DP3_DATA_ROOT}/${task_name}-${task_config}-${expert_data_num}${output_suffix}.zarr"

if [ ! -d "${zarr_dir}" ]; then
    bash process_data_shoe_se3_placement_comparison.sh \
        "${task_name}" "${task_config}" "${expert_data_num}" "${route}" \
        "${goal_table}" "${object_placeholders}" "${point_cloud_num}" "${output_suffix}"
fi

dataset_overrides=()
if [ "${route}" != "baseline" ]; then
    dataset_overrides+=(+task.dataset.extra_obs_keys=[se3_relation_token_A_to_B])
    dataset_overrides+=(+task.shape_meta.obs.se3_relation_token_A_to_B.shape=[11])
    dataset_overrides+=(+task.shape_meta.obs.se3_relation_token_A_to_B.type=low_dim)
fi

DEBUG=False
save_ckpt=True
wandb_mode=online
train_setting="${task_config}${output_suffix}"
exp_name="${task_name}-robot_dp3_placement_${route}-train"
run_dir="data/outputs/${exp_name}_seed${seed}"
zarr_path="${ROBOTWIN_DP3_DATA_ROOT}/${task_name}-${task_config}-${expert_data_num}${output_suffix}.zarr"

cd 3D-Diffusion-Policy
export HYDRA_FULL_ERROR=1
export CUDA_VISIBLE_DEVICES=${gpu_id}

python train_dp3.py --config-name=robot_dp3_objpc.yaml \
    task_name="${task_name}" \
    hydra.run.dir="${run_dir}" \
    training.debug=${DEBUG} \
    training.resume=${resume} \
    training.seed=${seed} \
    training.device="cuda:0" \
    training.use_ema=${use_ema} \
    training.gradient_accumulate_every=${gradient_accumulate_every} \
    exp_name="${exp_name}" \
    logging.mode=${wandb_mode} \
    checkpoint.save_ckpt=${save_ckpt} \
    expert_data_num=${expert_data_num} \
    setting="${train_setting}" \
    dataloader.batch_size=${batch_size} \
    dataloader.num_workers=${dataloader_num_workers} \
    dataloader.pin_memory=${pin_memory} \
    val_dataloader.batch_size=${val_batch_size} \
    val_dataloader.num_workers=${val_dataloader_num_workers} \
    val_dataloader.pin_memory=${val_pin_memory} \
    training.max_val_steps=${max_val_steps} \
    policy.encoder_output_dim=${encoder_output_dim} \
    task.dataset.zarr_path="${zarr_path}" \
    task.shape_meta.obs.point_cloud.shape=[${point_cloud_num},6] \
    "${dataset_overrides[@]}"
