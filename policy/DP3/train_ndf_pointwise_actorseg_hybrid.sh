#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/ndf_pointwise_arg_utils.sh"
raw_args=("$@")
normalize_ndf_train_args "$@"
actorseg_camera_names="${raw_args[19]:-head_camera,front_camera}"

if [ -n "${raw_args[11]:-}" ] && ! looks_like_integer_arg "${raw_args[11]}" ]; then
    actorseg_camera_names="${raw_args[11]}"
    batch_size=256
    val_batch_size=${batch_size}
    gradient_accumulate_every=1
    dataloader_num_workers=4
    val_dataloader_num_workers=2
    pin_memory=true
    val_pin_memory=false
    max_val_steps=2
fi

if [ "${ndf_legacy_shifted}" = true ]; then
    actorseg_camera_names="${raw_args[17]:-head_camera,front_camera}"
    if [ -n "${raw_args[10]:-}" ] && ! looks_like_integer_arg "${raw_args[10]}" ]; then
        actorseg_camera_names="${raw_args[10]}"
    fi
    echo "[train_ndf_pointwise_actorseg_hybrid.sh] detected legacy invocation without explicit ndf_dgcnn_placeholders; treating '${object_placeholders}' as object_placeholders." >&2
fi

cd "${SCRIPT_DIR}"

zarr_dir="./data/${task_name}-${task_config}-${expert_data_num}-objpc-actorseg-ndf-pointwise-hybrid.zarr"

zarr_complete() {
    local root="$1"
    [ -d "${root}" ] && \
    [ -e "${root}/data/point_cloud" ] && \
    [ -e "${root}/data/state" ] && \
    [ -e "${root}/data/action" ] && \
    [ -e "${root}/meta/episode_ends" ]
}

if zarr_complete "${zarr_dir}"; then
    echo "Found complete actorseg NDF hybrid zarr at ${zarr_dir}, skipping preprocessing."
else
    echo "Building or resuming actorseg NDF hybrid zarr at ${zarr_dir} ..."
    bash "${SCRIPT_DIR}/process_data_ndf_pointwise_actorseg_hybrid.sh" \
        "${task_name}" \
        "${task_config}" \
        "${expert_data_num}" \
        "${ndf_ckpt_A}" \
        "${ndf_ckpt_B}" \
        "${ndf_device}" \
        "${ndf_dgcnn_placeholders}" \
        "${object_placeholders}" \
        "${ndf_point_num}" \
        "${actorseg_camera_names}"
fi

DEBUG=False
save_ckpt=True
wandb_mode=online
train_setting="${task_config}-objpc-actorseg-ndf-pointwise-hybrid"
exp_name="${task_name}-robot_dp3_objpc_actorseg_ndf_pointwise_hybrid-train_ndf"
run_dir="data/outputs/${exp_name}_seed${seed}"
zarr_path="../../../data/${task_name}-${task_config}-${expert_data_num}-objpc-actorseg-ndf-pointwise-hybrid.zarr"

dataset_extra_keys=()
shape_overrides=()
if [ "${ndf_ckpt_A}" != "none" ] && [ -n "${ndf_ckpt_A}" ]; then
    dataset_extra_keys+=(ndf_point_cloud_A)
    shape_overrides+=("+task.shape_meta.obs.ndf_point_cloud_A.shape=[${ndf_point_num},259]")
    shape_overrides+=("+task.shape_meta.obs.ndf_point_cloud_A.type=point_cloud")
fi
if [ "${ndf_ckpt_B}" != "none" ] && [ -n "${ndf_ckpt_B}" ]; then
    dataset_extra_keys+=(ndf_point_cloud_B)
    shape_overrides+=("+task.shape_meta.obs.ndf_point_cloud_B.shape=[${ndf_point_num},259]")
    shape_overrides+=("+task.shape_meta.obs.ndf_point_cloud_B.type=point_cloud")
fi

dataset_override=()
if [ ${#dataset_extra_keys[@]} -gt 0 ]; then
    joined_keys=$(IFS=,; echo "${dataset_extra_keys[*]}")
    dataset_override=("+task.dataset.extra_obs_keys=[${joined_keys}]")
fi

cd 3D-Diffusion-Policy

export HYDRA_FULL_ERROR=1
export CUDA_VISIBLE_DEVICES=${gpu_id}
python train_dp3.py --config-name=robot_dp3_objpc_actorseg_ndf_pointwise_hybrid.yaml \
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
    dataloader.batch_size=${batch_size} \
    dataloader.num_workers=${dataloader_num_workers} \
    dataloader.pin_memory=${pin_memory} \
    val_dataloader.batch_size=${val_batch_size} \
    val_dataloader.num_workers=${val_dataloader_num_workers} \
    val_dataloader.pin_memory=${val_pin_memory} \
    training.gradient_accumulate_every=${gradient_accumulate_every} \
    training.max_val_steps=${max_val_steps} \
    task.dataset.zarr_path=${zarr_path} \
    "${dataset_override[@]}" \
    "${shape_overrides[@]}"
