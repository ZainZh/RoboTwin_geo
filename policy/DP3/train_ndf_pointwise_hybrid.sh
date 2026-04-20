#!/bin/bash

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${script_dir}/ndf_pointwise_arg_utils.sh"
normalize_ndf_train_args "$@"

if [ "${ndf_legacy_shifted}" = true ]; then
    echo "[train_ndf_pointwise_hybrid.sh] detected legacy invocation without explicit ndf_dgcnn_placeholders; treating '${object_placeholders}' as object_placeholders." >&2
fi

if [ ! -d "./data/${task_name}-${task_config}-${expert_data_num}-objpc-ndf-pointwise-hybrid.zarr" ]; then
    bash process_data_ndf_pointwise_hybrid.sh "${task_name}" "${task_config}" "${expert_data_num}" "${ndf_ckpt_A}" "${ndf_ckpt_B}" "${ndf_device}" "${ndf_dgcnn_placeholders}" "${object_placeholders}" "${ndf_point_num}"
fi

DEBUG=False
save_ckpt=True
wandb_mode=online
train_setting="${task_config}-objpc-ndf-pointwise-hybrid"
exp_name="${task_name}-robot_dp3_ndf_pointwise_hybrid-train_ndf"
run_dir="data/outputs/${exp_name}_seed${seed}"
zarr_path="../../../data/${task_name}-${task_config}-${expert_data_num}-objpc-ndf-pointwise-hybrid.zarr"

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
python train_dp3.py --config-name=robot_dp3_ndf_pointwise_hybrid.yaml \
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
