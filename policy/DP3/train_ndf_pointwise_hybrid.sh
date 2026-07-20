#!/bin/bash

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/data_paths.sh"

task_name=${1}
task_config=${2:-"demo_real_zed_sam2_objpc"}
expert_data_num=${3:-50}
seed=${4:-0}
gpu_id=${5:-0}
ndf_ckpt_A=${6:-none}
ndf_ckpt_B=${7:-none}
ndf_device=${8:-cuda:0}
object_placeholders=${9:-\{A\},\{B\}}
ndf_point_num=${10:-128}
ndf_feat_dim=${11:-256}
batch_size=${12:-256}
val_batch_size=${13:-${batch_size}}
use_ema=${14:-true}
gradient_accumulate_every=${15:-1}
encoder_output_dim=${16:-128}
dataloader_num_workers=${17:-4}
val_dataloader_num_workers=${18:-2}
pin_memory=${19:-true}
val_pin_memory=${20:-false}
max_val_steps=${21:-2}
point_cloud_num=${22:-1024}
ndf_dgcnn_placeholders=${23:-}
point_cloud_suffix=""
if [ "${point_cloud_num}" != "1024" ]; then
    point_cloud_suffix="-pc${point_cloud_num}"
fi
output_suffix="-objpc-ndf-pointwise-hybrid${point_cloud_suffix}"
zarr_dir="${ROBOTWIN_DP3_DATA_ROOT}/${task_name}-${task_config}-${expert_data_num}${output_suffix}.zarr"
meta_path="${ROBOTWIN_DP3_DATA_ROOT}/${task_name}-${task_config}-${expert_data_num}${output_suffix}_meta.json"
ndf_feature_placeholders=""

if [ ! -d "${zarr_dir}" ]; then
    bash process_data_ndf_pointwise_hybrid.sh "${task_name}" "${task_config}" "${expert_data_num}" "${ndf_ckpt_A}" "${ndf_ckpt_B}" "${ndf_device}" "${ndf_dgcnn_placeholders}" "${object_placeholders}" "${ndf_point_num}" "${point_cloud_num}" "${output_suffix}" "${ndf_feat_dim}"
fi

if [ -f "${meta_path}" ]; then
    mapfile -t ndf_meta < <(python -c 'import json,sys; m=json.load(open(sys.argv[1], "r", encoding="utf-8")); features=m.get("feature_placeholders"); features=features if features is not None else sorted({p for ep in m.get("episodes", []) for p in ep.get("feature_placeholders", [])}); print(int(m["ndf_num_points"])); print(int(m.get("ndf_feat_dim", 256))); print(",".join(features))' "${meta_path}")
    if [ ${#ndf_meta[@]} -ge 2 ]; then
        ndf_point_num=${ndf_meta[0]}
        ndf_feat_dim=${ndf_meta[1]}
    fi
    if [ ${#ndf_meta[@]} -ge 3 ]; then
        ndf_feature_placeholders="${ndf_meta[2]:-}"
    fi
fi

has_ndf_feature_placeholder() {
    local placeholder=$1
    local placeholder_csv=",${ndf_feature_placeholders},"
    [[ "${placeholder_csv}" == *",${placeholder},"* ]]
}

DEBUG=False
save_ckpt=True
wandb_mode=online
train_setting="${task_config}${output_suffix}"
exp_name="${task_name}-robot_dp3_ndf_pointwise_hybrid-train_ndf"
run_dir="data/outputs/${exp_name}_seed${seed}"
zarr_path="${ROBOTWIN_DP3_DATA_ROOT}/${task_name}-${task_config}-${expert_data_num}${output_suffix}.zarr"

dataset_extra_keys=()
shape_overrides=()
shape_overrides+=("task.shape_meta.obs.point_cloud.shape=[${point_cloud_num},6]")
if { [ "${ndf_ckpt_A}" != "none" ] && [ -n "${ndf_ckpt_A}" ]; } || has_ndf_feature_placeholder "{A}"; then
    dataset_extra_keys+=(ndf_point_cloud_A)
    shape_overrides+=("+task.shape_meta.obs.ndf_point_cloud_A.shape=[${ndf_point_num},$((3 + ndf_feat_dim))]")
    shape_overrides+=("+task.shape_meta.obs.ndf_point_cloud_A.type=point_cloud")
fi
if { [ "${ndf_ckpt_B}" != "none" ] && [ -n "${ndf_ckpt_B}" ]; } || has_ndf_feature_placeholder "{B}"; then
    dataset_extra_keys+=(ndf_point_cloud_B)
    shape_overrides+=("+task.shape_meta.obs.ndf_point_cloud_B.shape=[${ndf_point_num},$((3 + ndf_feat_dim))]")
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
    training.use_ema=${use_ema} \
    training.gradient_accumulate_every=${gradient_accumulate_every} \
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
    training.max_val_steps=${max_val_steps} \
    policy.encoder_output_dim=${encoder_output_dim} \
    task.dataset.zarr_path=${zarr_path} \
    "${dataset_override[@]}" \
    "${shape_overrides[@]}"
