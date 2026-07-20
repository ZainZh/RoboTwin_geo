#!/bin/bash
set -euo pipefail

DEFAULT_OBJECT_PLACEHOLDERS="{A},{B}"
task_name=${1}
task_config=${2:-demo_clean_3d_object_pc}
expert_data_num=${3:-50}
seed=${4:-0}
gpu_id=${5:-0}
ndf_ckpt_A=${6:-none}
ndf_ckpt_B=${7:-none}
ndf_device=${8:-cuda:0}
object_placeholders=${9:-${DEFAULT_OBJECT_PLACEHOLDERS}}
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
num_epochs=${24:-3000}
wandb_mode=${25:-online}
checkpoint_every=${26:-${num_epochs}}

point_cloud_suffix=""
if [ "${point_cloud_num}" != "1024" ]; then
    point_cloud_suffix="-pc${point_cloud_num}"
fi
output_suffix="-objpc-ndf-relation-v2${point_cloud_suffix}"
zarr_dir="./data/${task_name}-${task_config}-${expert_data_num}${output_suffix}.zarr"
meta_path="./data/${task_name}-${task_config}-${expert_data_num}${output_suffix}_meta.json"
ndf_feature_placeholders=""

export CUDA_VISIBLE_DEVICES=${gpu_id}
export HYDRA_FULL_ERROR=1

if [ ! -d "${zarr_dir}" ]; then
    bash process_data_ndf_relation_v2.sh \
        "${task_name}" "${task_config}" "${expert_data_num}" \
        "${ndf_ckpt_A}" "${ndf_ckpt_B}" "${ndf_device}" \
        "${ndf_dgcnn_placeholders}" "${object_placeholders}" \
        "${ndf_point_num}" "${point_cloud_num}" "${output_suffix}" "${ndf_feat_dim}"
fi

python scripts/validate_ndf_relation_v2_meta.py "${meta_path}"

mapfile -t ndf_meta < <(python -c 'import json,sys; m=json.load(open(sys.argv[1], "r", encoding="utf-8")); features=m.get("feature_placeholders"); features=features if features is not None else sorted({p for ep in m.get("episodes", []) for p in ep.get("feature_placeholders", [])}); print(int(m["ndf_num_points"])); print(int(m.get("ndf_feat_dim", 256))); print(",".join(features))' "${meta_path}")
if [ ${#ndf_meta[@]} -ge 2 ]; then
    ndf_point_num=${ndf_meta[0]}
    ndf_feat_dim=${ndf_meta[1]}
fi
if [ ${#ndf_meta[@]} -ge 3 ]; then
    ndf_feature_placeholders="${ndf_meta[2]:-}"
fi

has_placeholder() {
    local placeholder=$1
    local placeholder_csv=",${object_placeholders},"
    [[ "${placeholder_csv}" == *",${placeholder},"* ]]
}

has_ndf_feature_placeholder() {
    local placeholder=$1
    local placeholder_csv=",${ndf_feature_placeholders},"
    [[ "${placeholder_csv}" == *",${placeholder},"* ]]
}

add_point_cloud_shape() {
    local key=$1
    dataset_extra_keys+=("${key}")
    shape_overrides+=("+task.shape_meta.obs.${key}.shape=[${ndf_point_num},$((3 + ndf_feat_dim))]")
    shape_overrides+=("+task.shape_meta.obs.${key}.type=point_cloud")
}

add_support_branch() {
    local support_placeholder=$1
    local support_suffix=$2
    local support_ckpt=$3

    if ! has_placeholder "${support_placeholder}"; then
        return
    fi
    if { [ "${support_ckpt}" = "none" ] || [ -z "${support_ckpt}" ]; } && ! has_ndf_feature_placeholder "${support_placeholder}"; then
        return
    fi

    add_point_cloud_shape "ndf_point_cloud_${support_suffix}"
    if [ "${support_placeholder}" != "{A}" ] && has_placeholder "{A}"; then
        add_point_cloud_shape "ndf_relation_point_cloud_A_in_${support_suffix}"
    fi
    if [ "${support_placeholder}" != "{B}" ] && has_placeholder "{B}"; then
        add_point_cloud_shape "ndf_relation_point_cloud_B_in_${support_suffix}"
    fi
}

DEBUG=False
save_ckpt=True
train_setting="${task_config}${output_suffix}"
exp_name="${task_name}-robot_dp3_ndf_relation_v2-train_ndf"
run_dir="data/outputs/${exp_name}_seed${seed}"
zarr_path="../../../data/${task_name}-${task_config}-${expert_data_num}${output_suffix}.zarr"

dataset_extra_keys=()
shape_overrides=("task.shape_meta.obs.point_cloud.shape=[${point_cloud_num},6]")
add_support_branch "{A}" "A" "${ndf_ckpt_A}"
add_support_branch "{B}" "B" "${ndf_ckpt_B}"

dataset_override=()
if [ ${#dataset_extra_keys[@]} -gt 0 ]; then
    joined_keys=$(IFS=,; echo "${dataset_extra_keys[*]}")
    dataset_override=("+task.dataset.extra_obs_keys=[${joined_keys}]")
fi

cd 3D-Diffusion-Policy

python train_dp3.py --config-name=robot_dp3_ndf_relation_hybrid.yaml \
    task_name=${task_name} \
    hydra.run.dir=${run_dir} \
    training.debug=${DEBUG} \
    training.seed=${seed} \
    training.device="cuda:0" \
    training.use_ema=${use_ema} \
    training.gradient_accumulate_every=${gradient_accumulate_every} \
    training.num_epochs=${num_epochs} \
    training.checkpoint_every=${checkpoint_every} \
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
