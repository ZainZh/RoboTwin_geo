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
projection_dim=${27:-16}
projection_seed=${28:-0}
gate_near_ratio=${29:-0.35}
gate_far_ratio=${30:-0.75}
valid_query_radius=${31:-0.75}
valid_fraction_min=${32:-0.25}
valid_fraction_max=${33:-0.65}

point_cloud_suffix=""
if [ "${point_cloud_num}" != "1024" ]; then
    point_cloud_suffix="-pc${point_cloud_num}"
fi
output_suffix="-objpc-ndf-relation-token-v2${point_cloud_suffix}"
zarr_dir="./data/${task_name}-${task_config}-${expert_data_num}${output_suffix}.zarr"
meta_path="./data/${task_name}-${task_config}-${expert_data_num}${output_suffix}_meta.json"

export CUDA_VISIBLE_DEVICES=${gpu_id}
export HYDRA_FULL_ERROR=1

if [ ! -d "${zarr_dir}" ]; then
    bash process_data_ndf_relation_token_v2.sh \
        "${task_name}" "${task_config}" "${expert_data_num}" \
        "${ndf_ckpt_A}" "${ndf_ckpt_B}" "${ndf_device}" \
        "${ndf_dgcnn_placeholders}" "${object_placeholders}" \
        "${ndf_point_num}" "${point_cloud_num}" "${output_suffix}" "${ndf_feat_dim}" \
        "${projection_dim}" "${projection_seed}" "${gate_near_ratio}" "${gate_far_ratio}" \
        "${valid_query_radius}" "${valid_fraction_min}" "${valid_fraction_max}"
fi

python scripts/validate_ndf_relation_token_v2_meta.py "${meta_path}"
mapfile -t ndf_meta < <(python -c 'import json,sys; m=json.load(open(sys.argv[1], "r", encoding="utf-8")); print(int(m["ndf_num_points"])); print(int(m.get("ndf_feat_dim", 256))); print(",".join(m.get("feature_placeholders", []))); print(int(m["relation_token_dim"])); print(int(m["relation_token_projection_dim"])); print(int(m["relation_token_projection_seed"])); print(m["relation_token_gate_near_ratio"]); print(m["relation_token_gate_far_ratio"]); print(m["relation_token_valid_query_radius"]); print(m["relation_token_valid_fraction_min"]); print(m["relation_token_valid_fraction_max"])' "${meta_path}")
ndf_point_num=${ndf_meta[0]}
ndf_feat_dim=${ndf_meta[1]}
ndf_feature_placeholders=${ndf_meta[2]}
relation_token_dim=${ndf_meta[3]}
projection_dim=${ndf_meta[4]}
projection_seed=${ndf_meta[5]}
gate_near_ratio=${ndf_meta[6]}
gate_far_ratio=${ndf_meta[7]}
valid_query_radius=${ndf_meta[8]}
valid_fraction_min=${ndf_meta[9]}
valid_fraction_max=${ndf_meta[10]}

has_placeholder() {
    local placeholder=$1
    [[ ",${object_placeholders}," == *",${placeholder},"* ]]
}

has_feature_placeholder() {
    local placeholder=$1
    [[ ",${ndf_feature_placeholders}," == *",${placeholder},"* ]]
}

add_point_cloud_shape() {
    local key=$1
    dataset_extra_keys+=("${key}")
    shape_overrides+=("+task.shape_meta.obs.${key}.shape=[${ndf_point_num},$((3 + ndf_feat_dim))]")
    shape_overrides+=("+task.shape_meta.obs.${key}.type=point_cloud")
}

add_relation_token_shape() {
    local key=$1
    dataset_extra_keys+=("${key}")
    shape_overrides+=("+task.shape_meta.obs.${key}.shape=[${relation_token_dim}]")
    shape_overrides+=("+task.shape_meta.obs.${key}.type=low_dim")
}

dataset_extra_keys=()
shape_overrides=("task.shape_meta.obs.point_cloud.shape=[${point_cloud_num},6]")
if has_feature_placeholder "{A}"; then
    add_point_cloud_shape "ndf_point_cloud_A"
    if has_placeholder "{B}"; then
        add_relation_token_shape "ndf_relation_token_B_in_A"
    fi
fi
if has_feature_placeholder "{B}"; then
    add_point_cloud_shape "ndf_point_cloud_B"
    if has_placeholder "{A}"; then
        add_relation_token_shape "ndf_relation_token_A_in_B"
    fi
fi

dataset_override=()
if [ ${#dataset_extra_keys[@]} -gt 0 ]; then
    joined_keys=$(IFS=,; echo "${dataset_extra_keys[*]}")
    dataset_override=("+task.dataset.extra_obs_keys=[${joined_keys}]")
fi

DEBUG=False
save_ckpt=True
train_setting="${task_config}${output_suffix}"
exp_name="${task_name}-robot_dp3_ndf_relation_token_v2-train"
run_dir="data/outputs/${exp_name}_seed${seed}"
zarr_path="../../../data/${task_name}-${task_config}-${expert_data_num}${output_suffix}.zarr"

cd 3D-Diffusion-Policy
python train_dp3.py --config-name=robot_dp3_ndf_pointwise_hybrid.yaml \
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
