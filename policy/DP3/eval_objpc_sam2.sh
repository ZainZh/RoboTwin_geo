#!/bin/bash

policy_name=DP3
task_name=${1}
task_config=${2}
ckpt_setting_base=${3}
expert_data_num=${4}
seed=${5}
gpu_id=${6}
object_placeholders=${7:-\{A\},\{B\}}
checkpoint_num=${8:-3000}
sam2_bbox_prompt_path=${9:-}
sam2_camera_names=${10:-head_camera,front_camera}
sam2_checkpoint=${11:-${SAM2_CHECKPOINT:-${HOME}/Datasets/sam2/sam2.1_hiera_large.pt}}
sam2_device=${12:-cuda:0}
sam2_config=${13:-sam2.1/sam2.1_hiera_l.yaml}
sam2_interactive_init=${14:-True}
sam2_min_mask_points=${15:-16}
sam2_autocast_dtype=${16:-bfloat16}
sam2_root=${17:-${SAM2_STREAMING_ROOT:-}}
point_cloud_num=${18:-1024}
point_cloud_suffix=""
if [ "${point_cloud_num}" != "1024" ]; then
    point_cloud_suffix="-pc${point_cloud_num}"
fi
ckpt_setting="${ckpt_setting_base}${point_cloud_suffix}"

extra_overrides=(
    --sam2_camera_names "${sam2_camera_names}"
    --sam2_checkpoint "${sam2_checkpoint}"
    --sam2_device "${sam2_device}"
    --sam2_config "${sam2_config}"
    --sam2_interactive_init "${sam2_interactive_init}"
    --sam2_min_mask_points "${sam2_min_mask_points}"
    --sam2_autocast_dtype "${sam2_autocast_dtype}"
)
if [ -n "${sam2_bbox_prompt_path}" ]; then
    extra_overrides+=(--sam2_bbox_prompt_path "${sam2_bbox_prompt_path}")
fi
if [ -n "${sam2_root}" ]; then
    extra_overrides+=(--sam2_root "${sam2_root}")
fi

export CUDA_VISIBLE_DEVICES=${gpu_id}
export HYDRA_FULL_ERROR=1
echo -e "\033[33mgpu id (to use): ${gpu_id}\033[0m"

cd ../..

PYTHONWARNINGS=ignore::UserWarning \
python script/eval_policy.py --config policy/${policy_name}/deploy_policy.yml \
    --overrides \
    --policy_name "${policy_name}" \
    --task_name "${task_name}" \
    --task_config "${task_config}" \
    --ckpt_setting "${ckpt_setting}" \
    --expert_data_num "${expert_data_num}" \
    --seed "${seed}" \
    --config_name robot_dp3_objpc_sam2 \
    --object_placeholders "${object_placeholders}" \
    --checkpoint_num "${checkpoint_num}" \
    --point_cloud_num "${point_cloud_num}" \
    "${extra_overrides[@]}"
