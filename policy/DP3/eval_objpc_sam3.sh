#!/bin/bash

policy_name=DP3
task_name=${1}
task_config=${2}

# 1. old: bash eval_objpc_sam3.sh <task> <task_config> <expert_data_num> <seed> <gpu_id> [object_placeholders] [checkpoint_num]
# 2. new: bash eval_objpc_sam3.sh <task> <eval_task_config> <ckpt_task_config> <expert_data_num> <seed> <gpu_id> [object_placeholders] [checkpoint_num]
if [[ ${3:-} =~ ^[0-9]+$ ]]; then
    ckpt_config=${task_config}
    expert_data_num=${3}
    seed=${4}
    gpu_id=${5}
    object_placeholders=${6:-\{A\},\{B\}}
    checkpoint_num=${7:-3000}
    sam3_prompt_map=${8:-}
    sam3_model=${9:-/home/zheng/Datasets/sam3/sam3.pt}
    sam3_camera_names=${10:-head_camera,front_camera}
    sam3_text_refresh_every=${11:-15}
else
    ckpt_config=${3:-${task_config}}
    expert_data_num=${4}
    seed=${5}
    gpu_id=${6}
    object_placeholders=${7:-\{A\},\{B\}}
    checkpoint_num=${8:-3000}
    sam3_prompt_map=${9:-}
    sam3_model=${10:-/home/zheng/Datasets/sam3/sam3.pt}
    sam3_camera_names=${11:-head_camera,front_camera}
    sam3_text_refresh_every=${12:-15}
fi

ckpt_setting=${ckpt_config}-objpc-sam3

extra_overrides=(
    --sam3_model "${sam3_model}"
    --sam3_camera_names "${sam3_camera_names}"
    --sam3_text_refresh_every "${sam3_text_refresh_every}"
)
if [ -n "${sam3_prompt_map}" ]; then
    extra_overrides+=(--sam3_prompt_map "${sam3_prompt_map}")
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
    --config_name robot_dp3_objpc_sam3 \
    --object_placeholders "${object_placeholders}" \
    --checkpoint_num "${checkpoint_num}" \
    "${extra_overrides[@]}"
