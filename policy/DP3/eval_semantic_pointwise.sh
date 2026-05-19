#!/bin/bash

policy_name=DP3
task_name=${1}
task_config=${2}
ckpt_setting_base=${3}
expert_data_num=${4}
seed=${5}
gpu_id=${6}
semantic_ckpt_A=${7:-none}
semantic_ckpt_B=${8:-none}
semantic_device=${9:-cuda:0}
object_placeholders=${10:-\{A\},\{B\}}
checkpoint_num=${11:-3000}
semantic_point_num=${12:-128}
point_cloud_num=${13:-1024}
point_cloud_suffix=""
if [ "${point_cloud_num}" != "1024" ]; then
    point_cloud_suffix="-pc${point_cloud_num}"
fi
ckpt_setting="${ckpt_setting_base}${point_cloud_suffix}"

meta_path="./data/${task_name}-${task_config}-${expert_data_num}-objpc-semantic-pointwise${point_cloud_suffix}_meta.json"

if [ -f "${meta_path}" ]; then
    mapfile -t semantic_meta < <(python -c 'import json,sys; m=json.load(open(sys.argv[1], "r", encoding="utf-8")); print(int(m["semantic_num_points"]))' "${meta_path}")
    if [ ${#semantic_meta[@]} -ge 1 ]; then
        semantic_point_num=${semantic_meta[0]}
    fi
fi

extra_overrides=()
if [ "${semantic_ckpt_A}" != "none" ] && [ -n "${semantic_ckpt_A}" ]; then
    extra_overrides+=(--semantic_ckpt_A "${semantic_ckpt_A}")
fi
if [ "${semantic_ckpt_B}" != "none" ] && [ -n "${semantic_ckpt_B}" ]; then
    extra_overrides+=(--semantic_ckpt_B "${semantic_ckpt_B}")
fi
if [ -n "${semantic_device}" ]; then
    extra_overrides+=(--semantic_device "${semantic_device}")
fi

export CUDA_VISIBLE_DEVICES=${gpu_id}
export HYDRA_FULL_ERROR=1
echo -e "\033[33mgpu id (to use): ${gpu_id}\033[0m"

cd ../..

PYTHONWARNINGS=ignore::UserWarning \
python script/eval_policy.py --config policy/${policy_name}/deploy_policy.yml \
    --overrides \
    --policy_name ${policy_name} \
    --task_name ${task_name} \
    --task_config ${task_config} \
    --ckpt_setting ${ckpt_setting} \
    --expert_data_num ${expert_data_num} \
    --seed ${seed} \
    --config_name robot_dp3_semantic_pointwise \
    --object_placeholders "${object_placeholders}" \
    --checkpoint_num ${checkpoint_num} \
    --semantic_point_num ${semantic_point_num} \
    --point_cloud_num ${point_cloud_num} \
    "${extra_overrides[@]}"
