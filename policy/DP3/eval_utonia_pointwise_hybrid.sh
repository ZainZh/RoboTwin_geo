#!/bin/bash

policy_name=DP3
task_name=${1}
task_config=${2}
ckpt_setting_base=${3}
expert_data_num=${4}
seed=${5}
gpu_id=${6}
utonia_device=${7:-cuda:0}
object_placeholders=${8:-\{A\},\{B\}}
utonia_feature_placeholders=${9:-\{A\}}
checkpoint_num=${10:-3000}
utonia_point_num=${11:-128}
point_cloud_num=${12:-1024}
point_cloud_suffix=""
if [ "${point_cloud_num}" != "1024" ]; then
    point_cloud_suffix="-pc${point_cloud_num}"
fi
output_suffix="-objpc-utonia-pointwise-hybrid${point_cloud_suffix}"
ckpt_setting=${ckpt_setting_base}${output_suffix}

meta_path="./data/${task_name}-${task_config}-${expert_data_num}${output_suffix}_meta.json"

if [ -f "${meta_path}" ]; then
    mapfile -t utonia_meta < <(python -c 'import json,sys; m=json.load(open(sys.argv[1], "r", encoding="utf-8")); print(int(m["utonia_num_points"]))' "${meta_path}")
    if [ ${#utonia_meta[@]} -ge 1 ]; then
        utonia_point_num=${utonia_meta[0]}
    fi
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
    --config_name robot_dp3_utonia_pointwise_hybrid \
    --object_placeholders "${object_placeholders}" \
    --utonia_feature_placeholders "${utonia_feature_placeholders}" \
    --checkpoint_num ${checkpoint_num} \
    --utonia_point_num ${utonia_point_num} \
    --point_cloud_num ${point_cloud_num} \
    --utonia_device "${utonia_device}"
