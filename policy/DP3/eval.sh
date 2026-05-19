#!/bin/bash

policy_name=DP3
task_name=${1}
task_config=${2}
ckpt_setting_base=${3}
expert_data_num=${4}
seed=${5} # both policy and RoboTwin scene
gpu_id=${6}
point_cloud_num=${7:-1024}
point_cloud_suffix=""
if [ "${point_cloud_num}" != "1024" ]; then
    point_cloud_suffix="-pc${point_cloud_num}"
fi
ckpt_setting="${ckpt_setting_base}${point_cloud_suffix}"

export CUDA_VISIBLE_DEVICES=${gpu_id}
export HYDRA_FULL_ERROR=1
echo -e "\033[33mgpu id (to use): ${gpu_id}\033[0m"

cd ../.. # move to root

PYTHONWARNINGS=ignore::UserWarning \
python script/eval_policy.py --config policy/$policy_name/deploy_policy.yml \
    --overrides \
    --task_name ${task_name} \
    --task_config ${task_config} \
    --ckpt_setting ${ckpt_setting} \
    --expert_data_num ${expert_data_num} \
    --seed ${seed} \
    --policy_name ${policy_name} \
    --point_cloud_num ${point_cloud_num}
