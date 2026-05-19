#!/bin/bash

policy_name=DP3
task_name=${1}
task_config=${2}
ckpt_config=${3:-${task_config}}
expert_data_num=${4}
seed=${5}
gpu_id=${6}
object_placeholders=${7:-\{A\},\{B\}}
checkpoint_num=${8:-3000}
point_cloud_num=${9:-5000}
point_cloud_suffix=""
if [ "${point_cloud_num}" != "1024" ]; then
    point_cloud_suffix="-pc${point_cloud_num}"
fi

ckpt_setting=${ckpt_config}-objpc${point_cloud_suffix}

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
    --config_name robot_dp3_objpc_5000 \
    --object_placeholders "${object_placeholders}" \
    --checkpoint_num "${checkpoint_num}" \
    --point_cloud_num "${point_cloud_num}"
