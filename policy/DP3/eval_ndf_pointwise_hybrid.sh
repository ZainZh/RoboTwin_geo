#!/bin/bash

policy_name=DP3
task_name=${1}
task_config=${2}
ckpt_setting=${3}-objpc-ndf-pointwise-hybrid
expert_data_num=${4}
seed=${5}
gpu_id=${6}
ndf_ckpt_A=${7:-none}
ndf_ckpt_B=${8:-none}
ndf_device=${9:-cuda:0}
ndf_dgcnn_placeholders=${10:-}
object_placeholders=${11:-\{A\},\{B\}}
checkpoint_num=${12:-3000}
ndf_point_num=${13:-128}

extra_overrides=()
if [ "${ndf_ckpt_A}" != "none" ] && [ -n "${ndf_ckpt_A}" ]; then
    extra_overrides+=(--ndf_ckpt_A "${ndf_ckpt_A}")
fi
if [ "${ndf_ckpt_B}" != "none" ] && [ -n "${ndf_ckpt_B}" ]; then
    extra_overrides+=(--ndf_ckpt_B "${ndf_ckpt_B}")
fi
if [ -n "${ndf_device}" ]; then
    extra_overrides+=(--ndf_device "${ndf_device}")
fi
if [ -n "${ndf_dgcnn_placeholders}" ]; then
    extra_overrides+=(--ndf_dgcnn_placeholders "${ndf_dgcnn_placeholders}")
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
    --config_name robot_dp3_ndf_pointwise_hybrid \
    --object_placeholders "${object_placeholders}" \
    --checkpoint_num ${checkpoint_num} \
    --ndf_point_num ${ndf_point_num} \
    "${extra_overrides[@]}"
