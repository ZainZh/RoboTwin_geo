#!/bin/bash
set -euo pipefail

DEFAULT_OBJECT_PLACEHOLDERS="{A},{B}"
policy_name=DP3
task_name=${1}
task_config=${2}
ckpt_setting_base=${3}
expert_data_num=${4}
seed=${5}
gpu_id=${6}
ndf_ckpt_A=${7:-none}
ndf_ckpt_B=${8:-none}
ndf_device=${9:-cuda:0}
object_placeholders=${10:-${DEFAULT_OBJECT_PLACEHOLDERS}}
checkpoint_num=${11:-3000}
ndf_point_num=${12:-128}
point_cloud_num=${13:-1024}
ndf_dgcnn_placeholders=${14:-}
test_num=${15:-100}
gate_near_ratio=${16:-0.35}
gate_far_ratio=${17:-0.75}
valid_query_radius=${18:-0.75}
valid_fraction_min=${19:-0.25}
valid_fraction_max=${20:-0.65}

point_cloud_suffix=""
if [ "${point_cloud_num}" != "1024" ]; then
    point_cloud_suffix="-pc${point_cloud_num}"
fi
ckpt_setting=${ckpt_setting_base}-objpc-ndf-relation-token-v1${point_cloud_suffix}

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
    --point_cloud_num ${point_cloud_num} \
    --use_ndf_relation_token true \
    --ndf_relation_token_gate_near_ratio ${gate_near_ratio} \
    --ndf_relation_token_gate_far_ratio ${gate_far_ratio} \
    --ndf_relation_token_valid_query_radius ${valid_query_radius} \
    --ndf_relation_token_valid_fraction_min ${valid_fraction_min} \
    --ndf_relation_token_valid_fraction_max ${valid_fraction_max} \
    --test_num ${test_num} \
    "${extra_overrides[@]}"
