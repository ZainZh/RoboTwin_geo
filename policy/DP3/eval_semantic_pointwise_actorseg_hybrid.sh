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
actorseg_camera_names=${13:-head_camera,front_camera}
point_cloud_num=${14:-1024}
semantic_input_color_mode=${15:-debug_placeholder}
semantic_forward_mode=${16:-reference}
point_cloud_suffix=""
if [ "${point_cloud_num}" != "1024" ]; then
    point_cloud_suffix="-pc${point_cloud_num}"
fi
semantic_feature_suffix="-sem${semantic_input_color_mode}-${semantic_forward_mode}"
if [ "${semantic_input_color_mode}" = "debug_placeholder" ] && [ "${semantic_forward_mode}" = "reference" ]; then
    semantic_feature_suffix="-semdebugref"
fi
ckpt_setting="${ckpt_setting_base}${semantic_feature_suffix}${point_cloud_suffix}"

meta_path="./data/${task_name}-${task_config}-${expert_data_num}-objpc-actorseg-semantic-pointwise-hybrid${semantic_feature_suffix}${point_cloud_suffix}_meta.json"

if [ -f "${meta_path}" ]; then
    mapfile -t semantic_meta < <(python -c 'import json,sys; m=json.load(open(sys.argv[1], "r", encoding="utf-8")); print(int(m["semantic_num_points"])); print(str(m.get("semantic_input_color_mode", "debug_placeholder"))); print(str(m.get("semantic_forward_mode", "reference")))' "${meta_path}")
    if [ ${#semantic_meta[@]} -ge 1 ]; then
        semantic_point_num=${semantic_meta[0]}
    fi
    if [ ${#semantic_meta[@]} -ge 3 ]; then
        semantic_input_color_mode=${semantic_meta[1]}
        semantic_forward_mode=${semantic_meta[2]}
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
    --config_name robot_dp3_objpc_actorseg_semantic_pointwise_hybrid \
    --object_placeholders "${object_placeholders}" \
    --checkpoint_num ${checkpoint_num} \
    --semantic_point_num ${semantic_point_num} \
    --semantic_input_color_mode "${semantic_input_color_mode}" \
    --semantic_forward_mode "${semantic_forward_mode}" \
    --actorseg_camera_names "${actorseg_camera_names}" \
    --point_cloud_num ${point_cloud_num} \
    "${extra_overrides[@]}"
