#!/bin/bash
set -euo pipefail

policy_name=DP3
task_name=${1}
task_config=${2}
ckpt_setting_base=${3}
expert_data_num=${4}
seed=${5}
gpu_id=${6}
route=${7:-baseline}
goal_table=${8:-}
object_placeholders=${9:-\{A\},\{B\}}
checkpoint_num=${10:-3000}
point_cloud_num=${11:-1024}
test_num=${12:-100}

dependency_route=baseline
if [ "${route}" = "ndf_no_direction" ] || [ "${route}" = "ndf_direction" ]; then
    dependency_route=ndf
fi
python scripts/check_shoe_se3_dependencies.py --route "${dependency_route}" --training

if [ "${route}" = "baseline" ]; then
    route_suffix="baseline"
else
    route_suffix="se3-relation-${route//_/-}"
fi
point_cloud_suffix=""
if [ "${point_cloud_num}" != "1024" ]; then
    point_cloud_suffix="-pc${point_cloud_num}"
fi
ckpt_setting="${ckpt_setting_base}-objpc-placement-only-${route_suffix}${point_cloud_suffix}"

route_overrides=()
if [ "${route}" != "baseline" ]; then
    route_overrides+=(--use_se3_relation_token true)
    route_overrides+=(--se3_relation_route "${route}")
    if [ -n "${goal_table}" ]; then
        goal_table=$(realpath "${goal_table}")
        route_overrides+=(--se3_relation_goal_table "${goal_table}")
    fi
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
    --config_name robot_dp3_objpc \
    --object_placeholders "${object_placeholders}" \
    --checkpoint_num "${checkpoint_num}" \
    --point_cloud_num "${point_cloud_num}" \
    --policy_start_phase placement \
    --test_num "${test_num}" \
    "${route_overrides[@]}"
