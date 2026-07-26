#!/bin/bash
set -euo pipefail

task_name=${1}
task_config=${2}
ckpt_setting_base=${3}
expert_data_num=${4}
seed=${5}
gpu_id=${6}
observation_estimator_spec=${7}
ablation=${8}
constant_goal_spec=${9:-}
checkpoint_num=${10:-3000}
test_num=${11:-100}

if [ "${ablation}" != "zero" ] && [ "${ablation}" != "constant_goal" ]; then
    echo "ablation must be zero or constant_goal" >&2
    exit 2
fi
if [ "${ablation}" = "constant_goal" ] && [ -z "${constant_goal_spec}" ]; then
    echo "constant_goal_spec is required for ablation=constant_goal" >&2
    exit 2
fi

bash eval_shoe_se3_placement_comparison.sh \
    "${task_name}" \
    "${task_config}" \
    "${ckpt_setting_base}" \
    "${expert_data_num}" \
    "${seed}" \
    "${gpu_id}" \
    ndf_observation_goal \
    "${observation_estimator_spec}" \
    '{A},{B}' \
    "${checkpoint_num}" \
    1024 \
    "${test_num}" \
    "${ablation}" \
    "${constant_goal_spec}"
