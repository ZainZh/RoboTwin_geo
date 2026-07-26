#!/bin/bash
set -euo pipefail

task_name=${1}
task_config=${2}
expert_data_num=${3}
seed=${4}
gpu_id=${5}
output_dir=${6:-../../outputs/shoe_geometry_relation}

mkdir -p "${output_dir}"
output_dir=$(realpath "${output_dir}")
constant_spec="${output_dir}/constant_goal.json"

python scripts/build_constant_goal_estimator.py \
    "${task_name}" \
    "${task_config}" \
    "${expert_data_num}" \
    --output "${constant_spec}"

bash train_shoe_se3_placement_comparison.sh \
    "${task_name}" \
    "${task_config}" \
    "${expert_data_num}" \
    "${seed}" \
    "${gpu_id}" \
    constant_goal \
    "${constant_spec}"

echo "Constant-goal estimator spec: ${constant_spec}"
