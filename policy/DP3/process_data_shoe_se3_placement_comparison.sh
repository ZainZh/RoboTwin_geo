#!/bin/bash
set -euo pipefail

task_name=${1}
task_config=${2}
expert_data_num=${3}
route=${4}
goal_table=${5:-}
object_placeholders=${6:-\{A\},\{B\}}
point_cloud_num=${7:-1024}
output_suffix=${8:-}

python scripts/check_shoe_se3_dependencies.py --route baseline

extra_args=()
if [ -n "${goal_table}" ]; then
    goal_table=$(realpath "${goal_table}")
    extra_args+=(--goal_table "${goal_table}")
fi
if [ -n "${output_suffix}" ]; then
    extra_args+=(--output_suffix="${output_suffix}")
fi

python scripts/process_data_shoe_se3_placement_comparison.py \
    "${task_name}" \
    "${task_config}" \
    "${expert_data_num}" \
    --route "${route}" \
    --object_placeholders "${object_placeholders}" \
    --target_num_points "${point_cloud_num}" \
    "${extra_args[@]}"
