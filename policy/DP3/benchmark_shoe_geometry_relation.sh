#!/bin/bash
set -euo pipefail

task_name=${1}
task_config=${2}
expert_data_num=${3}
estimator_spec=${4}
gpu_id=${5:-0}
output=${6:-../../outputs/shoe_geometry_relation/benchmark.json}
include_object_ids=${7:-}

estimator_spec=$(realpath "${estimator_spec}")
output_parent=$(dirname "${output}")
mkdir -p "${output_parent}"
output=$(realpath "${output_parent}")/$(basename "${output}")

python scripts/check_shoe_se3_dependencies.py --route ndf

export CUDA_VISIBLE_DEVICES=${gpu_id}
extra_args=()
if [ -n "${include_object_ids}" ]; then
    extra_args+=(--include_shoe_ids "${include_object_ids}")
fi

python scripts/benchmark_shoe_geometry_relation.py \
    "${task_name}" \
    "${task_config}" \
    "${expert_data_num}" \
    --estimator_spec "${estimator_spec}" \
    --output "${output}" \
    --device cuda:0 \
    "${extra_args[@]}"
