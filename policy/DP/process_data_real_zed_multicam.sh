#!/bin/bash
set -euo pipefail

task_name=${1}
task_config=${2}
expert_data_num=${3}
camera_labels=${4:-global,left,right}
meta_path=${5:-}
resize_hw=${6:-}
extra_forward=("${@:7}")

extra_args=()
if [ -n "${meta_path}" ]; then
    extra_args+=(--meta_path "${meta_path}")
fi
if [ -n "${resize_hw}" ]; then
    extra_args+=(--resize_hw "${resize_hw}")
fi

python process_data_real_zed.py \
    "${task_name}" \
    "${task_config}" \
    "${expert_data_num}" \
    --camera_labels "${camera_labels}" \
    "${extra_args[@]}" \
    "${extra_forward[@]}"
