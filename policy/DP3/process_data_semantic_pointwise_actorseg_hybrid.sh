#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

DEFAULT_OBJECT_PLACEHOLDERS="{A},{B}"

task_name=${1}
task_config=${2}
expert_data_num=${3}
semantic_ckpt_A=${4:-none}
semantic_ckpt_B=${5:-none}
semantic_device=${6:-cuda:0}
object_placeholders=${7:-${DEFAULT_OBJECT_PLACEHOLDERS}}
semantic_point_num=${8:-128}
actorseg_camera_names=${9:-head_camera,front_camera}
target_num_points=${10:-1024}
output_suffix=${11:--objpc-actorseg-semantic-pointwise-hybrid}

extra_args=()
if [ "${semantic_ckpt_A}" != "none" ] && [ -n "${semantic_ckpt_A}" ]; then
    extra_args+=(--semantic_model "{A}=${semantic_ckpt_A}")
fi
if [ "${semantic_ckpt_B}" != "none" ] && [ -n "${semantic_ckpt_B}" ]; then
    extra_args+=(--semantic_model "{B}=${semantic_ckpt_B}")
fi

python scripts/process_data_semantic_pointwise_actorseg_hybrid.py \
    "${task_name}" \
    "${task_config}" \
    "${expert_data_num}" \
    --object_placeholders "${object_placeholders}" \
    --camera_names "${actorseg_camera_names}" \
    --semantic_device "${semantic_device}" \
    --semantic_num_points "${semantic_point_num}" \
    --target_num_points "${target_num_points}" \
    --output_suffix="${output_suffix}" \
    "${extra_args[@]}"
