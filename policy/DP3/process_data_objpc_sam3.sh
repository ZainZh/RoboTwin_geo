#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

task_name=${1}
task_config=${2}
expert_data_num=${3}
object_placeholders=${4:-\{A\},\{B\}}
sam3_model=${5:-/home/zheng/Datasets/sam3/sam3.pt}
camera_names=${6:-head_camera,front_camera}
sam3_prompt_map=${7:-}
text_refresh_every=${8:-15}

extra_args=()
if [ -n "${sam3_prompt_map}" ]; then
    extra_args+=(--sam3_prompt_map "${sam3_prompt_map}")
fi

python "${SCRIPT_DIR}/scripts/process_data_objpc_sam3.py" \
    "${task_name}" \
    "${task_config}" \
    "${expert_data_num}" \
    --object_placeholders "${object_placeholders}" \
    --sam3_model "${sam3_model}" \
    --camera_names "${camera_names}" \
    --text_refresh_every "${text_refresh_every}" \
    --save_placeholder_point_clouds \
    "${extra_args[@]}"
