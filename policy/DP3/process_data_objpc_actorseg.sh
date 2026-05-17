#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

task_name=${1}
task_config=${2}
expert_data_num=${3}
object_placeholders=${4:-\{A\},\{B\}}
camera_names=${5:-head_camera,front_camera}
target_num_points=${6:-1024}
output_suffix=${7:--objpc-actorseg}

cd "${SCRIPT_DIR}"

python scripts/process_data_objpc_actorseg.py \
    "${task_name}" \
    "${task_config}" \
    "${expert_data_num}" \
    --object_placeholders "${object_placeholders}" \
    --camera_names "${camera_names}" \
    --target_num_points "${target_num_points}" \
    --output_suffix="${output_suffix}" \
    --save_placeholder_point_clouds
