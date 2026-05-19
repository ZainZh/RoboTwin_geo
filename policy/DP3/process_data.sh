#!/bin/bash

task_name=${1}
task_config=${2}
expert_data_num=${3}
target_num_points=${4:-1024}
output_suffix=${5:-}

python scripts/process_data.py \
    "${task_name}" \
    "${task_config}" \
    "${expert_data_num}" \
    --target_num_points "${target_num_points}" \
    --output_suffix="${output_suffix}"
