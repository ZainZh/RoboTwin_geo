#!/bin/bash

task_name=${1}
task_config=${2}
expert_data_num=${3}
object_placeholders=${4:-\{A\},\{B\}}
target_num_points=${5:-1024}

python scripts/process_data_objpc.py \
    "${task_name}" \
    "${task_config}" \
    "${expert_data_num}" \
    --object_placeholders "${object_placeholders}" \
    --target_num_points "${target_num_points}"
