#!/bin/bash

task_name=${1}
task_config=${2}
expert_data_num=${3}
utonia_device=${4:-cuda:0}
object_placeholders=${5:-\{A\},\{B\}}
utonia_feature_placeholders=${6:-\{A\},\{B\}}
utonia_point_num=${7:-128}

python scripts/process_data_utonia_pointwise_hybrid.py \
    "${task_name}" \
    "${task_config}" \
    "${expert_data_num}" \
    --object_placeholders "${object_placeholders}" \
    --utonia_feature_placeholders "${utonia_feature_placeholders}" \
    --utonia_device "${utonia_device}" \
    --utonia_num_points "${utonia_point_num}"
