#!/bin/bash

task_name=${1}
task_config=${2}
expert_data_num=${3}
ndf_ckpt_A=${4:-none}
ndf_ckpt_B=${5:-none}
ndf_device=${6:-cuda:0}
ndf_dgcnn_placeholders=${7:-}
object_placeholders=${8:-\{A\},\{B\}}
target_num_points=${9:-1024}
output_suffix=${10:--objpc-ndf}

extra_args=()
if [ "${ndf_ckpt_A}" != "none" ] && [ -n "${ndf_ckpt_A}" ]; then
    extra_args+=(--ndf_model "{A}=${ndf_ckpt_A}")
fi
if [ "${ndf_ckpt_B}" != "none" ] && [ -n "${ndf_ckpt_B}" ]; then
    extra_args+=(--ndf_model "{B}=${ndf_ckpt_B}")
fi
if [ -n "${ndf_dgcnn_placeholders}" ]; then
    extra_args+=(--ndf_dgcnn "${ndf_dgcnn_placeholders}")
fi

python scripts/process_data_ndf.py \
    "${task_name}" \
    "${task_config}" \
    "${expert_data_num}" \
    --object_placeholders "${object_placeholders}" \
    --ndf_device "${ndf_device}" \
    --target_num_points "${target_num_points}" \
    --output_suffix="${output_suffix}" \
    "${extra_args[@]}"
