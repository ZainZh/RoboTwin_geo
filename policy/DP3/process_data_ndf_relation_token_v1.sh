#!/bin/bash
set -euo pipefail

DEFAULT_OBJECT_PLACEHOLDERS="{A},{B}"
task_name=${1}
task_config=${2}
expert_data_num=${3}
ndf_ckpt_A=${4:-none}
ndf_ckpt_B=${5:-none}
ndf_device=${6:-cuda:0}
ndf_dgcnn_placeholders=${7:-}
object_placeholders=${8:-${DEFAULT_OBJECT_PLACEHOLDERS}}
ndf_point_num=${9:-128}
target_num_points=${10:-1024}
output_suffix=${11:--objpc-ndf-relation-token-v1}
ndf_feat_dim=${12:-256}
gate_near_ratio=${13:-0.35}
gate_far_ratio=${14:-0.75}
valid_query_radius=${15:-0.75}
valid_fraction_min=${16:-0.25}
valid_fraction_max=${17:-0.65}

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

python scripts/process_data_ndf_relation_token_v1.py \
    "${task_name}" \
    "${task_config}" \
    "${expert_data_num}" \
    --object_placeholders "${object_placeholders}" \
    --ndf_device "${ndf_device}" \
    --ndf_feat_dim "${ndf_feat_dim}" \
    --ndf_num_points "${ndf_point_num}" \
    --target_num_points "${target_num_points}" \
    --output_suffix="${output_suffix}" \
    --relation_token_gate_near_ratio "${gate_near_ratio}" \
    --relation_token_gate_far_ratio "${gate_far_ratio}" \
    --relation_token_valid_query_radius "${valid_query_radius}" \
    --relation_token_valid_fraction_min "${valid_fraction_min}" \
    --relation_token_valid_fraction_max "${valid_fraction_max}" \
    "${extra_args[@]}"
