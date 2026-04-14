#!/bin/bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
policy_dir="$(cd "${script_dir}/.." && pwd)"
source "${policy_dir}/ndf_pointwise_arg_utils.sh"

normalize_ndf_train_args \
    beat_block_hammer \
    demo_clean_3d_object_pc \
    50 \
    0 \
    0 \
    /tmp/hammer.pth \
    "" \
    cuda:0 \
    "{A},{B}" \
    128

test "${ndf_dgcnn_placeholders}" = ""
test "${object_placeholders}" = "{A},{B}"
test "${ndf_point_num}" = "128"

normalize_ndf_eval_hybrid_args \
    beat_block_hammer \
    demo_clean_3d_object_pc \
    50 \
    0 \
    0 \
    /tmp/hammer.pth \
    "" \
    cuda:0 \
    "{A},{B}" \
    3000

test "${ndf_dgcnn_placeholders}" = ""
test "${object_placeholders}" = "{A},{B}"
test "${checkpoint_num}" = "3000"

normalize_ndf_process_actorseg_args \
    hanging_mug \
    demo_clean_3d_actorseg \
    50 \
    /tmp/mug.pth \
    "" \
    cuda:0 \
    "{A},{B}" \
    128 \
    head_camera,front_camera

test "${task_name}" = "hanging_mug"
test "${task_config}" = "demo_clean_3d_actorseg"
test "${expert_data_num}" = "50"
test "${ndf_ckpt_A}" = "/tmp/mug.pth"
test "${ndf_ckpt_B}" = "none"
test "${ndf_device}" = "cuda:0"
test "${ndf_dgcnn_placeholders}" = ""
test "${object_placeholders}" = "{A},{B}"
test "${ndf_point_num}" = "128"
test "${actorseg_camera_names}" = "head_camera,front_camera"
