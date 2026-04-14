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
