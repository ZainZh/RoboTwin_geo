#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/ndf_pointwise_arg_utils.sh"
normalize_ndf_process_actorseg_args "$@"

if [ "${ndf_legacy_shifted}" = true ]; then
    echo "[process_data_ndf_pointwise_actorseg_hybrid.sh] detected legacy invocation without explicit ndf_dgcnn_placeholders; treating '${object_placeholders}' as object_placeholders." >&2
fi

cd "${SCRIPT_DIR}"

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

python scripts/process_data_ndf_pointwise_actorseg_hybrid.py \
    "${task_name}" \
    "${task_config}" \
    "${expert_data_num}" \
    --object_placeholders "${object_placeholders}" \
    --camera_names "${actorseg_camera_names}" \
    --ndf_device "${ndf_device}" \
    --ndf_num_points "${ndf_point_num}" \
    --target_num_points "${target_num_points}" \
    --output_suffix="${output_suffix}" \
    "${extra_args[@]}"
