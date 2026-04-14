#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/ndf_pointwise_arg_utils.sh"

if [[ ${3:-} =~ ^[0-9]+$ ]]; then
    ckpt_config=${2}
    normalize_ndf_eval_hybrid_args "$@"
    actorseg_camera_names=${13:-head_camera,front_camera}
    if [ "${ndf_legacy_shifted}" = true ]; then
        actorseg_camera_names=${12:-head_camera,front_camera}
    fi
else
    eval_task_config=${2}
    ckpt_config=${3}
    shifted_args=("${1}" "${2}" "${4}" "${5}" "${6}" "${7}" "${8}" "${9}" "${10:-}" "${11:-}" "${12:-}" "${13:-}")
    normalize_ndf_eval_hybrid_args "${shifted_args[@]}"
    task_config="${eval_task_config}"
    actorseg_camera_names=${14:-head_camera,front_camera}
    if [ "${ndf_legacy_shifted}" = true ]; then
        actorseg_camera_names=${13:-head_camera,front_camera}
    fi
fi

if [ "${ndf_legacy_shifted}" = true ]; then
    echo "[eval_ndf_pointwise_actorseg_hybrid.sh] detected legacy invocation without explicit ndf_dgcnn_placeholders; treating '${object_placeholders}' as object_placeholders." >&2
fi

policy_name=DP3
ckpt_setting=${ckpt_config}-objpc-actorseg-ndf-pointwise-hybrid

extra_overrides=()
if [ "${ndf_ckpt_A}" != "none" ] && [ -n "${ndf_ckpt_A}" ]; then
    extra_overrides+=(--ndf_ckpt_A "${ndf_ckpt_A}")
fi
if [ "${ndf_ckpt_B}" != "none" ] && [ -n "${ndf_ckpt_B}" ]; then
    extra_overrides+=(--ndf_ckpt_B "${ndf_ckpt_B}")
fi
if [ -n "${ndf_device}" ]; then
    extra_overrides+=(--ndf_device "${ndf_device}")
fi
if [ -n "${ndf_dgcnn_placeholders}" ]; then
    extra_overrides+=(--ndf_dgcnn_placeholders "${ndf_dgcnn_placeholders}")
fi

export CUDA_VISIBLE_DEVICES=${gpu_id}
export HYDRA_FULL_ERROR=1
echo -e "\033[33mgpu id (to use): ${gpu_id}\033[0m"

cd "${SCRIPT_DIR}/../.."

PYTHONWARNINGS=ignore::UserWarning \
python script/eval_policy.py --config policy/${policy_name}/deploy_policy.yml \
    --overrides \
    --policy_name "${policy_name}" \
    --task_name "${task_name}" \
    --task_config "${task_config}" \
    --ckpt_setting "${ckpt_setting}" \
    --expert_data_num "${expert_data_num}" \
    --seed "${seed}" \
    --config_name robot_dp3_objpc_actorseg_ndf_pointwise_hybrid \
    --object_placeholders "${object_placeholders}" \
    --checkpoint_num "${checkpoint_num}" \
    --ndf_point_num "${ndf_point_num}" \
    --actorseg_camera_names "${actorseg_camera_names}" \
    "${extra_overrides[@]}"
