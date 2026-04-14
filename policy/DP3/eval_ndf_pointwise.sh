#!/bin/bash

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${script_dir}/ndf_pointwise_arg_utils.sh"
normalize_ndf_eval_args "$@"

if [ "${ndf_legacy_shifted}" = true ]; then
    echo "[eval_ndf_pointwise.sh] detected legacy invocation without explicit ndf_dgcnn_placeholders; treating '${object_placeholders}' as object_placeholders." >&2
fi

#ckpt_setting=${task_config}-objpc-ndf-pointwise

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

cd ../..

PYTHONWARNINGS=ignore::UserWarning \
python script/eval_policy.py --config policy/${policy_name}/deploy_policy.yml \
    --overrides \
    --policy_name ${policy_name} \
    --task_name ${task_name} \
    --task_config ${task_config} \
    --ckpt_setting ${ckpt_setting} \
    --expert_data_num ${expert_data_num} \
    --seed ${seed} \
    --config_name robot_dp3_ndf_pointwise \
    --object_placeholders "${object_placeholders}" \
    --checkpoint_num ${checkpoint_num} \
    --ndf_point_num ${ndf_point_num} \
    "${extra_overrides[@]}"
