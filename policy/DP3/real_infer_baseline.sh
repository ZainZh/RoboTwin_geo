#!/bin/bash

task_name=${1:-grasp_mug}
task_config=${2:-demo_real_zed_sam2_objpc}
expert_data_num=${3:-57}
seed=${4:-0}
gpu_id=${5:-0}
checkpoint_num=${6:-3000}
extra_flags=("${@:7}")

export CUDA_VISIBLE_DEVICES=${gpu_id}
export HYDRA_FULL_ERROR=1
echo -e "\033[33mgpu id (to use): ${gpu_id}\033[0m"

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "${script_dir}/../.."

PYTHONWARNINGS=ignore::UserWarning \
python script/real_zed_inference/real_dp3_inference.py \
    --mode baseline \
    --task_name "${task_name}" \
    --task_config "${task_config}" \
    --ckpt_setting "${task_config}" \
    --expert_data_num "${expert_data_num}" \
    --seed "${seed}" \
    --gpu_id "${gpu_id}" \
    --checkpoint_num "${checkpoint_num}" \
    "${extra_flags[@]}"
