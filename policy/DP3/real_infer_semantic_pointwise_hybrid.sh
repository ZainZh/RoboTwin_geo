#!/bin/bash

task_name=${1:-grasp_mug}
task_config=${2:-demo_real_zed_sam2_objpc}
expert_data_num=${3:-57}
seed=${4:-0}
gpu_id=${5:-0}
semantic_ckpt_A=${6:-/home/zheng/github/3d_semantic_train/outputs/utonia_universal_field/Mug_semantic/mug.pt}
semantic_ckpt_B=${7:-none}
semantic_device=${8:-cuda:0}
object_placeholders=${9:-\{A\},\{B\}}
checkpoint_num=${10:-3000}
semantic_point_num=${11:-128}
extra_flags=("${@:12}")

export CUDA_VISIBLE_DEVICES=${gpu_id}
export HYDRA_FULL_ERROR=1
echo -e "\033[33mgpu id (to use): ${gpu_id}\033[0m"

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "${script_dir}/../.."

PYTHONWARNINGS=ignore::UserWarning \
python script/real_zed_inference/real_dp3_inference.py \
    --mode semantic_pointwise_hybrid \
    --task_name "${task_name}" \
    --task_config "${task_config}" \
    --ckpt_setting "${task_config}-objpc-semantic-pointwise-hybrid" \
    --expert_data_num "${expert_data_num}" \
    --seed "${seed}" \
    --gpu_id "${gpu_id}" \
    --checkpoint_num "${checkpoint_num}" \
    --semantic_ckpt_A "${semantic_ckpt_A}" \
    --semantic_ckpt_B "${semantic_ckpt_B}" \
    --semantic_device "${semantic_device}" \
    --semantic_point_num "${semantic_point_num}" \
    --object_placeholders "${object_placeholders}" \
    --enable_sam2_objpc \
    "${extra_flags[@]}"
