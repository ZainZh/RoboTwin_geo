#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

task_name=${1}
task_config=${2}
expert_data_num=${3}
seed=${4}
gpu_id=${5}
object_placeholders=${6:-\{A\},\{B\}}
sam3_model=${7:-/home/zheng/Datasets/sam3/sam3.pt}
camera_names=${8:-head_camera,front_camera}
sam3_prompt_map=${9:-}
text_refresh_every=${10:-15}

zarr_dir="./data/${task_name}-${task_config}-${expert_data_num}-objpc-sam3.zarr"

is_complete_zarr() {
    local root="$1"
    [ -d "${root}" ] && \
    [ -e "${root}/data/point_cloud" ] && \
    [ -e "${root}/data/state" ] && \
    [ -e "${root}/data/action" ] && \
    [ -e "${root}/meta/episode_ends" ]
}

if ! is_complete_zarr "${zarr_dir}"; then
    if [ -d "${zarr_dir}" ]; then
        echo "Found incomplete SAM3 zarr at ${zarr_dir}. Resuming preprocessing output."
    fi
    bash "${SCRIPT_DIR}/process_data_objpc_sam3.sh" \
        "${task_name}" \
        "${task_config}" \
        "${expert_data_num}" \
        "${object_placeholders}" \
        "${sam3_model}" \
        "${camera_names}" \
        "${sam3_prompt_map}" \
        "${text_refresh_every}"
fi

DEBUG=False
save_ckpt=True
wandb_mode=online
train_setting="${task_config}-objpc-sam3"
exp_name="${task_name}-robot_dp3_objpc-train_objpc_sam3"
run_dir="data/outputs/${exp_name}_seed${seed}"
zarr_path="../../../data/${task_name}-${task_config}-${expert_data_num}-objpc-sam3.zarr"

cd "${SCRIPT_DIR}/3D-Diffusion-Policy"

export HYDRA_FULL_ERROR=1
export CUDA_VISIBLE_DEVICES=${gpu_id}
python train_dp3.py --config-name=robot_dp3_objpc_sam3.yaml \
    task_name=${task_name} \
    hydra.run.dir=${run_dir} \
    training.debug=${DEBUG} \
    training.seed=${seed} \
    training.device="cuda:0" \
    exp_name=${exp_name} \
    logging.mode=${wandb_mode} \
    checkpoint.save_ckpt=${save_ckpt} \
    expert_data_num=${expert_data_num} \
    setting=${train_setting} \
    task.dataset.zarr_path=${zarr_path}
