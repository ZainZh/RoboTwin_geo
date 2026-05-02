#!/bin/bash

task_name=${1}
task_config=${2}
expert_data_num=${3}
seed=${4}
action_dim=${5:-14}
gpu_id=${6}
camera_label=${7:-left}
head_camera_type=${8:-Large_L515}
resume=${9:-true}
meta_path=${10:-}

DEBUG=False
save_ckpt=True
wandb_mode=online
train_setting="${task_config}-dp-${camera_label}"
exp_name="${task_name}-robot_dp-real_zed_train"
run_dir="data/outputs/${exp_name}_seed${seed}"
zarr_path="data/${task_name}-${train_setting}-${expert_data_num}.zarr"

echo -e "\033[33mgpu id (to use): ${gpu_id}\033[0m"
echo -e "\033[33mReal-ZED DP baseline camera: ${camera_label}, head_camera_type: ${head_camera_type}\033[0m"

export HYDRA_FULL_ERROR=1
export CUDA_VISIBLE_DEVICES=${gpu_id}

needs_preprocess=false
if [ ! -d "./${zarr_path}" ]; then
    needs_preprocess=true
else
    python - "${zarr_path}" <<'PY'
import sys
from pathlib import Path

import zarr

path = Path(sys.argv[1])
try:
    root = zarr.open(str(path), "r")
    if "head_camera" not in root["data"]:
        raise KeyError("head_camera")
except Exception:
    raise SystemExit(1)
PY
    if [ $? -ne 0 ]; then
        echo -e "\033[33mExisting zarr is incompatible with single-camera DP; rebuilding ${zarr_path}\033[0m"
        needs_preprocess=true
    fi
fi

if [ "${needs_preprocess}" = true ]; then
    bash process_data_real_zed.sh "${task_name}" "${task_config}" "${expert_data_num}" "${camera_label}" "${meta_path}"
fi

python train.py --config-name=robot_dp_${action_dim}.yaml \
    task.name=${task_name} \
    task.dataset.zarr_path="${zarr_path}" \
    training.debug=${DEBUG} \
    training.resume=${resume} \
    training.seed=${seed} \
    training.device="cuda:0" \
    exp_name=${exp_name} \
    logging.mode=${wandb_mode} \
    setting=${train_setting} \
    expert_data_num=${expert_data_num} \
    head_camera_type=${head_camera_type}
