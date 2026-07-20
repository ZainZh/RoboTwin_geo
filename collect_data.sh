#!/bin/bash

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/script/data_paths.sh"

task_name=${1}
task_config=${2}
gpu_id=${3}


export CUDA_VISIBLE_DEVICES=${gpu_id}

PYTHONWARNINGS=ignore::UserWarning \
python script/collect_data.py $task_name $task_config
rm -rf "${ROBOTWIN_RAW_DATA_ROOT}/${task_name}/${task_config}/.cache"
