#!/bin/bash
set -euo pipefail

task_name=${1}
task_config=${2}
expert_data_num=${3}
ndf_checkpoint=${4}
gpu_id=${5:-0}
output_dir=${6:-../../outputs/shoe_geometry_relation}
# IDs are used only to define a strict held-out geometry benchmark split.
validation_object_ids=${7:-8,9}

ndf_checkpoint=$(realpath "${ndf_checkpoint}")
mkdir -p "${output_dir}"
output_dir=$(realpath "${output_dir}")
regressor_checkpoint="${output_dir}/ndf_goal_regressor.pt"
estimator_spec="${output_dir}/ndf_goal_regressor.json"

python scripts/check_shoe_se3_dependencies.py --route ndf

export CUDA_VISIBLE_DEVICES=${gpu_id}
extra_args=()
if [ -n "${validation_object_ids}" ]; then
    extra_args+=(--validation_object_ids "${validation_object_ids}")
fi

python scripts/train_ndf_goal_regressor.py \
    "${task_name}" \
    "${task_config}" \
    "${expert_data_num}" \
    --ndf_checkpoint "${ndf_checkpoint}" \
    --output "${regressor_checkpoint}" \
    --spec_output "${estimator_spec}" \
    --device cuda:0 \
    "${extra_args[@]}"

echo "Geometry estimator spec: ${estimator_spec}"
