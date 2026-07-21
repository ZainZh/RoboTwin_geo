#!/bin/bash
set -euo pipefail

ndf_checkpoint=${1}
gpu_id=${2:-0}
output_dir=${3:-../../outputs/ndf_shoe_ramp_se3/comparison}
demo_shoe_id=${4:-0}
trials=${5:-3}

python scripts/check_shoe_se3_dependencies.py --route ndf

mkdir -p "${output_dir}"
validation_json="${output_dir}/validation_with_transforms.json"
goal_table_json="${output_dir}/goal_table.json"

export CUDA_VISIBLE_DEVICES=${gpu_id}
python scripts/validate_ndf_shoe_ramp_se3.py \
    --checkpoint "${ndf_checkpoint}" \
    --device cuda:0 \
    --demo_shoe_id "${demo_shoe_id}" \
    --query_shoe_ids 0,1,2,3,4,5,6,7,8,9 \
    --direction_weights 0,5 \
    --trials "${trials}" \
    --output "${validation_json}"

python scripts/build_se3_goal_table_from_validation.py \
    "${validation_json}" "${goal_table_json}" \
    --no_direction_weight 0 \
    --direction_weight 5

echo "goal table: ${goal_table_json}"
