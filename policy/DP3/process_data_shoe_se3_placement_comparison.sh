#!/bin/bash
set -euo pipefail

task_name=${1}
task_config=${2}
expert_data_num=${3}
route=${4}
# Legacy NDF routes consume a goal-table JSON.  ndf_observation_goal consumes
# a GeometryRelationEstimator spec JSON in the same positional slot.
route_artifact=${5:-}
object_placeholders=${6:-\{A\},\{B\}}
point_cloud_num=${7:-1024}
output_suffix=${8:-}

dependency_route=baseline
if [ "${route}" = "ndf_observation_goal" ]; then
    dependency_route=ndf
fi
python scripts/check_shoe_se3_dependencies.py --route "${dependency_route}"

extra_args=()
if [ -n "${route_artifact}" ]; then
    route_artifact=$(realpath "${route_artifact}")
    if [ "${route}" = "ndf_observation_goal" ]; then
        extra_args+=(--geometry_estimator_spec "${route_artifact}")
    else
        extra_args+=(--goal_table "${route_artifact}")
    fi
fi
if [ -n "${output_suffix}" ]; then
    extra_args+=(--output_suffix="${output_suffix}")
fi

python scripts/process_data_shoe_se3_placement_comparison.py \
    "${task_name}" \
    "${task_config}" \
    "${expert_data_num}" \
    --route "${route}" \
    --object_placeholders "${object_placeholders}" \
    --target_num_points "${point_cloud_num}" \
    "${extra_args[@]}"
