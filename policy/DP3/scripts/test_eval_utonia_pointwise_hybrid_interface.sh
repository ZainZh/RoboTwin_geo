#!/bin/bash
set -euo pipefail

script_file="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/eval_utonia_pointwise_hybrid.sh"

grep -F -- 'policy_name=DP3' "${script_file}" >/dev/null
grep -F -- 'task_name=${1}' "${script_file}" >/dev/null
grep -F -- 'task_config=${2}' "${script_file}" >/dev/null
grep -F -- 'ckpt_setting_base=${3}' "${script_file}" >/dev/null
grep -F -- 'expert_data_num=${4}' "${script_file}" >/dev/null
grep -F -- 'seed=${5}' "${script_file}" >/dev/null
grep -F -- 'gpu_id=${6}' "${script_file}" >/dev/null
grep -F -- 'utonia_device=${7:-cuda:0}' "${script_file}" >/dev/null
grep -F -- 'object_placeholders=${8:-\{A\},\{B\}}' "${script_file}" >/dev/null
grep -F -- 'utonia_feature_placeholders=${9:-\{A\}}' "${script_file}" >/dev/null
grep -F -- 'checkpoint_num=${10:-3000}' "${script_file}" >/dev/null
grep -F -- 'utonia_point_num=${11:-128}' "${script_file}" >/dev/null
grep -F -- 'point_cloud_num=${12:-1024}' "${script_file}" >/dev/null
grep -F -- 'output_suffix="-objpc-utonia-pointwise-hybrid${point_cloud_suffix}"' "${script_file}" >/dev/null
grep -F -- '--point_cloud_num ${point_cloud_num}' "${script_file}" >/dev/null
