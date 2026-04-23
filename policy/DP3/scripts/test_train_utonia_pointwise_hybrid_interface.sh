#!/bin/bash
set -euo pipefail

script_file="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/train_utonia_pointwise_hybrid.sh"

grep -F -- 'task_name=${1}' "${script_file}" >/dev/null
grep -F -- 'task_config=${2}' "${script_file}" >/dev/null
grep -F -- 'expert_data_num=${3}' "${script_file}" >/dev/null
grep -F -- 'seed=${4}' "${script_file}" >/dev/null
grep -F -- 'gpu_id=${5}' "${script_file}" >/dev/null
grep -F -- 'utonia_device=${6:-cuda:0}' "${script_file}" >/dev/null
grep -F -- 'object_placeholders=${7:-\{A\},\{B\}}' "${script_file}" >/dev/null
grep -F -- 'utonia_feature_placeholders=${8:-\{A\},\{B\}}' "${script_file}" >/dev/null
grep -F -- 'utonia_point_num=${9:-128}' "${script_file}" >/dev/null
grep -F -- 'batch_size=${10:-256}' "${script_file}" >/dev/null
grep -F -- 'val_batch_size=${11:-${batch_size}}' "${script_file}" >/dev/null
grep -F -- 'use_ema=${12:-true}' "${script_file}" >/dev/null
grep -F -- 'gradient_accumulate_every=${13:-1}' "${script_file}" >/dev/null
grep -F -- 'encoder_output_dim=${14:-128}' "${script_file}" >/dev/null
