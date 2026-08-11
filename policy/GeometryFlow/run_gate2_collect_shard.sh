#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 2 ]]; then
  echo "usage: $0 EPISODE_START EPISODE_STOP" >&2
  exit 2
fi

episode_start="$1"
episode_stop="$2"
if (( episode_start < 0 || episode_start >= episode_stop || episode_stop > 30 )); then
  echo "expected 0 <= start < stop <= 30" >&2
  exit 2
fi

artifact_root="${TAGRT_ARTIFACT_ROOT:-/shared2/sz/TAGRT-v1}"
python_bin="${PYTHON_BIN:-/root/miniconda3/envs/RoboTwin/bin/python}"
gpu_id="${GPU_ID:-5}"
task_name="place_shoe_geometry_marker"
task_config="demo_clean_3d_object_pc_geometry_marker_densecontrol_gate2_train30"
log_dir="${artifact_root}/experiment_outputs/fulltask_tagrt_v2/gate2_train30"

mkdir -p "${log_dir}"
export CUDA_VISIBLE_DEVICES="${gpu_id}"
export ROBOTWIN_RAW_DATA_ROOT="${artifact_root}/simulator_data"
export PYTHONUNBUFFERED=1

"${python_bin}" script/collect_data.py \
  "${task_name}" \
  "${task_config}" \
  --episode_num 30 \
  --episode_start "${episode_start}" \
  --episode_stop "${episode_stop}" \
  --skip_instructions \
  2>&1 | tee "${log_dir}/collect_${episode_start}_${episode_stop}.log"
