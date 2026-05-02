#!/bin/bash
set -euo pipefail

task_name=${1}
task_config=${2}
expert_data_num=${3}
camera_labels=${4:-left}
package_dir=${5:-data/packages}

camera_setting=${camera_labels//,/_}
dataset_name="${task_name}-${task_config}-dp-${camera_setting}-${expert_data_num}.zarr"
zarr_path="data/${dataset_name}"

if [ ! -d "${zarr_path}" ]; then
    echo "DP zarr does not exist: ${zarr_path}" >&2
    echo "Run train_real_zed.sh or process_data_real_zed.sh first." >&2
    exit 1
fi

mkdir -p "${package_dir}"
archive_path="${package_dir}/${dataset_name}.tar.gz"
readme_path="${package_dir}/${dataset_name}.README.txt"

python - "${zarr_path}" <<'PY'
import sys
from pathlib import Path

import zarr

root = zarr.open(sys.argv[1], mode="a")
source = dict(root.attrs.get("source", {}))
if "meta_path" in source:
    source["meta_path"] = Path(str(source["meta_path"])).name
source["portable"] = True
root.attrs["source"] = source
PY

tar -czf "${archive_path}" -C "data" "${dataset_name}"

cat > "${readme_path}" <<EOF
Portable DP real-ZED dataset package

Dataset:
  ${dataset_name}

Extract on the server from RoboTwin_geo/policy/DP:
  mkdir -p data
  tar -xzf ${dataset_name}.tar.gz -C data

Single-camera training example:
  bash train_real_zed.sh ${task_name} ${task_config} ${expert_data_num} 0 14 0 ${camera_labels} Large_L515 true

The zarr contains all data needed by DP training. Raw /media/... paths are not required on the server.
EOF

echo "wrote ${archive_path}"
echo "wrote ${readme_path}"
