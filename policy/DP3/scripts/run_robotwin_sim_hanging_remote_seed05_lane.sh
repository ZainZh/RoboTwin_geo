#!/usr/bin/env bash
set -euo pipefail

if (( $# != 1 )); then echo "Usage: $0 LANE_ID" >&2; exit 2; fi
lane_id=$1
case "${lane_id}" in 0|1) ;; *) echo "LANE_ID must be 0 or 1" >&2; exit 2 ;; esac

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd "${script_dir}/../../.." && pwd)
launcher="${script_dir}/run_robotwin_sim_matched_policy.sh"
data_base=${DATA_BASE:-${repo_root}/policy/DP3/data/hanging_mug-demo_clean_3d_object_pc-50-objpc-semantic-pointwise-hybrid-sim-paper-v1}
output_root=${OUTPUT_ROOT:-${repo_root}/outputs/paper_revision/robotwin_sim_policy_hanging_mug_e300_bf16_v1}
epochs=${EPOCHS:-300}
python_bin=${PYTHON_BIN:-/workspace/venvs/geo-utonia/bin/python}

run_job() {
    local route=$1 suffix=$2 feature_dim=$3 obs_key=$4
    env EPOCHS="${epochs}" SEED=20260805 BATCH_SIZE=256 \
        TRAIN_WORKERS=1 VAL_WORKERS=1 \
        TRAIN_PERSISTENT_WORKERS=true VAL_PERSISTENT_WORKERS=true \
        PRECISION=bf16 ALLOW_TF32=true OUTPUT_ROOT="${output_root}" \
        PYTHON_BIN="${python_bin}" RUN_ROLE=remote OBS_KEY="${obs_key}" \
        bash "${launcher}" "${route}" 0 hanging_mug \
        "${data_base}${suffix}.zarr" "${feature_dim}"
}

if [[ "${lane_id}" == "0" ]]; then
    run_job field '' 131 semantic_point_cloud_A
    run_job utonia -utonia 579 utonia_point_cloud_A
else
    run_job partprob -partprob 5 semantic_point_cloud_A
    run_job uniformprob -uniformprob 5 semantic_point_cloud_A
    run_job shuffledprob -shuffledprob 5 semantic_point_cloud_A
fi
