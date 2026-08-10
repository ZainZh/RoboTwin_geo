#!/usr/bin/env bash
set -euo pipefail

if (( $# != 1 )); then
    echo "Usage: $0 LANE_ID" >&2
    exit 2
fi
lane_id=$1
case "${lane_id}" in
    0|1) ;;
    *) echo "LANE_ID must be 0 or 1, got ${lane_id}" >&2; exit 2 ;;
esac

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd "${script_dir}/../../.." && pwd)
launcher="${script_dir}/run_robotwin_sim_matched_policy.sh"
data_base=${DATA_BASE:-${repo_root}/policy/DP3/data/hanging_mug-demo_clean_3d_object_pc-50-objpc-semantic-pointwise-hybrid-sim-paper-v1}
output_root=${OUTPUT_ROOT:-${repo_root}/outputs/paper_revision/robotwin_sim_policy_hanging_mug_e300_bf16_v1}
data_marker=${DATA_MARKER:-${repo_root}/outputs/paper_revision/robotwin_sim_policy_hanging_mug_v1/matched_data.complete}
beat_smoke_marker=${BEAT_SMOKE_MARKER:-${repo_root}/outputs/paper_revision/robotwin_sim_policy_beat_hammer_e300_bf16_v1/evaluation_completion/local_fixed2_utonia_smoke.complete}
epochs=${EPOCHS:-300}
seed=${SEED:-20260805}
python_bin=${PYTHON_BIN:-/home/zheng/miniforge3/envs/RoboTwin/bin/python}

for marker in "${data_marker}" "${beat_smoke_marker}"; do
    until [[ -s "${marker}" ]]; do
        printf '[hanging-local-lane] lane=%s waiting marker=%s\n' "${lane_id}" "${marker}"
        sleep 60
    done
done

run_job() {
    local route=$1 suffix=$2 feature_dim=$3 obs_key=$4
    local zarr_path="${data_base}${suffix}.zarr"
    printf '[hanging-local-lane] lane=%s seed=%s route=%s zarr=%s\n' \
        "${lane_id}" "${seed}" "${route}" "${zarr_path}"
    env \
        EPOCHS="${epochs}" \
        SEED="${seed}" \
        BATCH_SIZE="${BATCH_SIZE:-256}" \
        TRAIN_WORKERS=0 \
        VAL_WORKERS=0 \
        PRECISION=bf16 \
        ALLOW_TF32=true \
        OUTPUT_ROOT="${output_root}" \
        PYTHON_BIN="${python_bin}" \
        RUN_ROLE=local \
        OBS_KEY="${obs_key}" \
        bash "${launcher}" "${route}" 0 hanging_mug "${zarr_path}" "${feature_dim}"
}

mkdir -p "${output_root}/lane_completion"
if [[ "${lane_id}" == "0" ]]; then
    BATCH_SIZE=${LANE0_BATCH_SIZE:-256} run_job field "" 131 semantic_point_cloud_A
    BATCH_SIZE=${LANE0_BATCH_SIZE:-256} run_job utonia -utonia 579 utonia_point_cloud_A
else
    for spec in 'xyz|-xyz|3' 'partprob|-partprob|5' 'uniformprob|-uniformprob|5' 'shuffledprob|-shuffledprob|5'; do
        IFS='|' read -r route suffix feature_dim <<< "${spec}"
        BATCH_SIZE=${LANE1_BATCH_SIZE:-224} run_job \
            "${route}" "${suffix}" "${feature_dim}" semantic_point_cloud_A
    done
fi

printf 'lane=%s completed_at_utc=%s\n' "${lane_id}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    > "${output_root}/lane_completion/local_lane${lane_id}.complete"
