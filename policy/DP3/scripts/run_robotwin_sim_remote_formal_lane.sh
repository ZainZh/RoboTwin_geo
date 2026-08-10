#!/usr/bin/env bash
set -euo pipefail

if (( $# != 1 )); then
    echo "Usage: $0 LANE_ID" >&2
    exit 2
fi

lane_id=$1
case "${lane_id}" in
    0|1) ;;
    *)
        echo "LANE_ID must be 0 or 1, got ${lane_id}" >&2
        exit 2
        ;;
esac

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd "${script_dir}/../../.." && pwd)
launcher="${script_dir}/run_robotwin_sim_matched_policy.sh"
data_base=${DATA_BASE:-${repo_root}/policy/DP3/data/beat_block_hammer-demo_clean_3d_object_pc-50-objpc-semantic-pointwise-hybrid-sim-paper-v1}
output_root=${OUTPUT_ROOT:-${repo_root}/outputs/paper_revision/robotwin_sim_policy_beat_hammer_e300_bf16_v1}
epochs=${EPOCHS:-300}
python_bin=${PYTHON_BIN:-python}

run_job() {
    local seed=$1
    local route=$2
    local suffix=$3
    local feature_dim=$4
    local obs_key=$5
    local zarr_path="${data_base}${suffix}.zarr"
    printf '[formal-lane] lane=%s seed=%s route=%s zarr=%s\n' \
        "${lane_id}" "${seed}" "${route}" "${zarr_path}"
    env \
        EPOCHS="${epochs}" \
        SEED="${seed}" \
        BATCH_SIZE="${BATCH_SIZE:-256}" \
        TRAIN_WORKERS="${TRAIN_WORKERS:-1}" \
        VAL_WORKERS="${VAL_WORKERS:-1}" \
        TRAIN_PERSISTENT_WORKERS="${TRAIN_PERSISTENT_WORKERS:-true}" \
        VAL_PERSISTENT_WORKERS="${VAL_PERSISTENT_WORKERS:-true}" \
        PRECISION="${PRECISION:-bf16}" \
        ALLOW_TF32="${ALLOW_TF32:-true}" \
        OUTPUT_ROOT="${output_root}" \
        PYTHON_BIN="${python_bin}" \
        RUN_ROLE=remote \
        OBS_KEY="${obs_key}" \
        bash "${launcher}" "${route}" 0 beat_block_hammer \
            "${zarr_path}" "${feature_dim}"
}

if [[ "${lane_id}" == "0" ]]; then
    run_job 20260806 field "" 131 semantic_point_cloud_A
    run_job 20260806 partprob -partprob 5 semantic_point_cloud_A
    run_job 20260806 shuffledprob -shuffledprob 5 semantic_point_cloud_A
    run_job 20260807 xyz -xyz 3 semantic_point_cloud_A
    run_job 20260807 uniformprob -uniformprob 5 semantic_point_cloud_A
    run_job 20260807 utonia -utonia 579 utonia_point_cloud_A
else
    run_job 20260806 xyz -xyz 3 semantic_point_cloud_A
    run_job 20260806 uniformprob -uniformprob 5 semantic_point_cloud_A
    run_job 20260806 utonia -utonia 579 utonia_point_cloud_A
    run_job 20260807 field "" 131 semantic_point_cloud_A
    run_job 20260807 partprob -partprob 5 semantic_point_cloud_A
    run_job 20260807 shuffledprob -shuffledprob 5 semantic_point_cloud_A
fi

mkdir -p "${output_root}/lane_completion"
printf 'lane=%s completed_at_utc=%s\n' "${lane_id}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    > "${output_root}/lane_completion/remote_lane${lane_id}.complete"
