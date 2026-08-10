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
seed=${SEED:-20260805}
python_bin=${PYTHON_BIN:-/home/zheng/miniforge3/envs/RoboTwin/bin/python}

weights_path() {
    local route=$1
    local run_name="beat_block_hammer_${route}_joint14_e${epochs}_seed${seed}"
    printf '%s/%s/%s.weights.pt\n' "${output_root}" "${run_name}" "${run_name}"
}

wait_for_weights() {
    local route=$1
    local path
    path=$(weights_path "${route}")
    until [[ -s "${path}" ]]; do
        printf '[local-successor] waiting route=%s weights=%s\n' "${route}" "${path}"
        sleep 20
    done
    printf '[local-successor] dependency ready route=%s weights=%s\n' "${route}" "${path}"
}

run_job() {
    local route=$1
    local suffix=$2
    local feature_dim=$3
    local obs_key=$4
    local zarr_path="${data_base}${suffix}.zarr"
    printf '[local-successor] lane=%s seed=%s route=%s zarr=%s\n' \
        "${lane_id}" "${seed}" "${route}" "${zarr_path}"
    env \
        EPOCHS="${epochs}" \
        SEED="${seed}" \
        BATCH_SIZE="${BATCH_SIZE:-256}" \
        TRAIN_WORKERS="${TRAIN_WORKERS:-0}" \
        VAL_WORKERS="${VAL_WORKERS:-0}" \
        PRECISION="${PRECISION:-bf16}" \
        ALLOW_TF32="${ALLOW_TF32:-true}" \
        OUTPUT_ROOT="${output_root}" \
        PYTHON_BIN="${python_bin}" \
        RUN_ROLE=local \
        OBS_KEY="${obs_key}" \
        bash "${launcher}" "${route}" 0 beat_block_hammer \
            "${zarr_path}" "${feature_dim}"
}

completion_dir="${output_root}/lane_completion"
mkdir -p "${completion_dir}"

if [[ "${lane_id}" == "0" ]]; then
    wait_for_weights xyz
    run_job uniformprob -uniformprob 5 semantic_point_cloud_A
    until [[ -s "${completion_dir}/local_lane1.complete" ]]; do
        sleep 20
    done
    run_job field "" 131 semantic_point_cloud_A
    run_job utonia -utonia 579 utonia_point_cloud_A
else
    wait_for_weights partprob
    run_job shuffledprob -shuffledprob 5 semantic_point_cloud_A
fi

printf 'lane=%s completed_at_utc=%s\n' "${lane_id}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    > "${completion_dir}/local_lane${lane_id}.complete"
