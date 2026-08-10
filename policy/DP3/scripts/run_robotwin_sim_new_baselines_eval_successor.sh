#!/usr/bin/env bash
set -euo pipefail

if (( $# != 2 )); then
    echo "Usage: $0 local|remote TRAINING_SEED" >&2
    exit 2
fi
role=$1
seed=$2
case "${role}" in local|remote) ;; *) echo "invalid role: ${role}" >&2; exit 2 ;; esac
[[ "${seed}" =~ ^[0-9]+$ ]] || { echo "seed must be an integer" >&2; exit 2; }

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd "${script_dir}/../../.." && pwd)
output_root=${OUTPUT_ROOT:-${repo_root}/outputs/paper_revision/robotwin_sim_policy_beat_hammer_e300_bf16_v1}
python_bin=${PYTHON_BIN:-python}
test_num=${TEST_NUM:-100}
interval=${WAIT_INTERVAL_SECONDS:-60}
lane_marker="${output_root}/lane_completion/new_baselines_${role}_seed${seed}.complete"
completion_dir="${output_root}/evaluation_completion"
completion="${completion_dir}/new_baselines_${role}_fixed${test_num}_seed${seed}.complete"

[[ "${test_num}" =~ ^[0-9]+$ ]] && (( test_num > 0 )) || {
    echo "TEST_NUM must be positive" >&2
    exit 2
}
[[ "${interval}" =~ ^[0-9]+$ ]] && (( interval >= 30 )) || {
    echo "WAIT_INTERVAL_SECONDS must be an integer >= 30" >&2
    exit 2
}

if [[ -s "${completion}" ]]; then
    echo "[new-baseline-eval] already complete: ${completion}"
    exit 0
fi
until [[ -s "${lane_marker}" ]]; do
    printf '[new-baseline-eval] waiting role=%s seed=%s marker=%s\n' \
        "${role}" "${seed}" "${lane_marker}"
    sleep "${interval}"
done

common_env=(
    "OUTPUT_ROOT=${output_root}" "PYTHON_BIN=${python_bin}"
    "TEST_NUM=${test_num}" "ROUTES=vanilla utonia128"
)
if [[ "${role}" == "local" ]]; then
    env "${common_env[@]}" "SEED=${seed}" \
        bash "${script_dir}/run_robotwin_sim_beat_fixed100_seed05.sh"
else
    env "${common_env[@]}" "TRAIN_SEEDS=${seed}" \
        bash "${script_dir}/run_robotwin_sim_beat_remote_fixed100.sh"
fi

mkdir -p "${completion_dir}"
printf 'role=%s seed=%s routes=vanilla,utonia128 test_num=%s completed_at_utc=%s\n' \
    "${role}" "${seed}" "${test_num}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    > "${completion}"
echo "[new-baseline-eval] complete: ${completion}"
