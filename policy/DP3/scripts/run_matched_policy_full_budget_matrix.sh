#!/usr/bin/env bash
set -euo pipefail

gpu_id=${1:-0}
epochs=${EPOCHS:-3000}
policy_seeds=${POLICY_SEEDS:-"20260805 20260806 20260807"}
minimum_free_gib=${MINIMUM_FREE_GIB:-40}

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd "${script_dir}/../../.." && pwd)
pilot_wrapper="${script_dir}/run_matched_policy_pilot_safe_exports.sh"
output_root=${OUTPUT_ROOT:-"${repo_root}/outputs/paper_revision/policy_full_budget_safe"}
mkdir -p "${output_root}"

read -r -a seeds <<< "${policy_seeds}"
routes=(field xyz partprob)

for seed in "${seeds[@]}"; do
    for route in "${routes[@]}"; do
        run_dir="${output_root}/beat_cube_${route}_e${epochs}_seed${seed}"
        weights_path="${run_dir}/beat_cube_${route}_e${epochs}_seed${seed}.weights.pt"
        if [[ -s "${weights_path}" ]]; then
            echo "[full-budget] skip completed ${weights_path}"
            continue
        fi

        mkdir -p "${run_dir}"
        free_bytes=$(df -PB1 "${output_root}" | awk 'NR == 2 {print $4}')
        minimum_free_bytes=$((minimum_free_gib * 1024 * 1024 * 1024))
        if (( free_bytes < minimum_free_bytes )); then
            echo "[full-budget] refusing to start: free disk is below ${minimum_free_gib} GiB" >&2
            exit 3
        fi

        echo "[full-budget] seed=${seed} route=${route} epochs=${epochs}"
        SEED="${seed}" EPOCHS="${epochs}" OUTPUT_ROOT="${output_root}" \
            bash "${pilot_wrapper}" "${route}" "${gpu_id}" \
            |& tee "${run_dir}/console.log"
    done
done

echo "[full-budget] matrix complete"
