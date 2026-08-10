#!/usr/bin/env bash
set -euo pipefail

if (( $# != 1 )); then
    echo "Usage: $0 local|remote|receipt" >&2
    exit 2
fi

role=$1
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd "${script_dir}/../../.." && pwd)
finalizer="${script_dir}/finalize_robotwin_sim_policy_run.py"
epochs=${EPOCHS:-300}
interval=${MONITOR_INTERVAL_SECONDS:-60}
output_root=${OUTPUT_ROOT:-${repo_root}/outputs/paper_revision/robotwin_sim_policy_beat_hammer_e300_bf16_v1}

case "${role}" in
    local)
        seeds=(20260805)
        expected_data_base=${EXPECTED_DATA_BASE:-${repo_root}/policy/DP3/data/beat_block_hammer-demo_clean_3d_object_pc-50-objpc-semantic-pointwise-hybrid-sim-paper-v1}
        python_bin=${PYTHON_BIN:-/home/zheng/miniforge3/envs/RoboTwin/bin/python}
        marker_suffix=completion-v2.json
        finalizer_role=local
        ;;
    remote)
        seeds=(20260806 20260807)
        expected_data_base=${EXPECTED_DATA_BASE:-${repo_root}/policy/DP3/data/beat_block_hammer-demo_clean_3d_object_pc-50-objpc-semantic-pointwise-hybrid-sim-paper-v1}
        python_bin=${PYTHON_BIN:-/workspace/venvs/geo-utonia/bin/python}
        marker_suffix=completion-v2.json
        finalizer_role=remote
        ;;
    receipt)
        seeds=(20260806 20260807)
        expected_data_base=${EXPECTED_DATA_BASE:-/workspace/RoboTwin_geo_sim_v1/policy/DP3/data/beat_block_hammer-demo_clean_3d_object_pc-50-objpc-semantic-pointwise-hybrid-sim-paper-v1}
        python_bin=${PYTHON_BIN:-/home/zheng/miniforge3/envs/RoboTwin/bin/python}
        marker_suffix=local-receipt-v2.json
        finalizer_role=local_receipt
        ;;
    *)
        echo "ROLE must be local, remote, or receipt, got ${role}" >&2
        exit 2
        ;;
esac

if [[ ! "${interval}" =~ ^[0-9]+$ ]] || (( interval < 30 )); then
    echo "MONITOR_INTERVAL_SECONDS must be an integer >= 30, got ${interval}" >&2
    exit 2
fi

specs=(
    'field||semantic_point_cloud_A|131'
    'xyz|-xyz|semantic_point_cloud_A|3'
    'partprob|-partprob|semantic_point_cloud_A|5'
    'uniformprob|-uniformprob|semantic_point_cloud_A|5'
    'shuffledprob|-shuffledprob|semantic_point_cloud_A|5'
    'utonia|-utonia|utonia_point_cloud_A|579'
)
code_paths=(
    "${repo_root}/policy/DP3/3D-Diffusion-Policy/train_dp3.py"
    "${repo_root}/policy/DP3/scripts/run_robotwin_sim_matched_policy.sh"
    "${repo_root}/policy/DP3/3D-Diffusion-Policy/diffusion_policy_3d/config/robot_dp3_semantic_pointwise_hybrid.yaml"
    "${repo_root}/policy/DP3/scripts/safe_dp3_checkpoint.py"
)

manifest_matches() {
    local manifest=$1 seed=$2 route=$3 zarr_path=$4 obs_key=$5 feature_dim=$6 expected_role=$7
    [[ -s "${manifest}" ]] || return 1
    "${python_bin}" -c '
import json, sys
path, seed, route, zarr_path, obs_key, feature_dim, role, epochs = sys.argv[1:]
with open(path, "r", encoding="utf-8") as handle:
    payload = json.load(handle)
expected = {
    "epoch": int(epochs), "seed": int(seed),
    "task_name": f"beat_block_hammer-paper-sim-{route}-joint14-e{epochs}-50",
    "zarr_path": zarr_path, "precision": "bf16", "allow_tf32": True,
}
assert payload.get("schema") == "robotwin_sim_policy_run_completion_v2"
assert payload.get("status") == "complete" and payload.get("role") == role
assert payload.get("expected_metadata") == expected
policy = payload.get("expected_policy", {})
expected_shape = {
    "obs": {
        "point_cloud": {"shape": [1024, 6], "type": "point_cloud"},
        "agent_pos": {"shape": [14], "type": "low_dim"},
        obs_key: {"shape": [128, int(feature_dim)], "type": "point_cloud"},
    },
    "action": {"shape": [14]},
}
assert policy.get("obs_key") == obs_key
assert policy.get("feature_dim") == int(feature_dim)
assert policy.get("use_ema") is True
assert policy.get("shape_meta") == expected_shape
' "${manifest}" "${seed}" "${route}" "${zarr_path}" "${obs_key}" \
        "${feature_dim}" "${expected_role}" "${epochs}" >/dev/null 2>&1
}

completed_count() {
    local count=0
    local seed spec route suffix obs_key feature_dim run_name manifest zarr_path
    for seed in "${seeds[@]}"; do
        for spec in "${specs[@]}"; do
            IFS='|' read -r route suffix obs_key feature_dim <<< "${spec}"
            run_name="beat_block_hammer_${route}_joint14_e${epochs}_seed${seed}"
            manifest="${output_root}/${run_name}/${run_name}.${marker_suffix}"
            zarr_path="${expected_data_base}${suffix}.zarr"
            if manifest_matches "${manifest}" "${seed}" "${route}" "${zarr_path}" \
                "${obs_key}" "${feature_dim}" "${finalizer_role}"; then
                (( count += 1 ))
            fi
        done
    done
    printf '%s\n' "${count}"
}

total=$(( ${#seeds[@]} * ${#specs[@]} ))
while true; do
    for seed in "${seeds[@]}"; do
        for spec in "${specs[@]}"; do
            IFS='|' read -r route suffix obs_key feature_dim <<< "${spec}"
            run_name="beat_block_hammer_${route}_joint14_e${epochs}_seed${seed}"
            run_dir="${output_root}/${run_name}"
            weights="${run_dir}/${run_name}.weights.pt"
            manifest="${run_dir}/${run_name}.${marker_suffix}"
            zarr_path="${expected_data_base}${suffix}.zarr"
            if manifest_matches "${manifest}" "${seed}" "${route}" "${zarr_path}" \
                "${obs_key}" "${feature_dim}" "${finalizer_role}"; then
                continue
            fi
            [[ -s "${weights}" ]] || continue
            args=(
                "${weights}"
                --manifest "${manifest}"
                --expected-epoch "${epochs}"
                --expected-seed "${seed}"
                --expected-task-name "beat_block_hammer-paper-sim-${route}-joint14-e${epochs}-50"
                --expected-zarr-path "${zarr_path}"
                --expected-precision bf16
                --expected-allow-tf32 true
                --expected-obs-key "${obs_key}"
                --expected-feature-dim "${feature_dim}"
                --hydra-dir "${run_dir}/.hydra"
                --role "${finalizer_role}"
            )
            for code_path in "${code_paths[@]}"; do
                args+=(--code-path "${code_path}")
            done
            if [[ "${role}" == "receipt" ]]; then
                source_manifest="${run_dir}/${run_name}.completion-v2.json"
                [[ -s "${source_manifest}" ]] || continue
                args+=(--source-manifest "${source_manifest}")
            fi
            "${python_bin}" "${finalizer}" "${args[@]}"
        done
    done

    completed=$(completed_count)
    printf '[formal-result-monitor] role=%s completed=%s/%s checked_at_utc=%s\n' \
        "${role}" "${completed}" "${total}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    if (( completed == total )); then
        exit 0
    fi
    if [[ "${ONCE:-0}" == "1" ]]; then
        exit 0
    fi
    sleep "${interval}"
done
