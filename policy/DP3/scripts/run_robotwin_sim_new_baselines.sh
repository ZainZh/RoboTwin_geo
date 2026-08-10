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
launcher="${script_dir}/run_robotwin_sim_matched_policy.sh"
finalizer="${script_dir}/finalize_robotwin_sim_policy_run.py"
data_base=${DATA_BASE:-${repo_root}/policy/DP3/data/beat_block_hammer-demo_clean_3d_object_pc-50-objpc-semantic-pointwise-hybrid-sim-paper-v1}
output_root=${OUTPUT_ROOT:-${repo_root}/outputs/paper_revision/robotwin_sim_policy_beat_hammer_e300_bf16_v1}
projection=${UTONIA_PROJECTION:-${repo_root}/policy/DP3/data/beat_block_hammer-utonia-rp128-seed20260809.npz}
epochs=${EPOCHS:-300}
python_bin=${PYTHON_BIN:-python}

if [[ "${role}" == "local" ]]; then
    train_workers=${TRAIN_WORKERS:-0}
    val_workers=${VAL_WORKERS:-0}
else
    train_workers=${TRAIN_WORKERS:-0}
    val_workers=${VAL_WORKERS:-0}
fi

run_one() {
    local route=$1 suffix=$2 obs_key=$3 feature_dim=$4
    local zarr_path="${data_base}${suffix}.zarr"
    local run_name="beat_block_hammer_${route}_joint14_e${epochs}_seed${seed}"
    local run_dir="${output_root}/${run_name}"
    local weights="${run_dir}/${run_name}.weights.pt"
    local manifest="${run_dir}/${run_name}.completion-v2.json"
    local config_name=robot_dp3_semantic_pointwise_hybrid.yaml
    [[ "${route}" == "vanilla" ]] && config_name=robot_dp3_objpc.yaml
    [[ -d "${zarr_path}" ]] || { echo "missing data: ${zarr_path}" >&2; exit 3; }

    environment=(
        "EPOCHS=${epochs}" "SEED=${seed}" "BATCH_SIZE=${BATCH_SIZE:-256}"
        "TRAIN_WORKERS=${train_workers}" "VAL_WORKERS=${val_workers}"
        "PRECISION=bf16" "ALLOW_TF32=true" "OUTPUT_ROOT=${output_root}"
        "PYTHON_BIN=${python_bin}" "RUN_ROLE=${role}" "OBS_KEY=${obs_key}"
        "DRY_RUN=${DRY_RUN:-0}"
    )
    env "${environment[@]}" bash "${launcher}" "${route}" 0 beat_block_hammer "${zarr_path}" "${feature_dim}"
    [[ "${DRY_RUN:-0}" == "1" ]] && return 0

    if [[ ! -s "${manifest}" ]]; then
        args=(
            "${weights}" --manifest "${manifest}"
            --expected-epoch "${epochs}" --expected-seed "${seed}"
            --expected-task-name "beat_block_hammer-paper-sim-${route}-joint14-e${epochs}-50"
            --expected-zarr-path "${zarr_path}"
            --expected-precision bf16 --expected-allow-tf32 true
            --expected-obs-key "${obs_key}" --expected-feature-dim "${feature_dim}"
            --hydra-dir "${run_dir}/.hydra" --role "${role}"
            --code-path "${repo_root}/policy/DP3/3D-Diffusion-Policy/train_dp3.py"
            --code-path "${launcher}"
            --code-path "${repo_root}/policy/DP3/3D-Diffusion-Policy/diffusion_policy_3d/config/${config_name}"
            --code-path "${script_dir}/safe_dp3_checkpoint.py"
        )
        if [[ "${route}" == "utonia128" ]]; then
            [[ -s "${projection}" ]] || { echo "missing projection: ${projection}" >&2; exit 3; }
            args+=(
                --code-path "${script_dir}/utonia_projection_utils.py"
                --code-path "${script_dir}/build_utonia128_policy_zarr.py"
                --code-path "${projection}"
            )
        fi
        "${python_bin}" "${finalizer}" "${args[@]}"
    fi
}

run_one vanilla "" none 0
run_one utonia128 -utonia128 utonia_point_cloud_A 131

mkdir -p "${output_root}/lane_completion"
printf 'role=%s seed=%s routes=vanilla,utonia128 completed_at_utc=%s\n' "${role}" "${seed}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "${output_root}/lane_completion/new_baselines_${role}_seed${seed}.complete"
