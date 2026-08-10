#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd "${script_dir}/../../.." && pwd)
finalizer="${script_dir}/finalize_robotwin_sim_policy_run.py"
python_bin=${PYTHON_BIN:-/home/zheng/miniforge3/envs/RoboTwin/bin/python}
data_base=${DATA_BASE:-${repo_root}/policy/DP3/data/hanging_mug-demo_clean_3d_object_pc-50-objpc-semantic-pointwise-hybrid-sim-paper-v1}
output_root=${OUTPUT_ROOT:-${repo_root}/outputs/paper_revision/robotwin_sim_policy_hanging_mug_e300_bf16_v1}
epochs=${EPOCHS:-300}
seed=${SEED:-20260805}
interval=${MONITOR_INTERVAL_SECONDS:-60}
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

while true; do
    completed=0
    for spec in "${specs[@]}"; do
        IFS='|' read -r route suffix obs_key feature_dim <<< "${spec}"
        run_name="hanging_mug_${route}_joint14_e${epochs}_seed${seed}"
        run_dir="${output_root}/${run_name}"
        weights="${run_dir}/${run_name}.weights.pt"
        manifest="${run_dir}/${run_name}.completion-v2.json"
        if [[ -s "${manifest}" ]]; then
            (( completed += 1 ))
            continue
        fi
        [[ -s "${weights}" ]] || continue
        args=(
            "${weights}" --manifest "${manifest}"
            --expected-epoch "${epochs}" --expected-seed "${seed}"
            --expected-task-name "hanging_mug-paper-sim-${route}-joint14-e${epochs}-50"
            --expected-zarr-path "${data_base}${suffix}.zarr"
            --expected-precision bf16 --expected-allow-tf32 true
            --expected-obs-key "${obs_key}" --expected-feature-dim "${feature_dim}"
            --hydra-dir "${run_dir}/.hydra" --role local
        )
        for code_path in "${code_paths[@]}"; do
            args+=(--code-path "${code_path}")
        done
        "${python_bin}" "${finalizer}" "${args[@]}"
        (( completed += 1 ))
    done
    printf '[hanging-result-monitor] completed=%s/6 checked_at_utc=%s\n' \
        "${completed}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    (( completed == 6 )) && exit 0
    sleep "${interval}"
done
