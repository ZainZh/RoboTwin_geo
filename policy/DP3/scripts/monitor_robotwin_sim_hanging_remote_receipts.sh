#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd "${script_dir}/../../.." && pwd)
finalizer="${script_dir}/finalize_robotwin_sim_policy_run.py"
python_bin=${PYTHON_BIN:-/home/zheng/miniforge3/envs/RoboTwin/bin/python}
remote_data_base=${REMOTE_DATA_BASE:-/workspace/RoboTwin_geo_sim_v1/policy/DP3/data/hanging_mug-demo_clean_3d_object_pc-50-objpc-semantic-pointwise-hybrid-sim-paper-v1}
output_root=${OUTPUT_ROOT:-${repo_root}/outputs/paper_revision/robotwin_sim_policy_hanging_mug_e300_bf16_v1}
epochs=${EPOCHS:-300}
specs=('field||semantic_point_cloud_A|131' 'xyz|-xyz|semantic_point_cloud_A|3' 'partprob|-partprob|semantic_point_cloud_A|5' 'uniformprob|-uniformprob|semantic_point_cloud_A|5' 'shuffledprob|-shuffledprob|semantic_point_cloud_A|5' 'utonia|-utonia|utonia_point_cloud_A|579')
code_paths=("${repo_root}/policy/DP3/3D-Diffusion-Policy/train_dp3.py" "${repo_root}/policy/DP3/scripts/run_robotwin_sim_matched_policy.sh" "${repo_root}/policy/DP3/3D-Diffusion-Policy/diffusion_policy_3d/config/robot_dp3_semantic_pointwise_hybrid.yaml" "${repo_root}/policy/DP3/scripts/safe_dp3_checkpoint.py")

while true; do
    completed=0
    for seed in 20260806 20260807; do
        for spec in "${specs[@]}"; do
            IFS='|' read -r route suffix obs_key feature_dim <<< "${spec}"
            run_name="hanging_mug_${route}_joint14_e${epochs}_seed${seed}"
            run_dir="${output_root}/${run_name}"
            weights="${run_dir}/${run_name}.weights.pt"
            source_manifest="${run_dir}/${run_name}.completion-v2.json"
            receipt="${run_dir}/${run_name}.local-receipt-v2.json"
            if [[ -s "${receipt}" ]]; then (( completed += 1 )); continue; fi
            [[ -s "${weights}" && -s "${source_manifest}" ]] || continue
            args=("${weights}" --manifest "${receipt}" --expected-epoch "${epochs}" --expected-seed "${seed}" --expected-task-name "hanging_mug-paper-sim-${route}-joint14-e${epochs}-50" --expected-zarr-path "${remote_data_base}${suffix}.zarr" --expected-precision bf16 --expected-allow-tf32 true --expected-obs-key "${obs_key}" --expected-feature-dim "${feature_dim}" --hydra-dir "${run_dir}/.hydra" --role local_receipt --source-manifest "${source_manifest}")
            for code_path in "${code_paths[@]}"; do args+=(--code-path "${code_path}"); done
            "${python_bin}" "${finalizer}" "${args[@]}"
            (( completed += 1 ))
        done
    done
    printf '[hanging-receipt-monitor] completed=%s/12 checked_at_utc=%s\n' "${completed}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    (( completed == 12 )) && exit 0
    sleep 60
done
