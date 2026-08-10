#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd "${script_dir}/../../.." && pwd)
output_root=${OUTPUT_ROOT:-${repo_root}/outputs/paper_revision/robotwin_sim_policy_beat_hammer_e300_bf16_v1}
python_bin=${PYTHON_BIN:-/home/zheng/miniforge3/envs/RoboTwin/bin/python}
semantic_checkpoint=${SEMANTIC_CHECKPOINT:-${repo_root}/outputs/paper_revision/hammer_stage42_independent_local4090_seed20260805_e4000_split424242_20260806/hammer_full_fixed_e4000_split424242_seed20260805/best_sem.pt}
utonia_checkpoint=${UTONIA_CHECKPOINT:-/home/zheng/.cache/utonia/ckpt/utonia.pth}
utonia_projection=${UTONIA_PROJECTION:-${repo_root}/policy/DP3/data/beat_block_hammer-utonia-rp128-seed20260809.npz}
seed=${SEED:-20260805}
epochs=${EPOCHS:-300}
task_config=${TASK_CONFIG_NAME:-demo_clean_3d_object_pc_hammer_formal_fixed}
test_num=${TEST_NUM:-100}
routes=${ROUTES:-"field xyz partprob uniformprob shuffledprob utonia"}

[[ "${test_num}" =~ ^[0-9]+$ ]] && (( test_num > 0 )) || { echo "TEST_NUM must be positive" >&2; exit 2; }
for required in "${semantic_checkpoint}"; do
    [[ -s "${required}" ]] || { echo "required artifact missing: ${required}" >&2; exit 3; }
done

run_eval() {
    local route=$1 output_mode
    case "${route}" in
        vanilla) output_mode=none ;;
        field) output_mode=embedding ;;
        xyz) output_mode=xyz ;;
        partprob) output_mode=part_prob ;;
        uniformprob) output_mode=uniform_prob ;;
        shuffledprob) output_mode=shuffled_prob ;;
        utonia) output_mode=matched_utonia ;;
        utonia128) output_mode=matched_utonia128 ;;
        *) echo "unsupported route: ${route}" >&2; exit 2 ;;
    esac
    local run_name="beat_block_hammer_${route}_joint14_e${epochs}_seed${seed}"
    local weights="${output_root}/${run_name}/${run_name}.weights.pt"
    local completion="${output_root}/${run_name}/${run_name}.completion-v2.json"
    [[ -s "${weights}" && -s "${completion}" ]] || { echo "missing formal checkpoint/manifest for ${route}" >&2; exit 3; }
    local config_name=robot_dp3_semantic_pointwise_hybrid
    if [[ "${route}" == "vanilla" ]]; then config_name=robot_dp3_objpc; fi
    local args=(
        "${python_bin}" script/eval_policy.py
        --config policy/DP3/deploy_policy.yml --overrides
        --policy_name DP3 --task_name beat_block_hammer
        --task_config "${task_config}"
        --ckpt_setting "paper-sim-${route}-joint14-e${epochs}"
        --expert_data_num 50 --seed 0 --test_num "${test_num}"
        --instruction_type unseen --config_name "${config_name}"
        --safe_checkpoint_path "${weights}" --use_rgb False
        --object_placeholders '{A},{B}'
        --point_cloud_num 1024
        --eval_observation_digest True
    )
    if [[ "${route}" != "vanilla" ]]; then
        args+=(
            --semantic_ckpt_A "${semantic_checkpoint}" --semantic_ckpt_B none
            --semantic_device cuda:0 --semantic_point_num 128
            --semantic_input_color_mode debug_placeholder --semantic_forward_mode reference
            --semantic_policy_output_mode "${output_mode}"
        )
    fi
    if [[ "${route}" == "utonia" || "${route}" == "utonia128" ]]; then
        [[ -s "${utonia_checkpoint}" ]] || { echo "missing Utonia checkpoint" >&2; exit 3; }
        args+=(--utonia_checkpoint "${utonia_checkpoint}" --utonia_feature_placeholders '{A}' --utonia_device cuda:0 --matched_utonia_color_mode debug_placeholder --matched_utonia_normal_mode fallback)
    fi
    if [[ "${route}" == "utonia128" ]]; then
        [[ -s "${utonia_projection}" ]] || { echo "missing Utonia projection" >&2; exit 3; }
        args+=(--matched_utonia_projection_path "${utonia_projection}" --matched_utonia_projection_dim 128)
    fi
    printf '[beat-fixed100] route=%s seed=%s test_num=%s checkpoint=%s\n' "${route}" "${seed}" "${test_num}" "${weights}"
    env CUDA_VISIBLE_DEVICES=0 HYDRA_FULL_ERROR=1 PYTHONWARNINGS=ignore::UserWarning UTONIA_ROOT="${UTONIA_ROOT:-/home/zheng/github/Utonia}" "${args[@]}"
}

cd "${repo_root}"
for route in ${routes}; do
    run_eval "${route}"
done

mkdir -p "${output_root}/evaluation_completion"
printf 'routes=%s test_num=%s completed_at_utc=%s\n' "${routes}" "${test_num}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    > "${output_root}/evaluation_completion/local_fixed${test_num}_seed${seed}.complete"
