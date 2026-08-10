#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd "${script_dir}/../../.." && pwd)
output_root=${OUTPUT_ROOT:-${repo_root}/outputs/paper_revision/robotwin_sim_policy_beat_hammer_e300_bf16_v1}
python_bin=${PYTHON_BIN:-/workspace/venvs/geo-utonia/bin/python}
semantic_checkpoint=${SEMANTIC_CHECKPOINT:-${repo_root}/outputs/paper_revision/semantic_eval_assets/hammer_full_seed20260805/best_sem.pt}
utonia_checkpoint=${UTONIA_CHECKPOINT:-/root/.cache/utonia/ckpt/utonia.pth}
utonia_projection=${UTONIA_PROJECTION:-${repo_root}/policy/DP3/data/beat_block_hammer-utonia-rp128-seed20260809.npz}
test_num=${TEST_NUM:-100}
task_config=${TASK_CONFIG_NAME:-demo_clean_3d_object_pc_hammer_formal_fixed_fast}
train_seeds=${TRAIN_SEEDS:-"20260806 20260807"}
routes=${ROUTES:-"field xyz partprob uniformprob shuffledprob utonia"}

until [[ -s "${semantic_checkpoint}" ]]; do
    printf '[beat-remote-fixed100] waiting for semantic checkpoint: %s\n' "${semantic_checkpoint}"
    sleep 60
done
until [[ -s "${utonia_checkpoint}" ]]; do
    printf '[beat-remote-fixed100] waiting for Utonia checkpoint: %s\n' "${utonia_checkpoint}"
    sleep 60
done
while pgrep -f 'train_dp3.py.*hanging_mug' >/dev/null; do
    printf '[beat-remote-fixed100] waiting for Hanging seed-05 training to release GPU\n'
    sleep 120
done

run_eval() {
    local seed=$1 route=$2 output_mode=$3
    local run_name="beat_block_hammer_${route}_joint14_e300_seed${seed}"
    local weights="${output_root}/${run_name}/${run_name}.weights.pt"
    local completion="${output_root}/${run_name}/${run_name}.completion-v2.json"
    [[ -s "${weights}" && -s "${completion}" ]] || { echo "missing formal run: ${run_name}" >&2; exit 3; }
    local config_name=robot_dp3_semantic_pointwise_hybrid
    if [[ "${route}" == "vanilla" ]]; then config_name=robot_dp3_objpc; fi
    local args=(
        "${python_bin}" script/eval_policy.py
        --config policy/DP3/deploy_policy.yml --overrides
        --policy_name DP3 --task_name beat_block_hammer
        --task_config "${task_config}"
        --ckpt_setting "paper-sim-${route}-joint14-e300"
        --result_label "paper-sim-${route}-joint14-e300-trainseed${seed}"
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
        args+=(--utonia_checkpoint "${utonia_checkpoint}" --utonia_feature_placeholders '{A}' --utonia_device cuda:0 --matched_utonia_color_mode debug_placeholder --matched_utonia_normal_mode fallback)
    fi
    if [[ "${route}" == "utonia128" ]]; then
        [[ -s "${utonia_projection}" ]] || { echo "missing Utonia projection" >&2; exit 3; }
        args+=(--matched_utonia_projection_path "${utonia_projection}" --matched_utonia_projection_dim 128)
    fi
    printf '[beat-remote-fixed100] seed=%s route=%s test_num=%s\n' "${seed}" "${route}" "${test_num}"
    env CUDA_VISIBLE_DEVICES=0 HYDRA_FULL_ERROR=1 PYTHONWARNINGS=ignore::UserWarning UTONIA_ROOT=/workspace/Utonia "${args[@]}"
}

cd "${repo_root}"
for seed in ${train_seeds}; do
    for route in ${routes}; do
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
        run_eval "${seed}" "${route}" "${output_mode}"
    done
done

mkdir -p "${output_root}/evaluation_completion"
seed_tag=${train_seeds// /_}
route_tag=${routes// /_}
printf 'seeds=%s routes=%s test_num=%s completed_at_utc=%s\n' "${train_seeds}" "${routes}" "${test_num}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    > "${output_root}/evaluation_completion/remote_fixed${test_num}_${seed_tag}_${route_tag}.complete"
