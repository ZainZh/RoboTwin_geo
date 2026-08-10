#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd "${script_dir}/../../.." && pwd)
python_bin=${PYTHON_BIN:-/workspace/venvs/geo-utonia/bin/python}
test_num=${TEST_NUM:-1}
output_dir=${OUTPUT_DIR:-${repo_root}/outputs/paper_revision/robotwin_sim_eval_perf_v1}
checkpoint_setting=${CHECKPOINT_SETTING:-paper-sim-field-joint14-e300}
weights=${WEIGHTS:-${repo_root}/outputs/paper_revision/robotwin_sim_policy_beat_hammer_e300_bf16_v1/beat_block_hammer_field_joint14_e300_seed20260806/beat_block_hammer_field_joint14_e300_seed20260806.weights.pt}
semantic_checkpoint=${SEMANTIC_CHECKPOINT:-${repo_root}/outputs/paper_revision/semantic_eval_assets/hammer_full_seed20260805/best_sem.pt}

mkdir -p "${output_dir}"
until ! pgrep -f 'train_dp3.py.*hanging_mug' >/dev/null; do
    printf '[fixed-eval-perf] waiting for Hanging training to release GPU\n'
    sleep 60
done
[[ -s "${weights}" ]] || { printf 'missing weights: %s\n' "${weights}" >&2; exit 3; }
[[ -s "${semantic_checkpoint}" ]] || { printf 'missing semantic checkpoint: %s\n' "${semantic_checkpoint}" >&2; exit 3; }

run_eval() {
    local label=$1 task_config=$2
    printf '[fixed-eval-perf] label=%s config=%s test_num=%s\n' "${label}" "${task_config}" "${test_num}"
    cd "${repo_root}"
    env CUDA_VISIBLE_DEVICES=0 HYDRA_FULL_ERROR=1 PYTHONWARNINGS=ignore::UserWarning UTONIA_ROOT="${UTONIA_ROOT:-/workspace/Utonia}" \
        "${python_bin}" script/eval_policy.py \
        --config policy/DP3/deploy_policy.yml --overrides \
        --policy_name DP3 --task_name beat_block_hammer \
        --task_config "${task_config}" --ckpt_setting "${checkpoint_setting}" \
        --result_label "${label}" \
        --expert_data_num 50 --seed 0 --test_num "${test_num}" \
        --instruction_type unseen --config_name robot_dp3_semantic_pointwise_hybrid \
        --safe_checkpoint_path "${weights}" --use_rgb False \
        --object_placeholders '{A},{B}' \
        --semantic_ckpt_A "${semantic_checkpoint}" --semantic_ckpt_B none \
        --semantic_device cuda:0 --semantic_point_num 128 \
        --semantic_input_color_mode debug_placeholder --semantic_forward_mode reference \
        --semantic_policy_output_mode embedding --point_cloud_num 1024 \
        --eval_observation_digest True
}

run_eval perf-baseline-a demo_clean_3d_object_pc_hammer_formal_fixed
run_eval perf-fast-a demo_clean_3d_object_pc_hammer_formal_fixed_fast
run_eval perf-fast-b demo_clean_3d_object_pc_hammer_formal_fixed_fast
run_eval perf-baseline-b demo_clean_3d_object_pc_hammer_formal_fixed

printf 'completed_at_utc=%s test_num=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${test_num}" > "${output_dir}/complete"
