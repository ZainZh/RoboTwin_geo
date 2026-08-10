#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd "${script_dir}/../../.." && pwd)
output_root=${OUTPUT_ROOT:-${repo_root}/outputs/paper_revision/robotwin_sim_policy_beat_hammer_e300_bf16_v1}
python_bin=${PYTHON_BIN:-/home/zheng/miniforge3/envs/RoboTwin/bin/python}
semantic_checkpoint=${SEMANTIC_CHECKPOINT:-${repo_root}/outputs/paper_revision/hammer_stage42_independent_local4090_seed20260805_e4000_split424242_20260806/hammer_full_fixed_e4000_split424242_seed20260805/best_sem.pt}
seed=${SEED:-20260805}
epochs=${EPOCHS:-300}

for lane in 0 1; do
    marker="${output_root}/lane_completion/local_lane${lane}.complete"
    until [[ -s "${marker}" ]]; do
        printf '[fixed-smoke] waiting for local lane %s completion\n' "${lane}"
        sleep 60
    done
done

# Let CUDA contexts and final checkpoint I/O settle before initializing SAPIEN.
sleep 30

run_smoke() {
    local route=$1
    local output_mode=$2
    local run_name="beat_block_hammer_${route}_joint14_e${epochs}_seed${seed}"
    local weights="${output_root}/${run_name}/${run_name}.weights.pt"
    local completion="${output_root}/${run_name}/${run_name}.completion-v2.json"
    if [[ ! -s "${weights}" || ! -s "${completion}" ]]; then
        echo "formal checkpoint or v2 completion is missing for ${route}" >&2
        exit 3
    fi

    printf '[fixed-smoke] route=%s checkpoint=%s\n' "${route}" "${weights}"
    env \
        CUDA_VISIBLE_DEVICES=0 \
        HYDRA_FULL_ERROR=1 \
        PYTHONWARNINGS=ignore::UserWarning \
        UTONIA_ROOT="${UTONIA_ROOT:-/home/zheng/github/Utonia}" \
        "${python_bin}" script/eval_policy.py \
        --config policy/DP3/deploy_policy.yml --overrides \
        --policy_name DP3 \
        --task_name beat_block_hammer \
        --task_config demo_clean_3d_object_pc_hammer_formal_fixed \
        --ckpt_setting "paper-sim-${route}-joint14-e${epochs}" \
        --expert_data_num 50 \
        --seed 0 \
        --test_num 2 \
        --instruction_type unseen \
        --config_name robot_dp3_semantic_pointwise_hybrid \
        --safe_checkpoint_path "${weights}" \
        --use_rgb False \
        --object_placeholders '{A},{B}' \
        --semantic_ckpt_A "${semantic_checkpoint}" \
        --semantic_ckpt_B none \
        --semantic_device cuda:0 \
        --semantic_point_num 128 \
        --semantic_input_color_mode debug_placeholder \
        --semantic_forward_mode reference \
        --semantic_policy_output_mode "${output_mode}" \
        --point_cloud_num 1024
}

cd "${repo_root}"
run_smoke xyz xyz
run_smoke partprob part_prob

mkdir -p "${output_root}/evaluation_completion"
printf 'routes=xyz,partprob completed_at_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    > "${output_root}/evaluation_completion/local_fixed2_smoke.complete"
