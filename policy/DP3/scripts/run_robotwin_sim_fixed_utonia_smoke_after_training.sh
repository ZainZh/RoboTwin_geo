#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd "${script_dir}/../../.." && pwd)
output_root=${OUTPUT_ROOT:-${repo_root}/outputs/paper_revision/robotwin_sim_policy_beat_hammer_e300_bf16_v1}
python_bin=${PYTHON_BIN:-/home/zheng/miniforge3/envs/RoboTwin/bin/python}
semantic_checkpoint=${SEMANTIC_CHECKPOINT:-${repo_root}/outputs/paper_revision/hammer_stage42_independent_local4090_seed20260805_e4000_split424242_20260806/hammer_full_fixed_e4000_split424242_seed20260805/best_sem.pt}
utonia_checkpoint=${UTONIA_CHECKPOINT:-/home/zheng/.cache/utonia/ckpt/utonia.pth}
seed=${SEED:-20260805}
epochs=${EPOCHS:-300}

fixed2_marker="${output_root}/evaluation_completion/local_fixed2_smoke.complete"
until [[ -s "${fixed2_marker}" ]]; do
    printf '[fixed-utonia-smoke] waiting for XYZ/part-prob fixed smoke\n'
    sleep 60
done

run_name="beat_block_hammer_utonia_joint14_e${epochs}_seed${seed}"
weights="${output_root}/${run_name}/${run_name}.weights.pt"
completion="${output_root}/${run_name}/${run_name}.completion-v2.json"
for required in "${weights}" "${completion}" "${semantic_checkpoint}" "${utonia_checkpoint}"; do
    if [[ ! -s "${required}" ]]; then
        echo "required formal artifact is missing: ${required}" >&2
        exit 3
    fi
done

printf '[fixed-utonia-smoke] policy_sha256=%s utonia_sha256=%s\n' \
    "$(sha256sum "${weights}" | awk '{print $1}')" \
    "$(sha256sum "${utonia_checkpoint}" | awk '{print $1}')"

cd "${repo_root}"
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
    --ckpt_setting "paper-sim-utonia-joint14-e${epochs}" \
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
    --semantic_policy_output_mode matched_utonia \
    --utonia_checkpoint "${utonia_checkpoint}" \
    --utonia_feature_placeholders '{A}' \
    --utonia_device cuda:0 \
    --matched_utonia_color_mode debug_placeholder \
    --matched_utonia_normal_mode fallback \
    --point_cloud_num 1024

mkdir -p "${output_root}/evaluation_completion"
printf 'routes=utonia test_num=2 completed_at_utc=%s\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    > "${output_root}/evaluation_completion/local_fixed2_utonia_smoke.complete"
