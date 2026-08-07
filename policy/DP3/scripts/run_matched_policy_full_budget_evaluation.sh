#!/usr/bin/env bash
set -euo pipefail

training_unit=${TRAINING_UNIT:-dp3-full-e3000-3seed-20260806.service}
epochs=${EPOCHS:-3000}
policy_seeds=${POLICY_SEEDS:-"20260805 20260806 20260807"}
eval_seed=${EVAL_SEED:-20260806}
gpu_id=${1:-0}

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd "${script_dir}/../../.." && pwd)
training_root=${TRAINING_ROOT:-"${repo_root}/outputs/paper_revision/policy_full_budget_safe"}
evaluation_root=${EVALUATION_ROOT:-"${repo_root}/outputs/paper_revision/policy_full_budget_offline_eval"}
mkdir -p "${evaluation_root}"

while systemctl --user is-active --quiet "${training_unit}"; do
    sleep 60
done

read -r -a seeds <<< "${policy_seeds}"
routes=(field xyz partprob)
expected_count=$((${#seeds[@]} * ${#routes[@]}))
actual_count=$(find "${training_root}" -type f -name "beat_cube_*_e${epochs}_seed*.weights.pt" | wc -l)
if (( actual_count != expected_count )); then
    echo "[full-budget-eval] expected ${expected_count} weights, found ${actual_count}; training was incomplete" >&2
    exit 4
fi

for seed in "${seeds[@]}"; do
    for route in "${routes[@]}"; do
        run_dir="${training_root}/beat_cube_${route}_e${epochs}_seed${seed}"
        output_path="${evaluation_root}/beat_cube_${route}_e${epochs}_seed${seed}.json"
        if [[ -s "${output_path}" ]]; then
            echo "[full-budget-eval] skip completed ${output_path}"
            continue
        fi
        CUDA_VISIBLE_DEVICES="${gpu_id}" python "${script_dir}/evaluate_dp3_checkpoint_offline.py" \
            --config "${run_dir}/.hydra/config.yaml" \
            --checkpoint "${run_dir}/beat_cube_${route}_e${epochs}_seed${seed}.weights.pt" \
            --output "${output_path}" \
            --route "${route}" \
            --device cuda:0 \
            --batch_size 256 \
            --num_workers 0 \
            --repeats 20 \
            --eval_seed "${eval_seed}"
    done
done

python "${script_dir}/summarize_matched_policy_offline_eval.py" \
    --input_dir "${evaluation_root}" \
    --output_json "${evaluation_root}/beat_cube_e${epochs}_three_seed_summary.json" \
    --output_csv "${evaluation_root}/beat_cube_e${epochs}_three_seed_summary.csv" \
    --epochs "${epochs}" \
    --seeds "${seeds[@]}"

echo "[full-budget-eval] matrix complete"
