#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd "${script_dir}/../../.." && pwd)
python_bin=${PYTHON_BIN:-/home/zheng/miniforge3/envs/RoboTwin/bin/python}
data_base="${repo_root}/policy/DP3/data/hanging_mug-demo_clean_3d_object_pc-50-objpc-semantic-pointwise-hybrid-sim-paper-v1"
field_zarr="${data_base}.zarr"
field_meta="${data_base}_meta.json"
semantic_checkpoint=${SEMANTIC_CHECKPOINT:-${repo_root}/include/3d_semantic_train/outputs/utonia_universal_field/Mug_semantic/best.pt}
raw_data_dir=${RAW_DATA_DIR:-${repo_root}/data/hanging_mug/demo_clean_3d_object_pc}

field_complete() {
    "${python_bin}" -c '
import json, sys, zarr
meta_path, zarr_path = sys.argv[1:]
with open(meta_path, "r", encoding="utf-8") as handle:
    meta = json.load(handle)
root = zarr.open(zarr_path, mode="r")
assert len(meta.get("episodes", [])) == 50
assert len(root["meta"]["episode_ends"][:]) == 50
assert int(root["meta"]["episode_ends"][-1]) == 16942
' "${field_meta}" "${field_zarr}" >/dev/null 2>&1
}

until field_complete; do
    printf '[hanging-data-successor] waiting for complete field zarr\n'
    sleep 60
done

run_control() {
    local source=$1
    local output=$2
    local mode=$3
    shift 3
    "${python_bin}" "${script_dir}/build_semantic_policy_ablation_zarr.py" \
        --source-zarr "${source}" \
        --output-zarr "${output}" \
        --mode "${mode}" \
        "$@"
}

run_control "${field_zarr}" "${data_base}-xyz.zarr" xyz &
xyz_pid=$!
run_control "${field_zarr}" "${data_base}-partprob.zarr" part_prob \
    --semantic-checkpoint "${semantic_checkpoint}" &
partprob_pid=$!
env UTONIA_ROOT="${UTONIA_ROOT:-/home/zheng/github/Utonia}" \
    "${python_bin}" "${script_dir}/build_matched_utonia_policy_zarr.py" \
    --source-zarr "${field_zarr}" \
    --output-zarr "${data_base}-utonia.zarr" \
    --source-meta "${field_meta}" \
    --placeholder '{A}' \
    --support-source raw \
    --raw-data-dir "${raw_data_dir}" \
    --raw-support-points 1024 \
    --color-mode debug_placeholder \
    --normal-mode fallback \
    --utonia-checkpoint auto \
    --device cuda:0 \
    --seed 20260805 &
utonia_pid=$!

partprob_status=0
wait "${partprob_pid}" || partprob_status=$?
if (( partprob_status == 0 )); then
    run_control "${data_base}-partprob.zarr" "${data_base}-uniformprob.zarr" uniform_prob &
    uniform_pid=$!
    run_control "${data_base}-partprob.zarr" "${data_base}-shuffledprob.zarr" shuffled_prob \
        --shuffle-seed 20260807 &
    shuffled_pid=$!
else
    uniform_pid=""
    shuffled_pid=""
fi

status=${partprob_status}
for pid in "${xyz_pid}" "${utonia_pid}" "${uniform_pid}" "${shuffled_pid}"; do
    [[ -n "${pid}" ]] || continue
    wait "${pid}" || status=1
done
if (( status != 0 )); then
    echo "one or more Hanging Mug matched-data builders failed" >&2
    exit "${status}"
fi

completion_dir="${repo_root}/outputs/paper_revision/robotwin_sim_policy_hanging_mug_v1"
mkdir -p "${completion_dir}"
printf 'frames=16942 routes=field,xyz,partprob,uniformprob,shuffledprob,utonia completed_at_utc=%s\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "${completion_dir}/matched_data.complete"
