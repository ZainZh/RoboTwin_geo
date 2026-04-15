#!/bin/bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
policy_dir="$(cd "${script_dir}/.." && pwd)"

tmpdir="$(mktemp -d)"
trap 'rm -rf "${tmpdir}"' EXIT

mkdir -p "${tmpdir}/bin"
cat > "${tmpdir}/bin/python" <<'EOF'
#!/bin/bash
printf '%s\n' "$@" > "${CAPTURE_FILE:?}"
EOF
chmod +x "${tmpdir}/bin/python"

mkdir -p "${policy_dir}/data/demo_task-demo_cfg-50-objpc-ndf-pointwise-hybrid-interact.zarr"

capture_file="$(mktemp "${tmpdir}/train_interact.XXXX")"
(
    cd "${policy_dir}"
    PATH="${tmpdir}/bin:${PATH}" \
    CAPTURE_FILE="${capture_file}" \
    bash train_ndf_pointwise_hybrid_interact.sh \
        demo_task \
        demo_cfg \
        50 \
        7 \
        3 \
        /tmp/a.pth \
        /tmp/b.pth \
        cuda:1 \
        "{A}" \
        "{A},{B}" \
        128 >/dev/null
)

assert_contains() {
    local needle="$1"
    grep -F -- "${needle}" "${capture_file}" >/dev/null
}

assert_contains "--config-name=robot_dp3_ndf_pointwise_hybrid_interact.yaml"
assert_contains "task_name=demo_task"
assert_contains "training.seed=7"
assert_contains "hydra.run.dir=data/outputs/demo_task-robot_dp3_ndf_pointwise_hybrid_interact-train_ndf_seed7"

script_file="${policy_dir}/train_ndf_pointwise_hybrid_interact.sh"
if grep -F -- "ndf_pointwise_arg_utils.sh" "${script_file}" >/dev/null; then
    echo "train interact script should not depend on ndf_pointwise_arg_utils.sh" >&2
    exit 1
fi
grep -F -- 'task_name=${1}' "${script_file}" >/dev/null
grep -F -- 'task_config=${2}' "${script_file}" >/dev/null
grep -F -- 'expert_data_num=${3}' "${script_file}" >/dev/null
grep -F -- 'seed=${4}' "${script_file}" >/dev/null
grep -F -- 'gpu_id=${5}' "${script_file}" >/dev/null
