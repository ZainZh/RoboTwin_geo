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

run_and_assert() {
    local script_name="$1"
    local zarr_dir="$2"
    shift 2
    local capture_file
    capture_file="$(mktemp "${tmpdir}/${script_name}.XXXX")"

    mkdir -p "${policy_dir}/${zarr_dir}"

    (
        cd "${policy_dir}"
        PATH="${tmpdir}/bin:${PATH}" \
        CAPTURE_FILE="${capture_file}" \
        bash "${script_name}" "$@" >/dev/null
    )

    grep -F -- "training.gradient_accumulate_every=3" "${capture_file}" >/dev/null
    grep -F -- "dataloader.batch_size=64" "${capture_file}" >/dev/null
    grep -F -- "val_dataloader.batch_size=96" "${capture_file}" >/dev/null
}

run_and_assert \
    train_ndf_pointwise_hybrid.sh \
    data/demo_task-demo_cfg-50-objpc-ndf-pointwise-hybrid.zarr \
    demo_task \
    demo_cfg \
    50 \
    7 \
    3 \
    /tmp/a.pth \
    /tmp/b.pth \
    cuda:1 \
    "{A},{B}" \
    128 \
    256 \
    64 \
    96 \
    false \
    3

run_and_assert \
    train_ndf_pointwise_hybrid_interact.sh \
    data/demo_task-demo_cfg-50-objpc-ndf-pointwise-hybrid-interact.zarr \
    demo_task \
    demo_cfg \
    50 \
    7 \
    3 \
    /tmp/a.pth \
    /tmp/b.pth \
    cuda:1 \
    "" \
    "{A},{B}" \
    128 \
    64 \
    96 \
    3

run_and_assert \
    train_ndf_pointwise_hybrid_feat5000.sh \
    data/demo_task-demo_cfg-50-objpc-ndf-pointwise-hybrid-feat512.zarr \
    demo_task \
    demo_cfg \
    50 \
    7 \
    3 \
    /tmp/a.pth \
    /tmp/b.pth \
    cuda:1 \
    "" \
    "{A},{B}" \
    512 \
    64 \
    96 \
    3
