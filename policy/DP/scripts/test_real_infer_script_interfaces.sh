#!/bin/bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
tmp_dir="$(mktemp -d)"
cleanup() {
    rm -rf "${tmp_dir}"
}
trap cleanup EXIT

fake_bin="${tmp_dir}/bin"
mkdir -p "${fake_bin}"
cat > "${fake_bin}/python" <<'PY'
#!/bin/bash
printf '%s\n' "$@" > "${REAL_DP_INFER_CAPTURE}"
PY
chmod +x "${fake_bin}/python"

capture_single="${tmp_dir}/single_args.txt"
REAL_DP_INFER_CAPTURE="${capture_single}" PATH="${fake_bin}:${PATH}" \
    bash "${repo_root}/policy/DP/real_infer.sh" \
    grasp_mug demo_real_zed_sam2_objpc 32 0 0 left 3000

grep -F -- 'script/real_zed_inference/real_dp_inference.py' "${capture_single}" >/dev/null
grep -F -- '--task_name' "${capture_single}" >/dev/null
grep -F -- 'grasp_mug' "${capture_single}" >/dev/null
grep -F -- '--ckpt_setting' "${capture_single}" >/dev/null
grep -F -- 'demo_real_zed_sam2_objpc-dp-left' "${capture_single}" >/dev/null
grep -F -- '--camera_labels' "${capture_single}" >/dev/null
grep -F -- 'left' "${capture_single}" >/dev/null
grep -F -- '--dp_camera_map' "${capture_single}" >/dev/null
grep -F -- 'head_cam:left' "${capture_single}" >/dev/null

capture_multi="${tmp_dir}/multi_args.txt"
REAL_DP_INFER_CAPTURE="${capture_multi}" PATH="${fake_bin}:${PATH}" \
    bash "${repo_root}/policy/DP/real_infer.sh" \
    grasp_mug demo_real_zed_sam2_objpc 32 0 0 global,left,right 3000

grep -F -- '--ckpt_setting' "${capture_multi}" >/dev/null
grep -F -- 'demo_real_zed_sam2_objpc-dp-global_left_right' "${capture_multi}" >/dev/null
grep -F -- '--camera_labels' "${capture_multi}" >/dev/null
grep -F -- 'global,left,right' "${capture_multi}" >/dev/null
grep -F -- '--dp_camera_map' "${capture_multi}" >/dev/null
grep -F -- 'head_cam:global,left_cam:left,right_cam:right' "${capture_multi}" >/dev/null
if grep -F -- '--execute' "${capture_multi}" >/dev/null; then
    echo "DP real inference wrapper should stay dry-run unless caller passes --execute" >&2
    exit 1
fi
