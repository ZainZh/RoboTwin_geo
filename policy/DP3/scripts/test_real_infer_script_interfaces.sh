#!/bin/bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
tmp_dir="$(mktemp -d)"
cleanup() {
    rm -rf "${tmp_dir}"
    rm -rf "${repo_root}/data/unit_real_zed_task"
}
trap cleanup EXIT

mkdir -p "${repo_root}/data/unit_real_zed_task/unit_real_zed_cfg"
printf '{"output_frame": "right_base"}\n' > "${repo_root}/data/unit_real_zed_task/unit_real_zed_cfg/real_zed_sam2_objpc_meta.json"

fake_bin="${tmp_dir}/bin"
mkdir -p "${fake_bin}"
cat > "${fake_bin}/python" <<'PY'
#!/bin/bash
printf '%s\n' "$@" > "${REAL_INFER_CAPTURE}"
PY
chmod +x "${fake_bin}/python"

capture_baseline="${tmp_dir}/baseline_args.txt"
REAL_INFER_CAPTURE="${capture_baseline}" PATH="${fake_bin}:${PATH}" \
    bash "${repo_root}/policy/DP3/real_infer_baseline.sh" \
    unit_real_zed_task unit_real_zed_cfg 31 0 0 3000

grep -F -- '--output_frame' "${capture_baseline}" >/dev/null
grep -F -- 'right_base' "${capture_baseline}" >/dev/null
grep -F -- '--robot_camera_calibration_path' "${capture_baseline}" >/dev/null
grep -F -- 'script/real_zed_collection/calibration/robot_camera_apriltag_right_global.yaml' "${capture_baseline}" >/dev/null

capture_semantic="${tmp_dir}/semantic_args.txt"
REAL_INFER_CAPTURE="${capture_semantic}" PATH="${fake_bin}:${PATH}" \
    bash "${repo_root}/policy/DP3/real_infer_semantic_pointwise_hybrid.sh" \
    unit_real_zed_task unit_real_zed_cfg 31 0 0 \
    /tmp/unit_semantic_A.pt none cuda:0 "{A},{B}" 3000 128

grep -F -- '--task_config' "${capture_semantic}" >/dev/null
grep -F -- 'unit_real_zed_cfg' "${capture_semantic}" >/dev/null
grep -F -- '--ckpt_setting' "${capture_semantic}" >/dev/null
grep -F -- 'unit_real_zed_cfg-objpc-semantic-pointwise-hybrid' "${capture_semantic}" >/dev/null
grep -F -- '--output_frame' "${capture_semantic}" >/dev/null
grep -F -- 'right_base' "${capture_semantic}" >/dev/null
grep -F -- '--robot_camera_calibration_path' "${capture_semantic}" >/dev/null
grep -F -- 'script/real_zed_collection/calibration/robot_camera_apriltag_right_global.yaml' "${capture_semantic}" >/dev/null

capture_explicit="${tmp_dir}/explicit_args.txt"
REAL_INFER_CAPTURE="${capture_explicit}" PATH="${fake_bin}:${PATH}" \
    bash "${repo_root}/policy/DP3/real_infer_baseline.sh" \
    unit_real_zed_task unit_real_zed_cfg 31 0 0 3000 workspace none --dry_run

grep -F -- '--output_frame' "${capture_explicit}" >/dev/null
grep -F -- 'workspace' "${capture_explicit}" >/dev/null
if grep -F -- '--robot_camera_calibration_path' "${capture_explicit}" >/dev/null; then
    echo "workspace output_frame should not pass robot camera calibration by default" >&2
    exit 1
fi
grep -F -- '--dry_run' "${capture_explicit}" >/dev/null

capture_name_fallback="${tmp_dir}/name_fallback_args.txt"
REAL_INFER_CAPTURE="${capture_name_fallback}" PATH="${fake_bin}:${PATH}" \
    bash "${repo_root}/policy/DP3/real_infer_baseline.sh" \
    unit_real_zed_task unit_cfg_rightbase 31 0 0 3000

grep -F -- '--output_frame' "${capture_name_fallback}" >/dev/null
grep -F -- 'right_base' "${capture_name_fallback}" >/dev/null

capture_dashed_frame="${tmp_dir}/dashed_frame_args.txt"
REAL_INFER_CAPTURE="${capture_dashed_frame}" PATH="${fake_bin}:${PATH}" \
    bash "${repo_root}/policy/DP3/real_infer_semantic_pointwise_hybrid.sh" \
    unit_real_zed_task unit_real_zed_cfg 31 0 0 \
    /tmp/unit_semantic_A.pt none cuda:0 "{A},{B}" 3000 128 --right_base --auto --dry_run

grep -F -- '--output_frame' "${capture_dashed_frame}" >/dev/null
grep -F -- 'right_base' "${capture_dashed_frame}" >/dev/null
grep -F -- '--right_base' "${capture_dashed_frame}" && {
    echo "dashed frame shorthand should be normalized before calling Python" >&2
    exit 1
}
grep -F -- '--auto' "${capture_dashed_frame}" && {
    echo "dashed auto shorthand should be normalized before calling Python" >&2
    exit 1
}
grep -F -- '--dry_run' "${capture_dashed_frame}" >/dev/null
