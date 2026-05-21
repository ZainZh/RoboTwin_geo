#!/bin/bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
tmp_dir="$(mktemp -d)"
cleanup() {
    rm -rf "${tmp_dir}"
    rm -rf "${repo_root}/policy/DP/data/unit_task-demo_cfg-dp-global-49.zarr"
}
trap cleanup EXIT

fake_bin="${tmp_dir}/bin"
mkdir -p "${fake_bin}"
cat > "${fake_bin}/bash" <<'FAKE_BASH'
#!/bin/bash
printf '%s\n' "$@" > "${DP_REAL_ZED_PREPROCESS_CAPTURE}"
exit 0
FAKE_BASH
cat > "${fake_bin}/python" <<'FAKE_PYTHON'
#!/bin/bash
printf '%s\n' "$@" > "${DP_REAL_ZED_TRAIN_CAPTURE}"
exit 0
FAKE_PYTHON
chmod +x "${fake_bin}/bash" "${fake_bin}/python"

capture_preprocess="${tmp_dir}/preprocess_args.txt"
capture_train="${tmp_dir}/train_args.txt"
(
    cd "${repo_root}/policy/DP"
    DP_REAL_ZED_PREPROCESS_CAPTURE="${capture_preprocess}" \
    DP_REAL_ZED_TRAIN_CAPTURE="${capture_train}" \
    PATH="${fake_bin}:${PATH}" \
        /bin/bash train_real_zed.sh unit_task demo_cfg 49 0 14 0 global Large_L515 true
)

grep -F -- 'process_data_real_zed.sh' "${capture_preprocess}" >/dev/null
grep -F -- 'unit_task' "${capture_preprocess}" >/dev/null
grep -F -- 'demo_cfg' "${capture_preprocess}" >/dev/null
grep -F -- '49' "${capture_preprocess}" >/dev/null
grep -F -- 'global' "${capture_preprocess}" >/dev/null
grep -F -- '360,640' "${capture_preprocess}" >/dev/null

grep -F -- 'train.py' "${capture_train}" >/dev/null
grep -F -- 'head_camera_type=Large_L515' "${capture_train}" >/dev/null

capture_multi_preprocess="${tmp_dir}/multi_preprocess_args.txt"
capture_multi_train="${tmp_dir}/multi_train_args.txt"
(
    cd "${repo_root}/policy/DP"
    DP_REAL_ZED_PREPROCESS_CAPTURE="${capture_multi_preprocess}" \
    DP_REAL_ZED_TRAIN_CAPTURE="${capture_multi_train}" \
    PATH="${fake_bin}:${PATH}" \
        /bin/bash train_real_zed_multicam.sh unit_task demo_cfg 49 0 14 0 global,left,right Large_L515 true
)

grep -F -- 'process_data_real_zed_multicam.sh' "${capture_multi_preprocess}" >/dev/null
grep -F -- 'global,left,right' "${capture_multi_preprocess}" >/dev/null
grep -F -- '360,640' "${capture_multi_preprocess}" >/dev/null
grep -F -- 'train.py' "${capture_multi_train}" >/dev/null
grep -F -- 'task=default_task_14_multicam' "${capture_multi_train}" >/dev/null

capture_eef_preprocess="${tmp_dir}/eef_preprocess_args.txt"
capture_eef_train="${tmp_dir}/eef_train_args.txt"
(
    cd "${repo_root}/policy/DP"
    DP_REAL_ZED_PREPROCESS_CAPTURE="${capture_eef_preprocess}" \
    DP_REAL_ZED_TRAIN_CAPTURE="${capture_eef_train}" \
    PATH="${fake_bin}:${PATH}" \
        /bin/bash train_real_zed_eef_absolute6d_global.sh unit_task demo_cfg 49 0 0 global Large_L515 true
)

grep -F -- 'process_data_real_zed.sh' "${capture_eef_preprocess}" >/dev/null
grep -F -- '--output_zarr' "${capture_eef_preprocess}" >/dev/null
grep -F -- 'data/unit_task-demo_cfg-dp-global-eef-absolute6d-global-49.zarr' "${capture_eef_preprocess}" >/dev/null
grep -F -- '--action_mode' "${capture_eef_preprocess}" >/dev/null
grep -F -- 'eef_absolute6d' "${capture_eef_preprocess}" >/dev/null
grep -F -- '--eef_frame_mode' "${capture_eef_preprocess}" >/dev/null
grep -F -- 'reference_camera' "${capture_eef_preprocess}" >/dev/null

grep -F -- 'train.py' "${capture_eef_train}" >/dev/null
grep -F -- '--config-name=robot_dp_20.yaml' "${capture_eef_train}" >/dev/null
grep -F -- 'task.dataset.zarr_path=data/unit_task-demo_cfg-dp-global-eef-absolute6d-global-49.zarr' "${capture_eef_train}" >/dev/null
grep -F -- 'setting=demo_cfg-dp-global-eef-absolute6d-global' "${capture_eef_train}" >/dev/null

capture_multi_eef_preprocess="${tmp_dir}/multi_eef_preprocess_args.txt"
capture_multi_eef_train="${tmp_dir}/multi_eef_train_args.txt"
(
    cd "${repo_root}/policy/DP"
    DP_REAL_ZED_PREPROCESS_CAPTURE="${capture_multi_eef_preprocess}" \
    DP_REAL_ZED_TRAIN_CAPTURE="${capture_multi_eef_train}" \
    PATH="${fake_bin}:${PATH}" \
        /bin/bash train_real_zed_multicam_eef_absolute6d_global.sh unit_task demo_cfg 49 0 0 global,left,right Large_L515 true
)

grep -F -- 'process_data_real_zed_multicam.sh' "${capture_multi_eef_preprocess}" >/dev/null
grep -F -- 'global,left,right' "${capture_multi_eef_preprocess}" >/dev/null
grep -F -- '--action_mode' "${capture_multi_eef_preprocess}" >/dev/null
grep -F -- 'eef_absolute6d' "${capture_multi_eef_preprocess}" >/dev/null
grep -F -- 'data/unit_task-demo_cfg-dp-global_left_right-eef-absolute6d-global-49.zarr' "${capture_multi_eef_preprocess}" >/dev/null

grep -F -- 'train.py' "${capture_multi_eef_train}" >/dev/null
grep -F -- '--config-name=robot_dp_20.yaml' "${capture_multi_eef_train}" >/dev/null
grep -F -- 'task=default_task_20_multicam' "${capture_multi_eef_train}" >/dev/null
grep -F -- 'setting=demo_cfg-dp-global_left_right-eef-absolute6d-global' "${capture_multi_eef_train}" >/dev/null
