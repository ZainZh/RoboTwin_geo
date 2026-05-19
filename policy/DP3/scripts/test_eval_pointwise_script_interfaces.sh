#!/bin/bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
policy_dir="$(cd "${script_dir}/.." && pwd)"
repo_root="$(cd "${policy_dir}/../.." && pwd)"

tmpdir="$(mktemp -d)"
trap 'rm -rf "${tmpdir}"' EXIT

mkdir -p "${tmpdir}/bin"
cat > "${tmpdir}/bin/python" <<'EOF'
#!/bin/bash
printf '%s\n' "$@" > "${CAPTURE_FILE:?}"
EOF
chmod +x "${tmpdir}/bin/python"

run_and_capture() {
    local capture_file="$1"
    shift
    (
        cd "${policy_dir}"
        PATH="${tmpdir}/bin:${PATH}" CAPTURE_FILE="${capture_file}" bash "$@" >/dev/null
    )
}

assert_flag_equals() {
    local capture_file="$1"
    local flag="$2"
    local expected="$3"
    local actual
    actual="$(awk -v flag="${flag}" '
        $0 == flag {
            getline
            print
            exit
        }
    ' "${capture_file}")"
    test "${actual}" = "${expected}"
}

capture_ndf="$(mktemp "${tmpdir}/ndf.XXXX")"
run_and_capture "${capture_ndf}" \
    eval_ndf_pointwise.sh \
    hanging_mug \
    demo_eval_cfg \
    demo_ckpt_cfg \
    50 \
    0 \
    0 \
    /tmp/mug_ndf.pth
assert_flag_equals "${capture_ndf}" "--task_config" "demo_eval_cfg"
assert_flag_equals "${capture_ndf}" "--ckpt_setting" "demo_ckpt_cfg"
assert_flag_equals "${capture_ndf}" "--point_cloud_num" "1024"

capture_ndf_hybrid="$(mktemp "${tmpdir}/ndf_hybrid.XXXX")"
run_and_capture "${capture_ndf_hybrid}" \
    eval_ndf_pointwise_hybrid.sh \
    hanging_mug \
    demo_eval_cfg \
    demo_ckpt_cfg \
    50 \
    0 \
    0 \
    /tmp/mug_ndf.pth
assert_flag_equals "${capture_ndf_hybrid}" "--task_config" "demo_eval_cfg"
assert_flag_equals "${capture_ndf_hybrid}" "--ckpt_setting" "demo_ckpt_cfg-objpc-ndf-pointwise-hybrid"
assert_flag_equals "${capture_ndf_hybrid}" "--point_cloud_num" "1024"

capture_ndf_actorseg="$(mktemp "${tmpdir}/ndf_actorseg.XXXX")"
run_and_capture "${capture_ndf_actorseg}" \
    eval_ndf_pointwise_actorseg_hybrid.sh \
    hanging_mug \
    demo_eval_cfg \
    demo_ckpt_cfg \
    50 \
    0 \
    0 \
    /tmp/mug_ndf.pth
assert_flag_equals "${capture_ndf_actorseg}" "--task_config" "demo_eval_cfg"
assert_flag_equals "${capture_ndf_actorseg}" "--ckpt_setting" "demo_ckpt_cfg"
assert_flag_equals "${capture_ndf_actorseg}" "--point_cloud_num" "1024"

capture_semantic="$(mktemp "${tmpdir}/semantic.XXXX")"
run_and_capture "${capture_semantic}" \
    eval_semantic_pointwise.sh \
    hanging_mug \
    demo_eval_cfg \
    demo_ckpt_cfg \
    50 \
    0 \
    0 \
    /tmp/mug_sem.ckpt
assert_flag_equals "${capture_semantic}" "--task_config" "demo_eval_cfg"
assert_flag_equals "${capture_semantic}" "--ckpt_setting" "demo_ckpt_cfg"
assert_flag_equals "${capture_semantic}" "--point_cloud_num" "1024"

capture_sem_actorseg="$(mktemp "${tmpdir}/semantic_actorseg.XXXX")"
run_and_capture "${capture_sem_actorseg}" \
    eval_semantic_pointwise_actorseg_hybrid.sh \
    hanging_mug \
    demo_eval_cfg \
    demo_ckpt_cfg \
    50 \
    0 \
    0 \
    /tmp/mug_sem.ckpt
assert_flag_equals "${capture_sem_actorseg}" "--task_config" "demo_eval_cfg"
assert_flag_equals "${capture_sem_actorseg}" "--ckpt_setting" "demo_ckpt_cfg"
assert_flag_equals "${capture_sem_actorseg}" "--point_cloud_num" "1024"

capture_semantic_hybrid_pc="$(mktemp "${tmpdir}/semantic_hybrid_pc.XXXX")"
run_and_capture "${capture_semantic_hybrid_pc}" \
    eval_semantic_pointwise_hybrid.sh \
    hanging_mug \
    demo_eval_cfg \
    demo_ckpt_cfg \
    50 \
    0 \
    0 \
    /tmp/mug_sem.ckpt \
    none \
    cuda:0 \
    "{A},{B}" \
    3000 \
    128 \
    2048
assert_flag_equals "${capture_semantic_hybrid_pc}" "--ckpt_setting" "demo_ckpt_cfg-objpc-semantic-pointwise-hybrid-pc2048"
assert_flag_equals "${capture_semantic_hybrid_pc}" "--point_cloud_num" "2048"
