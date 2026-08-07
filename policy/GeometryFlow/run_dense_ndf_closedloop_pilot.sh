#!/usr/bin/env bash
set -euo pipefail

# Strictly paired clean/zero closed-loop evaluation for the fold-0 dense
# camera-relative NDF policy.  It defaults to the original one-seed deployment
# smoke test, while environment overrides permit a larger fixed-seed shard once
# that contract check has passed.

python_bin="${ROBOTWIN_PYTHON:-/home/zheng/miniforge3/envs/RoboTwin/bin/python}"
gpu="${DENSE_PILOT_GPU:-0}"
seed_file="${DENSE_PILOT_SEED_FILE:-outputs/geometry_flow/closedloop_fail_seed300001.json}"
policy_checkpoint="${DENSE_PILOT_POLICY:-outputs/geometry_flow/remote4090/dense_single_ndf_policy_seed0_normal/fold0_interaction_functional_direct_flow_seed0.pt}"
clean_policy_checkpoint="${DENSE_PILOT_CLEAN_POLICY:-${policy_checkpoint}}"
zero_policy_checkpoint="${DENSE_PILOT_ZERO_POLICY:-${policy_checkpoint}}"
camera_metadata="${DENSE_PILOT_CAMERA_METADATA:-outputs/geometry_flow/remote4090/task_flow_remote50_ndf_ensemble_camera_conf_v2.json}"
ensemble_metadata="${DENSE_PILOT_ENSEMBLE_METADATA:-outputs/geometry_flow/remote4090/ndf_functional_frame_dense_fold0_ensemble.json}"
log_root="${DENSE_PILOT_LOG_ROOT:-outputs/geometry_flow/remote4090/dense_ndf_closedloop_pilot_seed300001}"
test_num="${DENSE_PILOT_TEST_NUM:-1}"
ckpt_suffix="${DENSE_PILOT_CKPT_SUFFIX:-seed300001}"
temporal_ensemble_decay="${DENSE_PILOT_TEMPORAL_ENSEMBLE_DECAY:-0.0}"
execute_steps="${DENSE_PILOT_EXECUTE_STEPS:-1}"
modes_raw="${DENSE_PILOT_MODES:-clean zero}"
frame_checkpoint_override="${DENSE_PILOT_FRAME_CHECKPOINTS:-}"
if [[ -n "${frame_checkpoint_override}" ]]; then
  IFS=, read -r -a frame_checkpoints <<< "${frame_checkpoint_override}"
else
  frame_checkpoints=(
    outputs/geometry_flow/remote4090/ndf_functional_frame_dense_fold0/fold0_correct_frame_seed0_checkpoint.pth
    outputs/geometry_flow/remote4090/ndf_functional_frame_dense_fold0_seeds12/fold0_correct_frame_seed1_checkpoint.pth
    outputs/geometry_flow/remote4090/ndf_functional_frame_dense_fold0_seeds12/fold0_correct_frame_seed2_checkpoint.pth
  )
fi

[[ "${test_num}" =~ ^[1-9][0-9]*$ ]] || {
  printf 'DENSE_PILOT_TEST_NUM must be a positive integer, got %s\n' \
    "${test_num}" >&2
  exit 2
}
[[ -n "${ckpt_suffix}" ]] || {
  printf 'DENSE_PILOT_CKPT_SUFFIX must not be empty\n' >&2
  exit 2
}
[[ "${temporal_ensemble_decay}" =~ ^[0-9]+([.][0-9]+)?$ ]] || {
  printf 'DENSE_PILOT_TEMPORAL_ENSEMBLE_DECAY must be nonnegative, got %s\n' \
    "${temporal_ensemble_decay}" >&2
  exit 2
}
[[ "${execute_steps}" =~ ^[1-9][0-9]*$ ]] || {
  printf 'DENSE_PILOT_EXECUTE_STEPS must be a positive integer, got %s\n' \
    "${execute_steps}" >&2
  exit 2
}
read -r -a modes <<< "${modes_raw}"
[[ ${#modes[@]} -gt 0 ]] || {
  printf 'DENSE_PILOT_MODES resolved to an empty list\n' >&2
  exit 2
}
for mode in "${modes[@]}"; do
  [[ "${mode}" == "clean" || "${mode}" == "zero" || "${mode}" == "oracle" ]] || {
    printf 'DENSE_PILOT_MODES entries must be clean, zero, or oracle, got %s\n' \
      "${mode}" >&2
    exit 2
  }
done
[[ ${#frame_checkpoints[@]} -gt 0 ]] || {
  printf 'DENSE_PILOT_FRAME_CHECKPOINTS resolved to an empty list\n' >&2
  exit 2
}

required=(
  "${python_bin}"
  "${seed_file}"
  "${clean_policy_checkpoint}"
  "${zero_policy_checkpoint}"
  "${camera_metadata}"
  "${ensemble_metadata}"
  "${frame_checkpoints[@]}"
)
for path in "${required[@]}"; do
  [[ -e "${path}" ]] || {
    printf 'missing dense pilot artifact: %s\n' "${path}" >&2
    exit 2
  }
done

mkdir -p "${log_root}"
checkpoint_csv="$(IFS=,; printf '%s' "${frame_checkpoints[*]}")"
for mode in "${modes[@]}"; do
  mode_policy="${clean_policy_checkpoint}"
  if [[ "${mode}" == "zero" ]]; then
    mode_policy="${zero_policy_checkpoint}"
  fi
  allow_privileged_oracle=false
  if [[ "${mode}" == "oracle" ]]; then
    allow_privileged_oracle=true
  fi
  log_path="${log_root}/${mode}.log"
  [[ ! -e "${log_path}" ]] || {
    printf 'refuse to overwrite dense pilot log: %s\n' "${log_path}" >&2
    exit 2
  }
  printf 'start paired dense NDF pilot mode=%s seed_file=%s\n' \
    "${mode}" "${seed_file}"
  CUDA_VISIBLE_DEVICES="${gpu}" \
    OMP_NUM_THREADS=4 \
    MKL_NUM_THREADS=4 \
    PYTHONUNBUFFERED=1 \
    "${python_bin}" script/eval_policy.py \
    --config policy/GeometryFlow/deploy_interaction_flow.yml \
    --overrides \
    --interaction_flow_checkpoint "${mode_policy}" \
    --interaction_flow_device cuda:0 \
    --interaction_flow_observation_points 128 \
    --interaction_flow_execute_steps "${execute_steps}" \
    --interaction_flow_max_policy_steps 240 \
    --interaction_flow_temporal_ensemble_decay "${temporal_ensemble_decay}" \
    --interaction_flow_dense_frame_checkpoints "${checkpoint_csv}" \
    --interaction_flow_dense_camera_metadata "${camera_metadata}" \
    --interaction_flow_dense_ensemble_metadata "${ensemble_metadata}" \
    --interaction_flow_dense_frame_mode "${mode}" \
    --interaction_flow_allow_privileged_oracle "${allow_privileged_oracle}" \
    --evaluation_seed_file "${seed_file}" \
    --test_num "${test_num}" \
    --skip_expert_check_on_replay true \
    --policy_start_phase placement \
    --ckpt_setting "dense-ndf-fold0-${mode}-${ckpt_suffix}" \
    > "${log_path}" 2>&1
  printf 'complete paired dense NDF pilot mode=%s\n' "${mode}"
done

printf 'dense NDF closed-loop pilot complete modes=%s: %s\n' \
  "${modes_raw}" "${log_root}"
