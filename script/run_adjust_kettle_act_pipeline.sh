#!/usr/bin/env bash
set -Eeuo pipefail

TASK_NAME="${TASK_NAME:-adjust_kettle}"
MID_CONFIG="${MID_CONFIG:-demo_clean_midrange}"
EXPANDED_CONFIG="${EXPANDED_CONFIG:-demo_clean}"
RUN_CONFIGS="${RUN_CONFIGS:-${MID_CONFIG} ${EXPANDED_CONFIG}}"
EXPERT_DATA_NUM="${EXPERT_DATA_NUM:-50}"
SEED="${SEED:-0}"
GPU_ID="${GPU_ID:-0}"

SSH_KEY="${SSH_KEY:-/tmp/robotwin_cloud_key}"
REMOTE_HOST="${REMOTE_HOST:-root@183.147.142.40}"
REMOTE_PORT="${REMOTE_PORT:-30335}"
REMOTE_ROOT="${REMOTE_ROOT:-/root/gpufree-data/robotwin_work/RoboTwin_geo}"
REMOTE_PYDEPS="${REMOTE_PYDEPS:-/root/gpufree-data/robotwin_work/pydeps}"

RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
LOCAL_LOG_ROOT="${LOCAL_LOG_ROOT:-logs/adjust_kettle_act_pipeline/${RUN_ID}}"
REMOTE_LOG_ROOT="${REMOTE_LOG_ROOT:-${REMOTE_ROOT}/pipeline_logs/adjust_kettle_act_pipeline/${RUN_ID}}"
CLEAN_PROCESSED_AFTER_TRAIN="${CLEAN_PROCESSED_AFTER_TRAIN:-1}"
FORCE_RETRAIN="${FORCE_RETRAIN:-1}"
POLL_SECONDS="${POLL_SECONDS:-60}"

mkdir -p "${LOCAL_LOG_ROOT}"
LOCAL_LOG="${LOCAL_LOG_ROOT}/pipeline.log"
STATUS_FILE="${LOCAL_LOG_ROOT}/status.tsv"
touch "${STATUS_FILE}"
exec > >(tee -a "${LOCAL_LOG}") 2>&1

SSH_OPTS=(
  -i "${SSH_KEY}"
  -o BatchMode=yes
  -o ServerAliveInterval=30
  -o ServerAliveCountMax=6
  -p "${REMOTE_PORT}"
)

log() {
  printf '[%s] %s\n' "$(date '+%F %T')" "$*"
}

status() {
  printf '%s\t%s\t%s\n' "$(date '+%F %T')" "$1" "$2" | tee -a "${STATUS_FILE}" >/dev/null
}

remote() {
  ssh "${SSH_OPTS[@]}" "${REMOTE_HOST}" "$@"
}

require_local_dataset() {
  local config="$1"
  local path="data/${TASK_NAME}/${config}/data"
  if [[ ! -d "${path}" ]]; then
    log "Missing local dataset directory: ${path}"
    exit 1
  fi

  local count
  count="$(find "${path}" -maxdepth 1 -type f -name 'episode*.hdf5' | wc -l)"
  if [[ "${count}" -lt "${EXPERT_DATA_NUM}" ]]; then
    log "Dataset ${config} has ${count} episodes, need ${EXPERT_DATA_NUM}"
    exit 1
  fi
  log "Local dataset ${config}: ${count} episode files"
}

upload_dataset() {
  local config="$1"
  local local_path="data/${TASK_NAME}/${config}"
  local remote_path="${REMOTE_ROOT}/data/${TASK_NAME}/${config}"

  status "${config}" "upload_start"
  log "Cleaning remote raw dataset: ${remote_path}"
  remote "rm -rf '${remote_path}' && mkdir -p '${REMOTE_ROOT}/data/${TASK_NAME}'"

  log "Uploading ${local_path} to ${REMOTE_HOST}:${remote_path}"
  tar -C . -czf - "${local_path}" | remote "cd '${REMOTE_ROOT}' && tar -xzf -"

  log "Verifying remote raw dataset ${config}"
  remote "test -d '${remote_path}/data' && count=\$(find '${remote_path}/data' -maxdepth 1 -type f -name 'episode*.hdf5' | wc -l) && du -sh '${remote_path}' && test \"\$count\" -ge '${EXPERT_DATA_NUM}'"
  status "${config}" "upload_done"
}

write_remote_stage_script() {
  local config="$1"
  local stage_dir="${REMOTE_LOG_ROOT}/${config}-${EXPERT_DATA_NUM}"
  local remote_script="${stage_dir}/run_stage.sh"
  local tmp_script

  tmp_script="$(mktemp)"
  cat > "${tmp_script}" <<EOF
#!/usr/bin/env bash
set -Eeuo pipefail

TASK_NAME="${TASK_NAME}"
CONFIG="${config}"
EXPERT_DATA_NUM="${EXPERT_DATA_NUM}"
SEED="${SEED}"
GPU_ID="${GPU_ID}"
REMOTE_ROOT="${REMOTE_ROOT}"
REMOTE_PYDEPS="${REMOTE_PYDEPS}"
CLEAN_PROCESSED_AFTER_TRAIN="${CLEAN_PROCESSED_AFTER_TRAIN}"
FORCE_RETRAIN="${FORCE_RETRAIN}"
STATUS_FILE="${stage_dir}/stage.status"

mark_failed() {
  local code="\$?"
  echo "failed:\${code}" > "\${STATUS_FILE}"
  exit "\${code}"
}
trap mark_failed ERR

echo "running" > "\${STATUS_FILE}"
cd "\${REMOTE_ROOT}/policy/ACT"
export PYTHONPATH="\${REMOTE_PYDEPS}:\${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="\${GPU_ID}"

echo "[\$(date '+%F %T')] python=\$(command -v python3)"
python3 - <<'PY'
import cv2, h5py, numpy, torch, torchvision, tqdm, einops, matplotlib, IPython
print("deps ok", numpy.__version__, torch.cuda.is_available())
PY

if [[ "\${FORCE_RETRAIN}" == "1" ]]; then
  rm -rf "processed_data/sim-\${TASK_NAME}/\${CONFIG}-\${EXPERT_DATA_NUM}"
  rm -rf "act_ckpt/act-\${TASK_NAME}/\${CONFIG}-\${EXPERT_DATA_NUM}"
fi

echo "[\$(date '+%F %T')] process_data \${TASK_NAME} \${CONFIG} \${EXPERT_DATA_NUM}"
python3 process_data.py "\${TASK_NAME}" "\${CONFIG}" "\${EXPERT_DATA_NUM}"

echo "[\$(date '+%F %T')] train ACT \${TASK_NAME} \${CONFIG} \${EXPERT_DATA_NUM}"
bash train.sh "\${TASK_NAME}" "\${CONFIG}" "\${EXPERT_DATA_NUM}" "\${SEED}" "\${GPU_ID}"

test -f "act_ckpt/act-\${TASK_NAME}/\${CONFIG}-\${EXPERT_DATA_NUM}/policy_best.ckpt"

if [[ "\${CLEAN_PROCESSED_AFTER_TRAIN}" == "1" ]]; then
  rm -rf "processed_data/sim-\${TASK_NAME}/\${CONFIG}-\${EXPERT_DATA_NUM}"
fi

df -h "\${REMOTE_ROOT}" || true
echo "success" > "\${STATUS_FILE}"
echo "[\$(date '+%F %T')] stage success"
EOF

  remote "mkdir -p '${stage_dir}' && cat > '${remote_script}' && chmod +x '${remote_script}'" < "${tmp_script}"
  rm -f "${tmp_script}"
  printf '%s\n' "${remote_script}"
}

run_remote_stage() {
  local config="$1"
  local stage_dir="${REMOTE_LOG_ROOT}/${config}-${EXPERT_DATA_NUM}"
  local remote_script
  local remote_log="${stage_dir}/stage.log"
  local remote_pid="${stage_dir}/stage.pid"
  local remote_status="${stage_dir}/stage.status"

  status "${config}" "remote_stage_prepare"
  remote_script="$(write_remote_stage_script "${config}")"

  log "Starting remote ACT stage for ${config}; log: ${remote_log}"
  remote "cd '${stage_dir}' && rm -f '${remote_status}' '${remote_pid}' && nohup '${remote_script}' > '${remote_log}' 2>&1 < /dev/null & echo \$! > '${remote_pid}'"

  status "${config}" "remote_stage_running"
  while true; do
    local state
    state="$(remote "cat '${remote_status}' 2>/dev/null || true")"
    case "${state}" in
      success)
        status "${config}" "remote_stage_success"
        log "Remote ACT stage succeeded for ${config}"
        break
        ;;
      failed:*)
        status "${config}" "remote_stage_failed"
        log "Remote ACT stage failed for ${config}; tailing remote log"
        remote "tail -n 120 '${remote_log}'"
        exit 1
        ;;
      *)
        log "Remote ACT stage still running for ${config}; remote log: ${remote_log}"
        sleep "${POLL_SECONDS}"
        ;;
    esac
  done
}

main() {
  log "Pipeline run id: ${RUN_ID}"
  log "Task: ${TASK_NAME}; configs=${RUN_CONFIGS}; episodes=${EXPERT_DATA_NUM}; gpu=${GPU_ID}"
  log "Local log: ${LOCAL_LOG}"
  log "Remote root: ${REMOTE_ROOT}"

  command -v ssh >/dev/null
  command -v tar >/dev/null
  test -f "${SSH_KEY}"

  for config in ${RUN_CONFIGS}; do
    require_local_dataset "${config}"
  done

  remote "mkdir -p '${REMOTE_LOG_ROOT}' && test -d '${REMOTE_ROOT}/policy/ACT' && test -d '${REMOTE_PYDEPS}'"
  remote "PYTHONPATH='${REMOTE_PYDEPS}' python3 -c 'import h5py, cv2, numpy, torch; print(\"remote deps ok\", torch.cuda.is_available())'"

  for config in ${RUN_CONFIGS}; do
    upload_dataset "${config}"
    run_remote_stage "${config}"
  done

  status "pipeline" "success"
  log "Pipeline completed successfully"
}

main "$@"
