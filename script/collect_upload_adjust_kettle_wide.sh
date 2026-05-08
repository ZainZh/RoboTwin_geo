#!/usr/bin/env bash
set -Eeuo pipefail

TASK_NAME="${TASK_NAME:-adjust_kettle}"
TASK_CONFIG="${TASK_CONFIG:-demo_clean_wide}"
EXPERT_DATA_NUM="${EXPERT_DATA_NUM:-50}"

SSH_KEY="${SSH_KEY:-/tmp/robotwin_cloud_key}"
REMOTE_HOST="${REMOTE_HOST:-root@183.147.142.40}"
REMOTE_PORT="${REMOTE_PORT:-30335}"
REMOTE_ROOT="${REMOTE_ROOT:-/root/gpufree-data/robotwin_work/RoboTwin_geo}"

RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
LOG_ROOT="${LOG_ROOT:-logs/adjust_kettle_collect_upload/${RUN_ID}}"
LOG_FILE="${LOG_ROOT}/collect_upload.log"
STATUS_FILE="${LOG_ROOT}/status.tsv"

mkdir -p "${LOG_ROOT}"
touch "${STATUS_FILE}"
exec > >(tee -a "${LOG_FILE}") 2>&1

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
  printf '%s\t%s\n' "$(date '+%F %T')" "$1" | tee -a "${STATUS_FILE}" >/dev/null
}

remote() {
  ssh "${SSH_OPTS[@]}" "${REMOTE_HOST}" "$@"
}

main() {
  log "Run id: ${RUN_ID}"
  log "Collecting ${TASK_NAME}/${TASK_CONFIG}, episodes=${EXPERT_DATA_NUM}"
  log "Log: ${LOG_FILE}"

  test -f "${SSH_KEY}"
  test -f "task_config/${TASK_CONFIG}.yml"
  test -f "envs/${TASK_NAME}.py"

  status "collect_start"
  python3 script/collect_data.py "${TASK_NAME}" "${TASK_CONFIG}"

  local data_dir="data/${TASK_NAME}/${TASK_CONFIG}/data"
  local count
  count="$(find "${data_dir}" -maxdepth 1 -type f -name 'episode*.hdf5' | wc -l)"
  log "Collected hdf5 count: ${count}"
  if [[ "${count}" -lt "${EXPERT_DATA_NUM}" ]]; then
    log "Not enough episodes: ${count}/${EXPERT_DATA_NUM}"
    exit 1
  fi
  status "collect_done"

  status "upload_start"
  remote "rm -rf '${REMOTE_ROOT}/data/${TASK_NAME}/${TASK_CONFIG}' && mkdir -p '${REMOTE_ROOT}/data/${TASK_NAME}'"
  tar -C . -czf - \
    "data/${TASK_NAME}/${TASK_CONFIG}" \
    "task_config/${TASK_CONFIG}.yml" \
    "envs/${TASK_NAME}.py" \
    | remote "cd '${REMOTE_ROOT}' && tar -xzf -"

  remote "test -d '${REMOTE_ROOT}/data/${TASK_NAME}/${TASK_CONFIG}/data' && count=\$(find '${REMOTE_ROOT}/data/${TASK_NAME}/${TASK_CONFIG}/data' -maxdepth 1 -type f -name 'episode*.hdf5' | wc -l) && du -sh '${REMOTE_ROOT}/data/${TASK_NAME}/${TASK_CONFIG}' && test \"\$count\" -ge '${EXPERT_DATA_NUM}'"
  status "upload_done"
  log "Done"
}

main "$@"
