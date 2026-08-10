#!/usr/bin/env bash
set -euo pipefail

remote_host=${REMOTE_HOST:-root@112.69.3.12}
remote_port=${REMOTE_PORT:-43678}
remote_root=${REMOTE_ROOT:-/workspace/RoboTwin_geo_sim_v1/outputs/paper_revision/robotwin_sim_policy_beat_hammer_e300_bf16_v1/}
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd "${script_dir}/../../.." && pwd)
local_root=${LOCAL_ROOT:-${repo_root}/outputs/paper_revision/robotwin_sim_policy_beat_hammer_e300_bf16_v1/}
interval=${SYNC_INTERVAL_SECONDS:-120}

if [[ ! "${interval}" =~ ^[0-9]+$ ]] || (( interval < 30 )); then
    echo "SYNC_INTERVAL_SECONDS must be an integer >= 30, got ${interval}" >&2
    exit 2
fi

mkdir -p "${local_root}"
while true; do
    printf '[remote-sync] started_at_utc=%s source=%s:%s destination=%s\n' \
        "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${remote_host}" "${remote_root}" "${local_root}"
    if rsync -a --partial --partial-dir=.rsync-partial \
        --info=progress2 --human-readable \
        --exclude='.*.tmp' \
        -e "ssh -p ${remote_port} -o BatchMode=yes -o ServerAliveInterval=30 -o ServerAliveCountMax=4" \
        "${remote_host}:${remote_root}" "${local_root}"; then
        printf '[remote-sync] completed_at_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    else
        status=$?
        printf '[remote-sync] failed_at_utc=%s status=%s; retrying after interval\n' \
            "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${status}" >&2
    fi
    sleep "${interval}"
done
