#!/usr/bin/env bash
set -euo pipefail

REMOTE_HOST="${REMOTE_HOST:-root@112.69.3.12}"
REMOTE_PORT="${REMOTE_PORT:-43678}"
POLL_SECONDS="${POLL_SECONDS:-30}"
LOCAL_BASE="${LOCAL_BASE:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)/outputs/paper_revision/remote_rtx6000/formal}"
REMOTE_PREFIX="/workspace/RoboTwin_geo/outputs/paper_revision"
SEEDS=(20260805 20260806)
VARIANTS=(full_fixed no_ce no_supcon no_consistency)

mkdir -p "${LOCAL_BASE}"

remote_root() {
  local seed="$1"
  printf '%s/hammer_semantic_loss_ablation_remote_seed%s_e4000_20260806' "${REMOTE_PREFIX}" "${seed}"
}

sync_variant() {
  local seed="$1"
  local variant="$2"
  local run_name="hammer_${variant}_e4000_split42_seed${seed}"
  local remote_seed_root
  remote_seed_root="$(remote_root "${seed}")"
  local local_seed_root="${LOCAL_BASE}/seed${seed}"
  local synced_marker="${local_seed_root}/completed/${run_name}.synced"
  [[ -f "${synced_marker}" ]] && return 0


  mkdir -p "${local_seed_root}/completed" "${local_seed_root}/logs" "${local_seed_root}/manifests"
  rsync -a --checksum --mkpath -e "ssh -o BatchMode=yes -p ${REMOTE_PORT}" \
    "${REMOTE_HOST}:${remote_seed_root}/${run_name}/" \
    "${local_seed_root}/${run_name}/"
  rsync -a --checksum -e "ssh -o BatchMode=yes -p ${REMOTE_PORT}" \
    "${REMOTE_HOST}:${remote_seed_root}/logs/${run_name}.log" \
    "${local_seed_root}/logs/${run_name}.log"
  rsync -a --checksum -e "ssh -o BatchMode=yes -p ${REMOTE_PORT}" \
    "${REMOTE_HOST}:${remote_seed_root}/manifests/${run_name}.sha256" \
    "${local_seed_root}/manifests/${run_name}.sha256"
  rsync -a --checksum -e "ssh -o BatchMode=yes -p ${REMOTE_PORT}" \
    "${REMOTE_HOST}:${remote_seed_root}/completed/${run_name}.complete" \
    "${local_seed_root}/completed/${run_name}.complete"

  local differences
  differences="$(rsync -ani --checksum --delete -e "ssh -o BatchMode=yes -p ${REMOTE_PORT}" \
    "${REMOTE_HOST}:${remote_seed_root}/${run_name}/" \
    "${local_seed_root}/${run_name}/" | grep -E '^[<>ch.*]' || true)"
  if [[ -n "${differences}" ]]; then
    echo "checksum verification failed for ${run_name}: ${differences}" >&2
    return 1
  fi
  date -Is > "${synced_marker}"
  echo "[remote-sync] synced and checksum-verified: ${run_name}"
}

while true; do
  all_complete=1
  for seed in "${SEEDS[@]}"; do
    seed_root="$(remote_root "${seed}")"
    available="$(ssh -o BatchMode=yes -o ConnectTimeout=12 -p "${REMOTE_PORT}" "${REMOTE_HOST}" \
      "find '${seed_root}/completed' -maxdepth 1 -type f -name '*.complete' -printf '%f\\n' 2>/dev/null" \
      | grep -E '^hammer_.*\\.complete$' || true)"
    for variant in "${VARIANTS[@]}"; do
      remote_marker="hammer_${variant}_e4000_split42_seed${seed}.complete"
      if grep -Fqx "${remote_marker}" <<< "${available}"; then
        sync_variant "${seed}" "${variant}" || true
      fi
      marker="${LOCAL_BASE}/seed${seed}/completed/hammer_${variant}_e4000_split42_seed${seed}.synced"
      [[ -f "${marker}" ]] || all_complete=0
    done
  done
  if [[ "${all_complete}" == "1" ]]; then
    echo "[remote-sync] all remote variants are local and checksum-verified"
    exit 0
  fi
  sleep "${POLL_SECONDS}"
done
