#!/usr/bin/env bash
set -euo pipefail

remote_host=${REMOTE_HOST:-root@112.69.3.12}
remote_port=${REMOTE_PORT:-43678}
remote_repo=${REMOTE_REPO:-/workspace/RoboTwin_geo_sim_v1}
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd "${script_dir}/../../.." && pwd)
data_stem=hanging_mug-demo_clean_3d_object_pc-50-objpc-semantic-pointwise-hybrid-sim-paper-v1
local_base="${repo_root}/policy/DP3/data/${data_stem}"
remote_data_dir="${remote_repo}/policy/DP3/data"
local_marker="${repo_root}/outputs/paper_revision/robotwin_sim_policy_hanging_mug_v1/matched_data.complete"
remote_marker="${remote_repo}/outputs/paper_revision/robotwin_sim_policy_hanging_mug_e300_bf16_v1/hanging_data.remote-ready"
ssh_cmd="ssh -p ${remote_port} -o BatchMode=yes -o ServerAliveInterval=30 -o ServerAliveCountMax=4"

until [[ -s "${local_marker}" ]]; do
    printf '[hanging-data-sync] waiting marker=%s\n' "${local_marker}"
    sleep 60
done

${ssh_cmd} "${remote_host}" "mkdir -p '${remote_data_dir}' '$(dirname "${remote_marker}")'"
for suffix in '' -xyz -partprob -uniformprob -shuffledprob -utonia; do
    source="${local_base}${suffix}.zarr"
    [[ -d "${source}" ]] || { echo "missing source zarr: ${source}" >&2; exit 3; }
    rsync -a --partial --partial-dir=.rsync-partial --info=progress2 --human-readable \
        -e "${ssh_cmd}" "${source}" "${remote_host}:${remote_data_dir}/"
done
rsync -a --partial --info=progress2 --human-readable -e "${ssh_cmd}" \
    "${local_base}_meta.json" "${remote_host}:${remote_data_dir}/"

remote_base="${remote_data_dir}/${data_stem}"
${ssh_cmd} "${remote_host}" \
    "cd '${remote_repo}' && /workspace/venvs/geo-utonia/bin/python policy/DP3/scripts/validate_semantic_policy_routes.py --field-zarr '${remote_base}.zarr' --xyz-zarr '${remote_base}-xyz.zarr' --partprob-zarr '${remote_base}-partprob.zarr' --uniform-zarr '${remote_base}-uniformprob.zarr' --shuffled-zarr '${remote_base}-shuffledprob.zarr' --semantic-keys semantic_point_cloud_A --expected-query-points 128 --expected-field-feature-dim 128 --output-json outputs/paper_revision/robotwin_sim_policy_hanging_mug_e300_bf16_v1/hanging_route_validation.json"
${ssh_cmd} "${remote_host}" \
    "cd '${remote_repo}' && /workspace/venvs/geo-utonia/bin/python -c \"import numpy as np,zarr; f=zarr.open('${remote_base}.zarr','r'); u=zarr.open('${remote_base}-utonia.zarr','r'); assert f['data/action'].shape==u['data/action'].shape==(16942,14); assert u['data/utonia_point_cloud_A'].shape==(16942,128,579); assert np.array_equal(f['meta/episode_ends'][:],u['meta/episode_ends'][:]); assert np.array_equal(f['data/action'][:],u['data/action'][:]); assert np.array_equal(f['data/state'][:],u['data/state'][:]); assert np.array_equal(f['data/point_cloud'][:],u['data/point_cloud'][:]); assert np.array_equal(f['data/semantic_point_cloud_A'][:,:,:3],u['data/utonia_point_cloud_A'][:,:,:3]); assert np.isfinite(u['data/utonia_point_cloud_A'][:]).all()\""
${ssh_cmd} "${remote_host}" \
    "printf 'frames=16942 routes=6 validated_at_utc=%s\\n' \"\$(date -u +%Y-%m-%dT%H:%M:%SZ)\" > '${remote_marker}'"
printf '[hanging-data-sync] remote data validated marker=%s\n' "${remote_marker}"
