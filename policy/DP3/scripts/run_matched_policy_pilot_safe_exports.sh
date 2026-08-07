#!/usr/bin/env bash
set -euo pipefail

route=${1:-all}
gpu_id=${2:-0}
seed=${SEED:-20260805}
epochs=${EPOCHS:-300}

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
dp3_root=$(cd "${script_dir}/.." && pwd)
repo_root=$(cd "${dp3_root}/../.." && pwd)
output_root=${OUTPUT_ROOT:-${repo_root}/outputs/paper_revision/policy_pilot_safe}
mkdir -p "${output_root}"

run_route() {
    local route_name=$1
    local feature_dim=$2
    local zarr_path=$3
    local run_dir="${output_root}/beat_cube_${route_name}_e${epochs}_seed${seed}"
    local weights_path="${run_dir}/beat_cube_${route_name}_e${epochs}_seed${seed}.weights.pt"

    mkdir -p "${run_dir}"
    echo "[pilot-safe] route=${route_name} epochs=${epochs} weights=${weights_path}"
    cd "${dp3_root}/3D-Diffusion-Policy"
    CUDA_VISIBLE_DEVICES=${gpu_id} HYDRA_FULL_ERROR=1 PYTHONWARNINGS=ignore::UserWarning \
    python train_dp3.py --config-name=robot_dp3_semantic_pointwise_hybrid_eef_absolute6d.yaml \
        task_name=beat_cube \
        hydra.run.dir="${run_dir}" \
        training.debug=false \
        training.seed="${seed}" \
        training.device=cuda:0 \
        training.use_ema=true \
        training.resume=false \
        training.num_epochs="${epochs}" \
        training.max_train_steps=null \
        training.checkpoint_every="${epochs}" \
        training.val_every=25 \
        training.max_val_steps=1 \
        training.tqdm_interval_sec=10 \
        checkpoint.save_ckpt=false \
        checkpoint.save_weights_only=true \
        checkpoint.weights_only_path="${weights_path}" \
        expert_data_num=80 \
        setting="paper-revision-${route_name}-e${epochs}-safe" \
        dataloader.batch_size=256 \
        dataloader.num_workers=4 \
        dataloader.pin_memory=true \
        val_dataloader.batch_size=256 \
        val_dataloader.num_workers=2 \
        val_dataloader.pin_memory=false \
        policy.encoder_output_dim=128 \
        task.dataset.zarr_path="${zarr_path}" \
        +task.dataset.extra_obs_keys=[semantic_point_cloud_A] \
        +task.shape_meta.obs.semantic_point_cloud_A.shape="[128,${feature_dim}]" \
        +task.shape_meta.obs.semantic_point_cloud_A.type=point_cloud
}

field_zarr="${dp3_root}/data/beat_cube-demo_real_zed_sam2_objpc_global-80-objpc-semantic-pointwise-hybrid-semdebugref-eef-absolute6d-global.zarr"
xyz_zarr="${dp3_root}/data/beat_cube-demo_real_zed_sam2_objpc_global-80-objpc-semantic-pointwise-hybrid-ablation-xyz-eef-absolute6d-global.zarr"
partprob_zarr="${dp3_root}/data/beat_cube-demo_real_zed_sam2_objpc_global-80-objpc-semantic-pointwise-hybrid-ablation-partprob-eef-absolute6d-global.zarr"

case "${route}" in
    field) run_route field 131 "${field_zarr}" ;;
    xyz) run_route xyz 3 "${xyz_zarr}" ;;
    partprob) run_route partprob 5 "${partprob_zarr}" ;;
    all)
        run_route field 131 "${field_zarr}"
        run_route xyz 3 "${xyz_zarr}"
        run_route partprob 5 "${partprob_zarr}"
        ;;
    *)
        echo "Usage: $0 [all|field|xyz|partprob] [gpu_id]" >&2
        exit 2
        ;;
esac
