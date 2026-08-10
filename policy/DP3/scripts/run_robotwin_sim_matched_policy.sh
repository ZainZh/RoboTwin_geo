#!/usr/bin/env bash
set -euo pipefail

if (( $# != 5 )); then
    echo "Usage: $0 ROUTE GPU_ID TASK_NAME ZARR_PATH FEATURE_DIM" >&2
    exit 2
fi

route=$1
gpu_id=$2
task_name=$3
zarr_path=$4
feature_dim=$5

case "${route}" in
    vanilla|field|xyz|partprob|uniformprob|shuffledprob|utonia|utonia128|dinov2) ;;
    *)
        echo "Unsupported route: ${route}" >&2
        exit 2
        ;;
esac

if [[ "${route}" == "vanilla" && "${feature_dim}" != "0" ]]; then
    echo "Vanilla route requires FEATURE_DIM=0, got ${feature_dim}" >&2
    exit 2
fi
if [[ "${route}" != "vanilla" ]] && {
    [[ ! "${feature_dim}" =~ ^[0-9]+$ ]] || (( feature_dim < 3 ));
}; then
    echo "FEATURE_DIM must be an integer >= 3, got ${feature_dim}" >&2
    exit 2
fi

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
dp3_root=$(cd "${script_dir}/.." && pwd)
repo_root=$(cd "${dp3_root}/../.." && pwd)

if [[ "${zarr_path}" != /* ]]; then
    zarr_path="${repo_root}/${zarr_path}"
fi
if [[ ! -d "${zarr_path}" ]]; then
    echo "Zarr dataset does not exist: ${zarr_path}" >&2
    exit 3
fi

seed=${SEED:-20260805}
epochs=${EPOCHS:-300}
expert_data_num=${EXPERT_DATA_NUM:-50}
semantic_point_num=${SEMANTIC_POINT_NUM:-128}
batch_size=${BATCH_SIZE:-256}
dataset_seed=${DATASET_SEED:-424242}
val_ratio=${VAL_RATIO:-0.1}
encoder_output_dim=${ENCODER_OUTPUT_DIM:-128}
python_bin=${PYTHON_BIN:-python}
precision=${PRECISION:-bf16}
allow_tf32=${ALLOW_TF32:-true}
pointcloud_fusion_mode=${POINTCLOUD_FUSION_MODE:-concat}
semantic_gate_init=${SEMANTIC_GATE_INIT:-0.0}
semantic_gate_trainable=${SEMANTIC_GATE_TRAINABLE:-true}
semantic_attention_heads=${SEMANTIC_ATTENTION_HEADS:-4}
semantic_attention_dropout=${SEMANTIC_ATTENTION_DROPOUT:-0.0}
part_pool_temperature=${PART_POOL_TEMPERATURE:-1.0}
warmstart_safe_checkpoint=${WARMSTART_SAFE_CHECKPOINT:-}
warmstart_use_ema_weights=${WARMSTART_USE_EMA_WEIGHTS:-true}
freeze_vanilla_backbone=${FREEZE_VANILLA_BACKBONE:-false}
obs_key=${OBS_KEY:-semantic_point_cloud_A}
run_role=${RUN_ROLE:-local}
dry_run=${DRY_RUN:-0}

case "${precision}" in
    fp32|bf16) ;;
    *)
        echo "PRECISION must be fp32 or bf16, got ${precision}" >&2
        exit 2
        ;;
esac
if [[ "${allow_tf32}" != "true" && "${allow_tf32}" != "false" ]]; then
    echo "ALLOW_TF32 must be true or false, got ${allow_tf32}" >&2
    exit 2
fi
if [[ ! "${obs_key}" =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]]; then
    echo "OBS_KEY must be a valid Hydra mapping key, got ${obs_key}" >&2
    exit 2
fi

case "${run_role}" in
    local)
        train_workers=${TRAIN_WORKERS:-0}
        val_workers=${VAL_WORKERS:-0}
        ;;
    remote)
        train_workers=${TRAIN_WORKERS:-0}
        val_workers=${VAL_WORKERS:-0}
        ;;
    *)
        echo "RUN_ROLE must be local or remote, got ${run_role}" >&2
        exit 2
        ;;
esac

train_persistent_workers=${TRAIN_PERSISTENT_WORKERS:-false}
val_persistent_workers=${VAL_PERSISTENT_WORKERS:-false}
if (( train_workers > 0 )) && [[ -z "${TRAIN_PERSISTENT_WORKERS+x}" ]]; then
    train_persistent_workers=true
fi
if (( val_workers > 0 )) && [[ -z "${VAL_PERSISTENT_WORKERS+x}" ]]; then
    val_persistent_workers=true
fi
for persistent_value in "${train_persistent_workers}" "${val_persistent_workers}"; do
    if [[ "${persistent_value}" != "true" && "${persistent_value}" != "false" ]]; then
        echo "persistent worker flags must be true or false, got ${persistent_value}" >&2
        exit 2
    fi
done
if (( train_workers == 0 )); then train_persistent_workers=false; fi
if (( val_workers == 0 )); then val_persistent_workers=false; fi

output_root=${OUTPUT_ROOT:-${repo_root}/outputs/paper_revision/robotwin_sim_policy_v1}
setting_route="${route}"
if [[ "${pointcloud_fusion_mode}" != "concat" ]]; then
    setting_route="${route}-fusion-${pointcloud_fusion_mode}-ginit${semantic_gate_init}-gtrain${semantic_gate_trainable}-h${semantic_attention_heads}-t${part_pool_temperature}"
fi
if [[ -n "${warmstart_safe_checkpoint}" ]]; then
    setting_route="${setting_route}-warmema${warmstart_use_ema_weights}-freezebase${freeze_vanilla_backbone}"
fi
run_name="${task_name}_${setting_route}_joint14_e${epochs}_seed${seed}"
run_dir="${output_root}/${run_name}"
weights_path="${run_dir}/${run_name}.weights.pt"
checkpoint_task_name="${task_name}-paper-sim-${setting_route}-joint14-e${epochs}-${expert_data_num}"
mkdir -p "${run_dir}"

config_name=robot_dp3_semantic_pointwise_hybrid.yaml
if [[ "${route}" == "vanilla" ]]; then
    config_name=robot_dp3_objpc.yaml
fi
case "${pointcloud_fusion_mode}" in
    concat|residual_pointnet|state_cross_attention|part_soft_pool) ;;
    *)
        echo "Unsupported POINTCLOUD_FUSION_MODE: ${pointcloud_fusion_mode}" >&2
        exit 2
        ;;
esac
if [[ "${semantic_gate_trainable}" != "true" && "${semantic_gate_trainable}" != "false" ]]; then
    echo "SEMANTIC_GATE_TRAINABLE must be true or false, got ${semantic_gate_trainable}" >&2
    exit 2
fi
for warmstart_bool in "${warmstart_use_ema_weights}" "${freeze_vanilla_backbone}"; do
    if [[ "${warmstart_bool}" != "true" && "${warmstart_bool}" != "false" ]]; then
        echo "Warm-start boolean flags must be true or false, got ${warmstart_bool}" >&2
        exit 2
    fi
done
if [[ -n "${warmstart_safe_checkpoint}" && ! -s "${warmstart_safe_checkpoint}" ]]; then
    echo "WARMSTART_SAFE_CHECKPOINT does not exist or is empty: ${warmstart_safe_checkpoint}" >&2
    exit 3
fi
if [[ "${freeze_vanilla_backbone}" == "true" && -z "${warmstart_safe_checkpoint}" ]]; then
    echo "FREEZE_VANILLA_BACKBONE=true requires WARMSTART_SAFE_CHECKPOINT" >&2
    exit 2
fi
if [[ -n "${warmstart_safe_checkpoint}" && "${pointcloud_fusion_mode}" == "concat" ]]; then
    echo "Warm-start adapter training requires a non-concat fusion mode" >&2
    exit 2
fi
if [[ "${route}" == "vanilla" && "${pointcloud_fusion_mode}" != "concat" ]]; then
    echo "Strict Vanilla does not have a semantic branch; fusion mode must be concat" >&2
    exit 2
fi
if [[ "${pointcloud_fusion_mode}" == "part_soft_pool" ]] && (( feature_dim <= 3 )); then
    echo "part_soft_pool requires XYZ plus part-score channels" >&2
    exit 2
fi

if [[ -s "${weights_path}" ]]; then
    if "${python_bin}" "${script_dir}/safe_dp3_checkpoint.py" "${weights_path}" \
        --expected-epoch "${epochs}" \
        --expected-seed "${seed}" \
        --expected-task-name "${checkpoint_task_name}" \
        --expected-zarr-path "${zarr_path}" \
        --expected-precision "${precision}" \
        --expected-allow-tf32 "${allow_tf32}"; then
        echo "[robotwin-sim] validated completed weights: ${weights_path}"
        exit 0
    fi
    echo "[robotwin-sim] existing weights failed validation; training will replace them atomically: ${weights_path}" >&2
fi

command=(
    "${python_bin}" train_dp3.py
    "--config-name=${config_name}"
    "task_name=${task_name}"
    "hydra.run.dir=${run_dir}"
    training.debug=false
    "training.seed=${seed}"
    "training.precision=${precision}"
    "training.allow_tf32=${allow_tf32}"
    training.device=cuda:0
    training.use_ema=true
    training.resume=false
    "training.num_epochs=${epochs}"
    training.max_train_steps=null
    "training.checkpoint_every=${epochs}"
    training.val_every=25
    training.max_val_steps=null
    training.tqdm_interval_sec=10
    checkpoint.save_ckpt=false
    checkpoint.save_last_ckpt=false
    +checkpoint.save_weights_only=true
    "+checkpoint.weights_only_path=${weights_path}"
    "expert_data_num=${expert_data_num}"
    "setting=paper-sim-${setting_route}-joint14-e${epochs}"
    logging.mode=disabled
    "dataloader.batch_size=${batch_size}"
    "dataloader.num_workers=${train_workers}"
    dataloader.pin_memory=true
    "dataloader.persistent_workers=${train_persistent_workers}"
    "val_dataloader.batch_size=${batch_size}"
    "val_dataloader.num_workers=${val_workers}"
    val_dataloader.pin_memory=false
    "val_dataloader.persistent_workers=${val_persistent_workers}"
    "policy.encoder_output_dim=${encoder_output_dim}"
    "task.dataset.zarr_path=${zarr_path}"
    "task.dataset.seed=${dataset_seed}"
    "task.dataset.val_ratio=${val_ratio}"
)

if [[ "${pointcloud_fusion_mode}" != "concat" ]]; then
    command+=(
        "policy.pointcloud_fusion_mode=${pointcloud_fusion_mode}"
        "policy.semantic_gate_init=${semantic_gate_init}"
        "policy.semantic_gate_trainable=${semantic_gate_trainable}"
        "policy.semantic_attention_heads=${semantic_attention_heads}"
        "policy.semantic_attention_dropout=${semantic_attention_dropout}"
        "policy.part_pool_temperature=${part_pool_temperature}"
    )
fi

if [[ -n "${warmstart_safe_checkpoint}" ]]; then
    command+=(
        "training.warmstart_safe_checkpoint=${warmstart_safe_checkpoint}"
        "training.warmstart_use_ema_weights=${warmstart_use_ema_weights}"
        "training.freeze_vanilla_backbone=${freeze_vanilla_backbone}"
    )
fi

if [[ "${route}" != "vanilla" ]]; then
    command+=(
        "+task.dataset.extra_obs_keys=[${obs_key}]"
        "+task.shape_meta.obs.${obs_key}.shape=[${semantic_point_num},${feature_dim}]"
        "+task.shape_meta.obs.${obs_key}.type=point_cloud"
    )
fi

printf '[robotwin-sim] route=%s task=%s seed=%s epochs=%s role=%s precision=%s tf32=%s obs_key=%s weights=%s\n' \
    "${route}" "${task_name}" "${seed}" "${epochs}" "${run_role}" "${precision}" "${allow_tf32}" "${obs_key}" "${weights_path}"
if [[ "${dry_run}" == "1" ]]; then
    printf '%q ' env "CUDA_VISIBLE_DEVICES=${gpu_id}" "${command[@]}"
    printf '\n'
    exit 0
fi

cd "${dp3_root}/3D-Diffusion-Policy"
exec env CUDA_VISIBLE_DEVICES="${gpu_id}" "${command[@]}"
