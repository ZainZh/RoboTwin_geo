#!/usr/bin/env bash
set -euo pipefail

dataset="${1:-outputs/geometry_flow/remote4090/task_flow_remote50_ndf_ensemble_camera_conf.npz}"
output_root="${2:-outputs/geometry_flow/remote4090}"
python_bin="${ROBOTWIN_PYTHON:-/home/zheng/miniforge3/envs/RoboTwin/bin/python}"

for seed in 1 2; do
  for mode in normal zero shuffled; do
    output_dir="${output_root}/dense_single_ndf_policy_seed${seed}_${mode}"
    result="${output_dir}/fold0_interaction_functional_direct_flow_seed${seed}.json"
    if [[ -f "${result}" ]]; then
      echo "skip completed ${result}"
      continue
    fi
    mkdir -p "${output_dir}"
    echo "start seed=${seed} mode=${mode}"
    CUDA_VISIBLE_DEVICES=0 \
      OMP_NUM_THREADS=4 \
      MKL_NUM_THREADS=4 \
      PYTHONUNBUFFERED=1 \
      "${python_bin}" -m policy.GeometryFlow.train_interaction_flow_tokens \
      --dataset "${dataset}" \
      --output-dir "${output_dir}" \
      --conditions interaction_functional_direct_flow \
      --folds 0 \
      --seed "${seed}" \
      --epochs 200 \
      --patience 30 \
      --batch-size 32 \
      --learning-rate 0.0003 \
      --weight-decay 0.00001 \
      --feature-dim 128 \
      --layers 2 \
      --num-anchors 16 \
      --flow-steps 4 \
      --geometry-loss-weight 0.1 \
      --rigidity-loss-weight 0.02 \
      --rotation-loss-weight 0 \
      --flow-loss-mode unordered \
      --motion-loss-scale 10 \
      --flow-parameterization free \
      --target-horizon-mode terminal \
      --geometry-phase all \
      --active-arm-loss-weight 1 \
      --relation-loss-weight 1 \
      --rotation-action-loss-weight 1 \
      --action-frame world \
      --functional-frame-source dataset_relative \
      --target-frame-mode "${mode}" \
      --sample-mode all \
      --device cuda:0 \
      --num-workers 0 \
      --episode-balanced-sampling \
      > "${output_dir}.log" 2>&1
    echo "complete seed=${seed} mode=${mode}"
  done
done
