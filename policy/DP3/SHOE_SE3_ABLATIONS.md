# Shoe SE(3) token 消融

以下三个实验回答两个不同问题：

- 公平的 `constant_goal` 重训：不看点云的平均目标是否也能让 DP3 达到 90%。
- 已训练 90% 模型的 `zero` 推理：该 DP3 是否真的依赖关系 token。
- 已训练 90% 模型的 `constant_goal` 推理：该 DP3 是否需要实例相关目标。

所有命令在 `policy/DP3` 下运行。使用与主实验相同的 training seed、GPU、
checkpoint 和 evaluation seed。

## 1. 公平的 constant-goal 重训

这条命令先从 50 个训练 episode 的 oracle `T_A_from_B` 求 SE(3) 平均值，
写出 `constant_goal.json`，再生成独立 zarr 并训练独立 DP3：

```bash
bash train_shoe_constant_goal_ablation.sh \
  place_shoe_rotating_block \
  demo_clean_3d_object_pc_se3_relation \
  50 0 7 \
  ../../outputs/shoe_geometry_relation
```

constant estimator 不读取点云、`shoe_id`、NDF checkpoint 或资产路径。

训练完成后评估：

```bash
bash eval_shoe_se3_placement_comparison.sh \
  place_shoe_rotating_block \
  demo_clean_3d_object_pc_se3_relation \
  demo_clean_3d_object_pc_se3_relation \
  50 0 7 \
  constant_goal \
  ../../outputs/shoe_geometry_relation/constant_goal.json
```

## 2. 已有 90% 模型：zero-token 推理

不重新训练，把已有 `ndf_observation_goal` DP3 的 11 维 token 全部置零：

```bash
bash eval_shoe_se3_token_ablation.sh \
  place_shoe_rotating_block \
  demo_clean_3d_object_pc_se3_relation \
  demo_clean_3d_object_pc_se3_relation \
  50 0 7 \
  ../../outputs/shoe_geometry_relation/ndf_goal_regressor.json \
  zero
```

`zero` 分支不会加载或执行 NDF/geometry estimator。

## 3. 已有 90% 模型：constant-token 推理

不重新训练，把 NDF 预测目标替换为训练集平均目标：

```bash
bash eval_shoe_se3_token_ablation.sh \
  place_shoe_rotating_block \
  demo_clean_3d_object_pc_se3_relation \
  demo_clean_3d_object_pc_se3_relation \
  50 0 7 \
  ../../outputs/shoe_geometry_relation/ndf_goal_regressor.json \
  constant_goal \
  ../../outputs/shoe_geometry_relation/constant_goal.json
```

## 解释

- `zero` 大幅下降：新 DP3 确实使用了关系 token。
- `zero` 仍接近 90%：DP3 可能忽略 token，提升可能来自训练随机性。
- 公平重训的 `constant_goal` 接近 90%：当前任务主要验证显式 correction，
  不能证明 NDF 实例几何贡献。
- 公平重训的 `constant_goal` 明显低于 90%：点云派生目标提供了额外信息。
- 现有模型的 constant-token 低于 90%：策略对实例相关目标敏感。
