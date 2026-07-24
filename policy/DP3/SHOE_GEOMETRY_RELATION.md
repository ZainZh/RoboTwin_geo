# 鞋—斜坡观测几何关系路线

这条路线的运行时数据流是：

```text
object_pointcloud/{A},{B}
  -> 冻结的 shoe NDF（256×3 等变特征）
  -> 向量模长 + A/B 旋转不变尺度/径向统计
  -> 小型 MLP
  -> 目标关系 T_A_from_B
  -> simulator 当前 A/B pose（只在此处使用）
  -> 11 维 SE(3) correction token
  -> DP3
```

估计器接口不接收 `shoe_id`、资产路径、functional matrix 或 simulator
pose。`shoe_id` 仅可用于训练/验证划分以及 benchmark 报告。PCA 不在这条
主路线中。

旧数据若没有 `task_state/goal_T_A_from_B_oracle` 会直接报错；只有显式传入
`--allow_legacy_asset_supervision` 才允许用资产 ID 重建离线标签，该兼容模式不应
用于主实验。

## 1. 训练几何关系头

在 `policy/DP3` 下运行：

```bash
bash train_shoe_geometry_relation.sh \
  place_shoe_rotating_block \
  demo_clean_3d_object_pc_se3_relation \
  50 \
  /path/to/shoe.pth \
  7 \
  ../../outputs/shoe_geometry_relation \
  8,9
```

最后一个参数表示把 shoe 8、9 留作几何验证，只用于划分，不进入模型。
输出包括：

- `ndf_goal_regressor.pt`
- `ndf_goal_regressor.json`（后续命令使用这个 spec）

训练报告同时给出 `constant_goal_validation`。这是一个完全不看点云、只
输出训练集平均目标的必要消融；判断 NDF 是否提供额外几何信息时，应将
它与 `validation` 的毫米/角度误差比较，而不是只看宽松的 success rate。

## 2. 纯几何 benchmark

```bash
bash benchmark_shoe_geometry_relation.sh \
  place_shoe_rotating_block \
  demo_clean_3d_object_pc_se3_relation \
  50 \
  ../../outputs/shoe_geometry_relation/ndf_goal_regressor.json \
  7 \
  ../../outputs/shoe_geometry_relation/benchmark_shoe89.json \
  8,9
```

先检查 held-out shoe 的平移误差、旋转误差和 180° flip。这个 benchmark
调用估计器时只传 A/B 点云；oracle 和 ID 在推理完成后才用于计分。

## 3. 生成 observation-derived token 并训练 DP3

```bash
bash train_shoe_se3_placement_comparison.sh \
  place_shoe_rotating_block \
  demo_clean_3d_object_pc_se3_relation \
  50 0 7 \
  ndf_observation_goal \
  ../../outputs/shoe_geometry_relation/ndf_goal_regressor.json
```

新 route 的第七个参数是 estimator spec；旧的
`ndf_no_direction`/`ndf_direction` route 在同一位置仍传 goal table，
因此原四组实验不受影响。

## 4. 评估新 DP3

```bash
bash eval_shoe_se3_placement_comparison.sh \
  place_shoe_rotating_block \
  demo_clean_3d_object_pc_se3_relation \
  demo_clean_3d_object_pc_se3_relation \
  50 0 7 \
  ndf_observation_goal \
  ../../outputs/shoe_geometry_relation/ndf_goal_regressor.json
```

这一步仍保留 simulator 当前 object pose；移除当前 pose 是后续独立阶段，
不应和本次“去除目标查询中的 shoe_id”混在同一个实验里。
