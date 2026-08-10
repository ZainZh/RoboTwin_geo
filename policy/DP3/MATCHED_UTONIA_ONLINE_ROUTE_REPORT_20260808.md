# Matched Utonia 在线闭环路由实现报告

日期：2026-08-08

## 目标与契约

正式 matched Utonia 策略训练数据使用 semantic field 路线确定的 128 个 query XYZ，并在同帧原始 object point cloud support 上查询 576D Utonia 特征。在线闭环必须复现这个顺序，输出：

- observation key：`utonia_point_cloud_A`
- shape：`[128, 579]`
- channels：query world XYZ 3D + Utonia feature 576D
- support：当前观测的原始 `object_pointcloud["{A}"]`
- query：semantic query 路径生成的同一批 XYZ，不允许 legacy Utonia 再次采样

## 实现

`policy/DP3/deploy_policy.py` 新增显式
`semantic_policy_output_mode=matched_utonia`：

1. 保持 semantic model 加载与 query 生成路径；
2. 强制 semantic helper 使用 `output_mode=xyz`；
3. 将该 XYZ 原样传给
   `matched_utonia_feature_utils.compute_utonia_features_at_queries`；
4. 使用同帧原始 object cloud 作为 support；
5. 写入 `utonia_point_cloud_A`，不写
   `semantic_point_cloud_A`；
6. 通过现有 Utonia loader 加载 checkpoint，默认
   `utonia_checkpoint=auto`；
7. 在构造 DP3 前生成 `[128,579]` shape_meta，并继续执行现有
   safe checkpoint 完整字典严格相等校验。

matched 模式只选择同时具有 semantic checkpoint 且被
`utonia_feature_placeholders` 选中的对象。它不能与 config 名触发的 legacy
`utonia_pointwise` 同时启用，避免 legacy 分支覆盖 matched 输出。

默认 Utonia 输入设置与正式离线 builder 一致：

- `matched_utonia_color_mode=debug_placeholder`
- `matched_utonia_normal_mode=fallback`

## 正式评估参数

Beat Block Hammer 与 Hanging Mug 均应继续使用 semantic config，并至少传入：

```text
--config_name robot_dp3_semantic_pointwise_hybrid
--semantic_policy_output_mode matched_utonia
--semantic_ckpt_A <对应 Hammer 或 Mug semantic checkpoint>
--semantic_ckpt_B none
--semantic_point_num 128
--semantic_input_color_mode debug_placeholder
--semantic_forward_mode reference
--utonia_checkpoint auto
--utonia_feature_placeholders {A}
--utonia_device cuda:0
```

safe policy checkpoint 本身必须声明
`utonia_point_cloud_A: {shape: [128,579], type: point_cloud}`。任何 observation
key、点数、通道或其他 shape_meta 差异都会在 DP3 构造前失败，不会静默降级。

## CPU 验证

在 `CUDA_VISIBLE_DEVICES=''` 下运行：

```bash
cd policy/DP3/scripts
PYTHONPATH=/home/zheng/github/Utonia \
  /home/zheng/miniforge3/envs/RoboTwin/bin/python -m unittest \
  test_semantic_policy_output_modes.py \
  test_dp3_safe_deploy_path.py \
  test_matched_utonia_feature_utils.py \
  test_semantic_pointwise_hybrid.py \
  test_utonia_pointwise_hybrid.py \
  test_actorseg_pointwise_hybrid.py
```

结果：29/29 通过。覆盖：

- matched 输出 key 与 `[Q,579]` shape；
- semantic query 逐值不变地传入 Utonia helper；
- 原始 object point cloud 作为 support；
- semantic 与 Utonia checkpoint/artifact 同时加载；
- safe checkpoint 完整 shape_meta 严格通过；
- embedding、xyz、part probability、uniform、shuffled 旧模式；
- legacy Utonia pointwise/hybrid 与 actor-seg context 路由。

本次没有启动 GPU、仿真或真实机器人。下一步是在正式 safe checkpoint 上先跑
1–2 个 fixed-seed simulator smoke，再启动 Beat/Hanging 完整闭环评估。
