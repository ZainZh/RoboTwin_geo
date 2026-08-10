# Matched Policy Pilot：固定划分 Raw/EMA 结果

> **已被三 seed 结果取代：** 请以 `MATCHED_POLICY_3SEED_RESULTS_20260806_zh.md` 为准；本文件只保留单 seed 诊断的历史记录。

日期：2026-08-06
状态：单任务、单训练 seed、e300 早期诊断；不是论文闭环主表。

## 1. 为什么做这个实验

原稿没有回答一个最关键的公平性问题：策略提升究竟来自 semantic field，还是来自额外对象点分支、更多参数或对象坐标本身。本实验固定 DP3 主干、action route、query、训练预算和数据划分，只替换对象分支每个 query 点携带的特征：

1. XYZ only；
2. XYZ + part probability；
3. XYZ + 128D semantic field embedding。

formal matched Utonia 尚未纳入，因为 Beat 原始 `{A}` object support 数据目录缺失；现有 context-nearest Utonia zarr 只用于工程 smoke，manifest 明确禁止作为论文对照。

## 2. 严格匹配协议

三条路线都来自同一份 Beat Cube 实机 ZED 数据：80 episodes、2774 frames、20维 EEF absolute-6D action。

- XYZ 与 part-prob 从 field zarr 派生；
- action、state、scene point cloud、episode boundaries 逐元素不变；
- semantic query xyz 完全一致；
- policy seed：20260805；
- 训练：300 epochs，每 epoch 11 updates，共3300 updates；
- batch size：256；
- 同一 DP3 主干、同一128维 point encoder output；
- validation：dataset seed 0、val ratio 0.02；episode 50、67，共73个 sequence samples；
- 离线评测：20个固定 diffusion-noise seeds；raw 与 EMA 使用成对噪声；
- 指标：held-out diffusion imitation loss，越低越好。

旧 workspace `.ckpt` 将 OmegaConf、dill、optimizer 和权重混合在同一个 pickle 容器中，不能被 `weights_only=True` 安全读取。本轮按完全相同配置重跑；所有可见 validation 点与原 pilot 逐位一致，同时输出仅含 raw/EMA state_dict 和 primitive metadata 的安全 companion。三份 companion 均已由独立进程用 `weights_only=True` 回读。

## 3. 结果

| 表征 | 输入形状 | 参数量 | Raw mean ± SEM | EMA mean ± SEM | EMA - Raw |
|---|---:|---:|---:|---:|---:|
| semantic field embedding | `[128,131]` | 273.7122M | 0.00977856 ± 0.00000278 | 0.00977729 ± 0.00000283 | -0.00000128 |
| XYZ only | `[128,3]` | 273.7040M | 0.01066560 ± 0.00000936 | 0.01067183 ± 0.00000971 | +0.00000623 |
| XYZ + part probability | `[128,5]` | 273.7041M | **0.00954789 ± 0.00000405** | **0.00955390 ± 0.00000413** | +0.00000600 |

Raw model 的 paired 对比：

- field 比 XYZ loss 低 8.32%；`field - XYZ = -0.00088704`，noise-seed 95% CI `[-0.00090694, -0.00086714]`，20/20 个噪声 seed 更低；
- part-prob 比 field 低 2.36%；`field - part-prob = +0.00023067`，95% CI `[+0.00021921, +0.00024213]`，20/20 个 seed 中 part-prob 更低；
- part-prob 比 XYZ 低 10.48%；`part-prob - XYZ = -0.00111771`，95% CI `[-0.00113824, -0.00109718]`。

EMA 排序相同。上述置信区间只描述固定模型、固定73个validation samples下的 diffusion noise 波动，不能替代跨训练 seed 或跨 episode 的统计不确定性。

## 4. 论文含义

当前可以支持：

- 在完全 matched 的早期训练诊断中，field embedding 相对 capacity-matched XYZ 更容易拟合 held-out action distribution；
- 对象分支的 part structure 确实有用，收益不是单纯来自对象坐标或约8k参数差异；
- 原训练日志实际验证 raw model；e300 时 EMA 与 raw 很接近，而且 EMA 并非对所有路线更好。

当前最重要的负结果是：part probability 比完整128D field embedding 更低。因此现有证据不支持“高维连续 embedding 比显式 part categories 对策略更有价值”。更可能成立的表述是：稳定的 part-aware query interface 优于纯对象坐标。连续 field 的额外价值仍需用多 seed、闭环成功率、遮挡和跨实例任务证明。

当前不能支持：

- 不能把 imitation loss 写成 task success rate 或物理轨迹误差；
- 不能从1个训练 seed、2个validation episodes 推断跨实例泛化；
- 不能把20个 diffusion-noise repeats 当成20次独立训练或20个任务 trial；
- 不能把 context-nearest Utonia smoke 放入论文表格；
- 不能仅凭本实验决定实机部署最终选择 field 或 part-prob。

## 5. 下一轮实验决策

1. 先补 field / part-prob / XYZ 至少3个 policy seeds；如果稳定得到 `part-prob ≤ field < XYZ`，论文贡献应收窄为“稳定的 part-aware query interface”。
2. 恢复 Beat 原始 `{A}` object support 后立即生成 formal matched Utonia；缺少这一行仍无法完整回应 reviewer 的公平 baseline 质疑。
3. 用已经补齐的实机 `semantic_policy_output_mode` 对三路线做相同初始条件闭环 trial，保存逐 trial 成败、初始状态和失败阶段。
4. 增加 semantic label corruption / probability shuffle。如果 part-prob 的优势来自真正的 part structure，破坏标签后应显著退化。
5. 正式论文结果至少使用3个训练 seeds；闭环 trial 用 paired episode initializations，报告 Wilson CI 与 paired difference。

## 6. 产物

实现：

- `policy/DP3/scripts/safe_dp3_checkpoint.py`
- `policy/DP3/scripts/evaluate_dp3_checkpoint_offline.py`
- `policy/DP3/scripts/run_matched_policy_pilot_safe_exports.sh`
- `policy/DP3/scripts/test_safe_dp3_checkpoint.py`
- `policy/DP3/scripts/test_evaluate_dp3_checkpoint_offline.py`

训练与评测：

- `outputs/paper_revision/policy_pilot_safe/`
- `outputs/paper_revision/policy_offline_eval/beat_cube_field_e300_seed20260805.json`
- `outputs/paper_revision/policy_offline_eval/beat_cube_xyz_e300_seed20260805.json`
- `outputs/paper_revision/policy_offline_eval/beat_cube_partprob_e300_seed20260805.json`

每个评测 JSON 都包含完整20次 loss、raw/EMA paired delta、validation episode IDs、checkpoint SHA256、运行环境、耗时和峰值显存。
