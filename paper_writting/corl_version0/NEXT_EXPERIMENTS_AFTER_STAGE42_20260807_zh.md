# Stage42 之后的实验优先级

日期：2026-08-07

## 已冻结结果

- Stage42 的 15 个训练 run、30 个 held-out R1/R2 评测、汇总表和最终结果文档已冻结；309 个文件、601,457,472 bytes 均已逐文件 SHA-256 验证。
- 策略层 field / XYZ / part-prob 的 3 seed、e3000 full-budget 模型与离线评测也已冻结；66 个文件、19,711,925,573 bytes 均已验证。
- e3000 下 raw imitation loss（越低越好）为：part-prob 0.009016 +/- 0.000194、XYZ 0.009914 +/- 0.000495、field 0.010136 +/- 0.001990。part-prob 在 3/3 seeds 优于 XYZ；128D field 未显示稳定优势。

因此，论文应把主张收窄为：3D semantic field 产生稳定的 part-aware policy interface；part probability 是当前最有证据的策略输入，128D embedding 是消融，不应再声称它默认最优。

## E1：part semantics 破坏对照（立即自动运行）

目的：验证 part-prob 的收益来自正确的逐点语义，而非多两个数值通道。

在现有 matched zarr 上构建并训练两条对照：

1. uniform-prob：每点固定 [0.5, 0.5]，容量不变、语义信息移除；
2. within-frame shuffled-prob：每帧打乱 query 点的概率，保留统计分布与尺度，但破坏点-语义对应。

二者与现有 part-prob 严格共享 query xyz、state、action、episode split、DP3 容量、e3000 预算和 3 个 training seeds。先比固定 split 离线 loss；若语义版本稳定胜出，再一起进入闭环。双机并行预计 6--10 小时。

## E2：同初始条件闭环成功率（论文最高优先级）

这是把表征稳定性连接到控制收益的核心证据。使用已经完成的 e3000 三 seed 模型，至少比较：

1. capacity-matched XYZ；
2. XYZ + part probability；
3. XYZ + 128D field embedding。

每条路线共享冻结初始状态、成功判据和最大时长；逐 trial 保存模型 seed、初始状态、成功/失败、完成时间和失败阶段。先做 10 个相同初始条件 pilot 验证部署，再扩展到预先固定的正式 trial 数。离线 imitation loss 不能替代闭环成功率。

## E3：formal matched Utonia（审稿人关键公平性问题）

当前因 Beat 原始 5000-point object support 缺失而阻塞；现有 zarr 的 128 query 点无法逆推 raw support。恢复原始数据后，Utonia 必须和 ours 统一 query indices/count、preprocessing、EEF20/global action、DP3 容量、3 seeds，并纳入 E2 的同一闭环协议。

## E4：visibility-aware DINOv2（审稿人关键强 baseline）

投影、深度遮挡和融合已经可运行，但 back 相机外参失效；当前只能如实称为 global-only，不能当有效多视角基线。正式比较需要重新标定 back 相机（或只在独立 calibration subset 拟合一次固定 SE(3) 并冻结）、生成 matched-query DINO zarr，再按 E2 协议训练和评估。禁止逐 test frame ICP。

## E5：解释 sim-real gap 的闭环感知退化

在同一 policy 上施加与 R2 对齐的 dropout、view/crop 缺失和 outlier contamination；在仿真和实物统一记录成功率和失败阶段。该实验用于验证 R2 的结论：随机 dropout/小噪声影响小，结构性缺失与 outlier 是主要弱点。

## E6：第二任务与跨实例闭环

Beat Cube 外至少补一个数据链路完整的代表任务，优先 Stirring。重复 XYZ / part-prob / field 的 matched 比较；若无法完成，论文主张必须收窄为单任务。

## 执行顺序

- 立即用 GPU 跑 E1，不再重复 Stage42 或现有 e3000 矩阵。
- 并行生成 R1/R2 图表和 held-out 失败案例可视化，直接服务改稿。
- 准备 E2 的 trial manifest 和部署核验，等待机器人实机采集。
- E3 需恢复 Beat raw support；E4 需新 back 相机标定或独立 calibration subset。

不做：追加 Stage42 epoch、重复 e3000、把 noise repeats 当独立样本、把 global-only DINO 写成 multi-view、从 128 query 点伪造 raw support，或依靠挑 seed/删失败案例改善数字。
