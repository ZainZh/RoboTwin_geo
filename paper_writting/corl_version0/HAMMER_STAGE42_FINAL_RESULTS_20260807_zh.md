# Hammer Stage42 独立测试集最终结果

日期：2026-08-07

## 1. 结果资格与协议

- 训练实例划分：`split_seed=424242`，`val_ratio=0.1`，`test_ratio=0.2`。
- 每个 run 使用 33 个训练、5 个验证、10 个独立测试 Hammer 实例。
- 训练预算：1750 epochs；路径中的 `e4000` 仅是历史 artifact tag，最终 config、last 和 resume 均严格审计为 1750。
- 训练 seeds：20260805、20260806、20260807。
- 变体：Full、No-CE、No-SupCon、No-Consistency、CE-only。
- R1：独立 test，10 个 held-out instances，5 repeats，FP32。
- R2：固定 query，5 trials，support/reference normalization，共40个扰动条件，FP32。
- 最终门禁：15/15 training runs、30/30 evaluation results 验证通过。
- manifest：`matrix_complete=true`、`paper_claim_eligible=true`、`evaluation_status=held_out_test`。
- 统计单位是 training seed，而不是 point、trial 或 test object；表中误差均为三 training seeds 的 sample standard deviation。

## 2. R1 held-out instance 定量结果

| 变体 | mIoU (%) | Accuracy (%) | Handle IoU (%) | Head IoU (%) |
|---|---:|---:|---:|---:|
| Full | 75.82 ± 1.89 | 86.25 ± 1.22 | 75.87 ± 1.35 | 75.77 ± 2.47 |
| No-CE | 24.72 ± 0.96 | 49.44 ± 1.93 | 17.22 ± 29.83 | 32.22 ± 27.90 |
| No-SupCon | 71.04 ± 1.72 | 83.08 ± 1.15 | 71.55 ± 1.19 | 70.53 ± 2.34 |
| No-Consistency | 74.77 ± 1.31 | 85.56 ± 0.85 | 75.09 ± 0.96 | 74.44 ± 1.74 |
| CE-only | 77.43 ± 5.30 | 87.22 ± 3.36 | 77.65 ± 4.98 | 77.22 ± 5.63 |

### Paired seed 差值

下表为 `Full mIoU - 对照 mIoU`；正值代表 Full 更高。

| 对照 | seed05 | seed06 | seed07 | 平均差值（points） |
|---|---:|---:|---:|---:|
| No-CE | +49.51 | +50.05 | +53.74 | +51.10 |
| No-SupCon | +2.38 | +3.73 | +8.23 | +4.78 |
| No-Consistency | +1.90 | -0.56 | +1.83 | +1.06 |
| CE-only | +3.07 | -8.67 | +0.76 | -1.61 |

### R1结论

1. CE 是不可替代的核心监督。No-CE 三个 seed 均退化为单类别预测，mIoU 从75.82%降至24.72%。
2. 在 Full 组合中移除 SupCon，三个 seed 均下降，平均下降4.78 points，说明 SupCon 在与 Consistency 联合使用时有正作用。
3. 移除 Consistency 的干净 mIoU 仅平均下降1.06 points，并且一个 seed 反而上升；不能声称它稳定提高干净分割精度。
4. CE-only 的均值最高，但标准差5.30 points，是 Full 1.89 points 的约2.8倍；其提升主要由 seed06 驱动，不能据此声称 CE-only 显著优于 Full。
5. 论文应避免“所有辅助损失都提高mIoU”的表述，改为CE保证语义可识别，SupCon/Consistency主要约束表征稳定性与训练方差。

## 3. R2 表征稳定性与不完整点云鲁棒性

Full 的未扰动 reference mIoU 为 `76.74 ± 2.23%`；CE-only 为 `77.75%`。下表报告 support normalization 下跨三个 training seeds 的均值。

| 条件 | Full ΔmIoU | CE-only ΔmIoU | Full cosine | CE-only cosine | Full L2 | CE-only L2 |
|---|---:|---:|---:|---:|---:|---:|
| support 128 points | -13.22 | -17.23 | 0.976 | 0.671 | 0.155 | 0.657 |
| support 512 points | -4.08 | -2.19 | 0.987 | 0.847 | 0.101 | 0.398 |
| dropout 50% | +0.66 | +0.72 | 0.997 | 0.967 | 0.043 | 0.159 |
| dropout 75% | +0.13 | +0.13 | 0.995 | 0.930 | 0.063 | 0.248 |
| noise σ=0.005 | +0.08 | -0.11 | 0.998 | 0.968 | 0.040 | 0.167 |
| noise σ=0.01 | -0.64 | -1.85 | 0.998 | 0.958 | 0.046 | 0.190 |
| crop keep 50% | -25.64 | -26.26 | 0.954 | 0.602 | 0.208 | 0.680 |
| crop keep 25% | -45.74 | -46.68 | 0.903 | 0.277 | 0.349 | 1.019 |
| outlier replace 10% | -22.31 | -31.60 | 0.979 | 0.743 | 0.160 | 0.582 |
| outlier replace 20% | -19.96 | -25.40 | 0.982 | 0.796 | 0.148 | 0.519 |
| normal-view keep 50% | -13.55 | -18.62 | 0.984 | 0.757 | 0.129 | 0.538 |
| normal-view keep 25% | -20.05 | -27.56 | 0.972 | 0.581 | 0.176 | 0.751 |

### R2结论

1. Full 对随机 dropout 和小幅高斯噪声较稳定：mIoU变化大致在±0.7 points，cosine约0.995--0.998。
2. Full 对 support 点数减少有渐进退化：512点下降4.08 points，128点下降13.22 points。
3. 结构性视角缺失是主要弱点：crop keep25%下降45.74 points，normal-view keep25%下降20.05 points。
4. outlier contamination 也是显著弱点，10%/20%替换下降约22.31/19.96 points，且跨seed方差较大。
5. CE-only 虽然干净mIoU均值略高，但几乎所有强扰动下的feature cosine显著更低、L2 drift显著更高。例如crop keep25%时Full cosine=0.903、CE-only=0.277；Full L2=0.349、CE-only=1.019。
6. 因此辅助损失最有力、最诚实的定位是“稳定跨support和不完整观测的特征空间”，而不是提高干净分割均值。

## 4. 失败案例

- held-out instance `275ceccff41a4160b0537e1b875d670b` 在Full三个seed上的平均mIoU仅约19.15%，明显低于其余实例。
- 其三个seed分别约12.86%、16.32%、28.25%，说明这是稳定的instance-level failure，而不是单seed随机波动。
- 正文/附录应展示该实例与一个高性能实例，分析形状、部件比例或几何域偏移。

## 5. DINOv2 baseline当前边界

- visibility-aware DINOv2-S/14的投影、深度遮挡、mask和fusion链路已完成真实RGB-D smoke，并记录完整参数及逐视角visibility统计。
- 当前back相机外参失效：mask非空，但点云投影mask/depth overlap为0/5000；global为5000/5000。
- 当前只能诚实报告global-only工程baseline，不能称为有效multi-view DINO lifting。
- 正式多视角比较需要重新物理标定，或只在独立training calibration subset上估计一次固定SE(3)并冻结；禁止逐test frame/instance ICP。
- 仍需实现matched-query DINO zarr builder及相同policy train/eval launcher，才能进入策略成功率比较表。

## 6. 推荐论文表述

推荐核心陈述：

> Cross-entropy supervision is essential for semantic part identification. While the CE-only model attains comparable clean held-out mIoU, it exhibits substantially larger across-seed variance and severe feature drift under support resampling, viewpoint-induced incompleteness, and outlier contamination. The auxiliary contrastive and consistency objectives should therefore be interpreted as representation-stabilizing regularizers rather than guaranteed clean-accuracy improvements.

不应使用的表述：

- “SupCon和Consistency都显著提高clean mIoU。”
- “模型对不完整点云普遍鲁棒。”
- “当前DINO baseline是有效双视角融合。”
- 把trial、point或10个test objects当作独立training replicates计算显著性。

## 7. 结果文件

- 训练/评估manifest：`outputs/paper_revision/hammer_stage42_e1750_15run_manifest_v1/manifest.json`
- 最终summary：`outputs/paper_revision/hammer_stage42_independent_test_summary_e1750_v1/summary.json`
- R1 seed表：`outputs/paper_revision/hammer_stage42_independent_test_summary_e1750_v1/r1_seed_metrics.csv`
- R1 per-part：`outputs/paper_revision/hammer_stage42_independent_test_summary_e1750_v1/r1_per_part.csv`
- R1 confusion：`outputs/paper_revision/hammer_stage42_independent_test_summary_e1750_v1/r1_confusion.csv`
- R2 long table：`outputs/paper_revision/hammer_stage42_independent_test_summary_e1750_v1/r2_conditions_long.csv`
