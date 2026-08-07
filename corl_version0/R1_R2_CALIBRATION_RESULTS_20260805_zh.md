# R1 / R2 表征诊断与 Hammer 校准实验记录

日期：2026-08-05
状态：旧 validation split 上的诊断实验，尚不是论文可直接使用的 independent-test 结果

## 1. 结论摘要

1. 单张 RTX 4090 足够完成当前表征诊断和校准。三类别完整 R2、Hammer 30-epoch calibration 及复评均未接近显存上限。
2. 旧 checkpoint 的 clean 表征并不差：Hammer / Mug / Spoon 的 validation mIoU 分别为 0.8580 / 0.9636 / 0.9641。
3. Mug 的 Closure 在旧 validation split 中没有任何 GT 点，所以 0.9636 实际只平均 Container 和 Handle，不能写成完整三部件 mIoU。
4. R2 说明 Gaussian noise 不是当前主要问题；真正薄弱项是低覆盖的稀疏 support 和单侧 partial view。
5. 30-epoch Hammer fixed-query calibration 保持并小幅提高 clean R1 mIoU（0.8580 → 0.8644），同时显著改善 128–1024 点稀疏 support 以及中等程度 partial view。
6. 极端 25% crop 在当前 support-normalization 下仍没有可靠改善，说明单靠 loss/augmentation 不够，归一化和 support-dependent tri-plane bounds 仍是结构性瓶颈。
7. 所有这些结果仍来自旧的 checkpoint-selected validation split。正式投稿前必须重新建立 train/val/test manifest、检查每个 split 的 part coverage，并在 independent test 上复现。

## 2. R1 的 mIoU 是什么

对每个部件类别 \(c\)，先计算

\[
\mathrm{IoU}_c =
\frac{\mathrm{TP}_c}
{\mathrm{TP}_c+\mathrm{FP}_c+\mathrm{FN}_c}.
\]

然后对当前 split 中有有效 GT / union 的类别取平均：

\[
\mathrm{mIoU} = \frac{1}{C}\sum_{c=1}^{C}\mathrm{IoU}_c.
\]

它比 overall point accuracy 更适合部件表征，因为 accuracy 很容易被大部件占比主导。mIoU 会分别惩罚每个部件的漏检和误检。

需要特别区分：

- R2 中的 128 / 256 / 512 / 1024 / 5000 是 semantic field 的 conditioning support 点数。
- DP3 策略输入中的 128 个 semantic query 点是另一个超参数。
- 两者不能在论文中混写。

## 3. 实验协议

### 3.1 R1：clean part quality

- split：checkpoint 中原有的 validation split；
- checkpoint：Hammer / Mug / Spoon 现有 best checkpoint；
- 每个对象独立重采样 5 次；
- support/query：5000 / 2048；
- 精度：FP32；
- 指标：accuracy、mIoU、per-part IoU、precision、recall、confusion matrix；
- 所有结果保存逐对象、逐 repeat、逐 part CSV 和可重算 JSON。

### 3.2 R2：fixed-query support perturbation

对每个对象只采样一次带 GT 的 raw query 坐标。所有条件保持这些 query 和标签不变，只改变 support：

- 独立重采样：128、256、512、1024、5000 点；
- 从同一 5000 点 reference 随机 dropout：25%、50%、75%；
- Gaussian noise：物体半径的 0.25%、0.5%、1%；
- 单侧 crop：保留 75%、50%、25%；
- 每个对象、每个条件运行 5 个随机 trial；
- 同时评估 support-normalization 和 reference-normalization。

两个归一化模式的含义：

- support：按当前 perturbed support 重新计算中心、尺度和 z-shift，等价于当前真实推理路径；
- reference：输入坐标使用固定 reference normalization，用来隔离归一化漂移；
- 即使使用 reference，当前 tri-plane projector 的 bounds 仍由 perturbed support 的 AABB 决定，所以它不是完全固定边界的实验。

## 4. R1 clean 结果

| 类别 / 模型 | val 对象数 | Point accuracy | mIoU | Per-part IoU |
|---|---:|---:|---:|---|
| Hammer 原 checkpoint | 5 | 0.9237 | 0.8580 | Handle 0.8539；Head 0.8622 |
| Hammer calibrated | 5 | 0.9274 | 0.8644 | Handle 0.8599；Head 0.8690 |
| Mug 原 checkpoint | 16 | 0.9823 | 0.9636* | Container 0.9713；Handle 0.9559；Closure N/A |
| Spoon 原 checkpoint | 5 | 0.9817 | 0.9641 | Handle 0.9633；Head 0.9650 |

\* Mug 的 Closure 在旧 validation 中 GT 点数为 0，表中 mIoU 只平均两个出现的类别。

Hammer 原 checkpoint 与 calibrated checkpoint 使用完全相同的采样 seed，可做逐对象逐 repeat 配对：

- pooled mIoU：+0.0064；
- object-mean paired delta：+0.0067；
- 25 个 object-repeat 中 19 个提高；
- 5 对象 bootstrap 95% CI：[-0.0020, +0.0190]。

因此 clean 结果支持“没有退化且有小幅上升”，但对象数只有 5，不能把它写成已经统计显著的泛化提升。

## 5. 原 checkpoint 的 R2 结果

以下表格使用 reference-normalization 的 pooled confusion mIoU。

### 5.1 独立 support 重采样

| 类别 | Reference | 5000 | 1024 | 512 | 256 | 128 |
|---|---:|---:|---:|---:|---:|---:|
| Hammer | 0.8664 | 0.8700 | 0.8630 | 0.7893 | 0.7107 | 0.5442 |
| Spoon | 0.9653 | 0.9667 | 0.9659 | 0.9501 | 0.8958 | 0.7300 |
| Mug* | 0.9649 | 0.9652 | 0.7822 | 0.6485 | 0.4276 | 0.2330 |

结论：不能宣称存在统一的点数阈值。Spoon 在 1024 点几乎无损，Mug 已明显下降；关键是类别几何、part 尺寸和空间覆盖。

### 5.2 单侧 partial view

| 类别 | Keep 75% | Keep 50% | Keep 25% |
|---|---:|---:|---:|
| Hammer | 0.7073 | 0.6494 | 0.4672 |
| Spoon | 0.9008 | 0.5309 | 0.3121 |
| Mug* | 0.8322 | 0.6590 | 0.4446 |

partial-view weakness 在三个类别上都成立，但难度排序随几何和视角而变。

### 5.3 Dropout 与噪声

- Spoon 的 dropout 和 noise 基本无损；
- Hammer 对 dropout/noise 整体稳定；
- Mug 对 75% dropout 有明显下降，但对 1% radius noise 仍较稳定；
- 因此下一阶段不应把主要训练预算放在 Gaussian noise 上。

## 6. Hammer fixed-query calibration

### 6.1 训练设置

- 初始化：原 Hammer checkpoint（epoch 3714）；
- 冻结：Utonia encoder 与所有非 semantic 分支；
- 训练：semantic adapter、semantic local fusion、semantic decoder；
- paired data：同一 raw query / GT 下构造 clean/full 与 perturbed support；
- perturbations：count 512、count 1024、crop 75%、crop 50%、crop 25%；
- 为支持 batch，观测源点有放回采样到 5000 个 network support 点；归一化参数只由真实观测源点计算；
- loss：clean/perturbed supervised CE + pointwise embedding consistency + 原 checkpoint clean teacher preservation；
- 30 epochs，batch 2，AMP，单张 RTX 4090；
- 选模：robust mIoU 为主，同时奖励 clean mIoU 并惩罚超过容差的 clean drop。

### 6.2 校准内部验证

| 指标 | Source | Best calibrated | Delta |
|---|---:|---:|---:|
| Clean mIoU | 0.8602 | 0.8668 | +0.0066 |
| Robust aggregate mIoU | 0.6767 | 0.7003 | +0.0237 |
| Count 512 | 0.7891 | 0.8562 | +0.0671 |
| Count 1024 | 0.8623 | 0.8879 | +0.0256 |
| Crop 75% | 0.6991 | 0.7300 | +0.0308 |
| Crop 50% | 0.7311 | 0.7797 | +0.0486 |
| Crop 25% | 0.3692 | 0.3598 | -0.0094 |

训练耗时 273.4 秒，平均 9.11 秒/epoch；peak allocated 937.4 MiB、reserved 998 MiB。

### 6.3 使用原始 R2 协议复评

下面不是 calibration 内部指标，而是使用完全不变的原始 R2 evaluator、相同对象和相同随机 seed 的 apples-to-apples 对照。

| 条件 | Reference norm：原 → 校准 | Delta | Support norm：原 → 校准 | Delta |
|---|---:|---:|---:|---:|
| Count 5000 | 0.8700 → 0.8805 | +0.0105 | 0.8598 → 0.8674 | +0.0076 |
| Count 1024 | 0.8630 → 0.8953 | +0.0323 | 0.8544 → 0.8915 | +0.0371 |
| Count 512 | 0.7893 → 0.8540 | +0.0647 | 0.7880 → 0.8607 | +0.0727 |
| Count 256 | 0.7107 → 0.8222 | +0.1115 | 0.7084 → 0.8134 | +0.1050 |
| Count 128 | 0.5442 → 0.6313 | +0.0871 | 0.5307 → 0.6241 | +0.0934 |
| Crop 75% | 0.7073 → 0.7528 | +0.0455 | 0.6849 → 0.7189 | +0.0340 |
| Crop 50% | 0.6494 → 0.6887 | +0.0393 | 0.6758 → 0.7083 | +0.0326 |
| Crop 25% | 0.4672 → 0.5564 | +0.0892 | 0.3699 → 0.3630 | -0.0069 |
| Dropout 75% | 0.8771 → 0.8967 | +0.0196 | 0.8778 → 0.8981 | +0.0203 |
| Noise 1% radius | 0.8843 → 0.8818 | -0.0025 | 0.8824 → 0.8789 | -0.0035 |

稀疏 support 的提升不仅来自 pooled confusion：

- Count 512：25/25 个配对 object-trial 在 reference norm 下提高；object-level paired delta +0.0625，bootstrap 95% CI [+0.0319, +0.1028]；
- Count 256：24/25 提高；+0.1109，95% CI [+0.0693, +0.1403]；
- Count 128：23/25 提高；+0.0903，95% CI [+0.0683, +0.1152]；
- Crop 75%：22/25 提高；+0.0413，95% CI [+0.0176, +0.0806]；
- Crop 50%：18/25 提高；+0.0340，95% CI [+0.0070, +0.0625]；
- Crop 25%：均值提高但对象差异很大，95% CI 跨 0；在真实 support-normalization 下没有可靠改善。

这些置信区间以 5 个对象的 paired mean 为 bootstrap 单元，避免把同一对象的 5 个 trial 当成 25 个独立对象。

## 7. 如何解释这些结果

### 7.1 可以支持的判断

- 原模型对 clean full support、普通 dropout 和小噪声较稳；
- 模型此前没有针对 sparse / partial-view support 训练；
- fixed-query paired calibration 能在不牺牲 clean mIoU 的情况下，明显改善低点数 support 和中等 partial view；
- 平均 embedding cosine 很高并不等于 pointwise part prediction 稳定，必须同时报告 mIoU / label agreement；
- support coverage 比裸点数更重要，点数阈值依赖类别。

### 7.2 不能支持的判断

- 不能称 R1 为 independent held-out test；
- 不能把 Mug 的 0.9636 写成完整三类 mIoU；
- 不能声称极端 partial view 已解决；
- 不能把内部 calibration validation 的提升当成下游 policy success；
- 不能只凭 5 个 Hammer val 对象宣称统计上普适。

### 7.3 当前最重要的技术发现

极端 25% crop 在 reference-normalization 下明显改善、在当前 support-normalization 下却没有改善。这说明 semantic head 已学到一部分缺失观测鲁棒性，但当前由 partial support 决定的中心/尺度/z-shift，以及 support-dependent tri-plane AABB，会放大坐标和查询漂移。

下一版结构实验优先级应为：

1. 使用稳定的外部/机器人坐标归一化，或由 clean/reference frame 提供 normalization；
2. 给 tri-plane 使用固定 canonical bounds，或训练时显式随机化并正则化 bounds；
3. 再评估 extreme crop，而不是继续单纯增加 crop loss 权重。

## 8. 对论文实验章节的直接修改

建议新增三张表：

1. Part Representation Quality：三类别 R1 accuracy、mIoU、per-part IoU、对象数和 part coverage；
2. Fixed-query Stability：support count、dropout、noise、partial view，报告 pooled mIoU 与 object-level mean ± CI；
3. Consistency Calibration Ablation：source、CE-only、CE+consistency、完整 teacher-preserved calibration，并同时列 clean 和 robust 指标。

建议新增一幅图：

- 横轴为 support count / crop keep ratio；
- 纵轴为 mIoU；
- 对比 source 与 calibrated；
- Hammer / Spoon / Mug 分面；
- 图注明 fixed query、5 object-level trials，以及 Mug Closure N/A。

## 9. 下一步实验顺序

P0：

1. 实现严格 train/val/test manifest，并验证每个 split 的每个 part 都有覆盖；
2. 用正式 split 重训，而不是在旧 checkpoint-selected val 上继续报数；
3. 下游策略做 matched XYZ / Utonia / semantic embedding / part-probability 对照；
4. 每个策略条件至少 3 个训练 seed，固定 episode 列表，并报告 success、Wilson CI 与 paired difference；
5. 对 extreme partial view 做 normalization / fixed-bounds 结构消融。

P1：

1. 在真实/仿真相机 partial point cloud 上复现 R2，而不是只用 mesh crop；
2. 增加 cross-category 和 unseen-instance policy generalization；
3. 做 semantic label corruption / embedding shuffle sanity check，证明策略增益来自语义结构。

## 10. 产物索引

代码：

- policy/DP3/scripts/evaluate_semantic_field.py
- policy/DP3/scripts/benchmark_semantic_field_stability.py
- policy/DP3/scripts/semantic_field_eval_utils.py
- policy/DP3/scripts/semantic_support_perturbations.py
- policy/DP3/scripts/calibrate_semantic_field.py
- policy/DP3/scripts/test_semantic_field_eval_utils.py

主要结果：

- outputs/paper_revision/r1_val_diagnostic/
- outputs/paper_revision/r2_val_diagnostic/
- outputs/paper_revision/calibration_pilot/hammer_fixed_query_e30_seed20260805/
- outputs/paper_revision/calibration_pilot/r1_hammer_calibrated_e30_seed20260805_r5_fp32/
- outputs/paper_revision/calibration_pilot/r2_hammer_calibrated_e30_seed20260805_t5_fp32/

所有 evaluator/calibration 输出都包含 checkpoint/config hash、split records、运行环境、GPU、Git 状态、逐 trial CSV 和可重算 confusion matrix。

## 11. Matched policy 对照的工程进度

Beat Cube 已从同一 80-episode / 2774-frame semantic zarr 派生四条策略输入路线。四条路线使用同一 20 维 EEF absolute-6D action 配置、seed=20260805、batch=8，并完成 1 train step + 1 validation step 的端到端 smoke：

| 表征 | 额外点形状 | DP3参数量 | smoke val loss | 当前证据等级 |
|---|---:|---:|---:|---|
| semantic field embedding | `[128,131]` | 273.7122M | 0.2480 | pipeline only |
| XYZ only | `[128,3]` | 273.7040M | 0.2789 | pipeline only |
| XYZ + part probability | `[128,5]` | 273.7041M | 0.2405 | pipeline only |
| XYZ + frozen Utonia feature | `[128,579]` | 273.7409M | 0.2323 | pipeline only；support非正式 |

一次更新后的 loss 不能用于方法排序。这里的结论仅是四路数据、shape meta、20维action、PointNet编码器和优化器均可运行，且参数量差异相对273M主干很小。

XYZ和part-prob严格保留原semantic query xyz，action/state/main point cloud/episode boundaries逐元素不变。Utonia也实现了固定query接口；一帧512点support前向约0.082s、峰值约552MiB，完整2774帧约183s。

当前Beat原始演示链接指向的目录不存在：

`/media/zheng/Extreme SSD/geo_mani_data/beat_cube/robotwin_objpc/demo_real_zed_sam2_objpc_global`

因此现有完整Utonia zarr只能从merged context中选取离query最近的512点来做工程smoke。其manifest强制写入 `formal_comparison_eligible=false`，不得进入论文表格。正式Utonia对照必须在恢复原始 `{A}` object point cloud后重新生成；builder的raw模式会在原始数据缺失时于任何写入前安全失败。

新增代码与产物：

- `policy/DP3/scripts/build_semantic_policy_ablation_zarr.py`
- `policy/DP3/scripts/matched_utonia_feature_utils.py`
- `policy/DP3/scripts/build_matched_utonia_policy_zarr.py`
- `policy/DP3/scripts/test_semantic_policy_ablation_zarr.py`
- `policy/DP3/scripts/test_matched_utonia_feature_utils.py`
- `policy/DP3/data/beat_cube-demo_real_zed_sam2_objpc_global-80-objpc-utonia-pointwise-hybrid-matchedquery-contextsmoke-eef-absolute6d-global_meta.json`

## 12. Matched policy e300 离线结果

完整协议、raw/EMA配对结果、置信区间、限制与下一轮决策见 `MATCHED_POLICY_PILOT_RESULTS_20260806_zh.md`。

单seed早期诊断的排序为 part-prob < field < XYZ；它支持part-aware interface相对纯坐标的价值，但尚不支持高维连续embedding优于显式part probability，也不能替代闭环成功率。
