# CoRL v0 拒稿诊断、论文重构与补实验执行计划

> 日期：2026-08-05
> 建议主目标：ICRA 2027（2026-09-15 截止）
> 备选：RA-L；完整扩展版：T-RO
> 本文档基于实际匿名投稿稿、当前修订稿、三份评审、视频稿，以及物体表征与 DP3 的完整代码/配置/现有 checkpoint 审计。

## 1. 一句话结论

这项工作不是因为方向错误而被拒：三位审稿人都认可问题重要、support/query 设计直观、冻结 Utonia 后对 DP3 的实物收益明显；它被拒的核心原因是**方法定位大于现有证据、关键消融和表征定量评测缺失、baseline 不够强且不完全公平**。

代码审计又发现了比 review 更紧急的问题：论文中的损失权重、语义点数、action 维度、坐标系和“仅改变表征”的说法与实际实现不一致；当前稳定性损失也不是文中描述的“固定 query、只改变 support”。在这些问题修正前，直接补更多 success rate 不会形成可信的下一稿。

此外，2026 年 2 月公开的 [PA3FF/PADP](https://arxiv.org/abs/2602.14193) 与当前“part-aware continuous 3D field + diffusion policy + generalizable manipulation”叙事高度重叠。下一稿必须主动收窄并验证差异，不能继续依赖“首个 part-aware continuous field”式的新颖性表述。

## 2. 已完整审阅的材料

### 2.1 论文与评审

- [三位审稿意见与 meta-review](review_comment.md)
- [当前 16 页本地修订稿](Beyond_Point_Attached_Semantics__Object_Centric_Semantic_Fields_for_Generalizable_Manipulation.pdf)
- 实际送审的 10 页匿名版本：`include/3d_semantic_train/reference/corl2026_version0_comment.pdf`
- 更早的语义场稿：`include/3d_semantic_train/reference/semantic_field_corl.pdf`
- 13 页视频稿：`include/3d_semantic_train/reference/corl_version0_视频.pdf`
- 更早的 Semantic Skill Fields/NDF 方向：`include/3d_semantic_train/reference/previous.pdf`

需要特别区分：三位审稿人评的是 10 页匿名版，而不是当前 16 页版本。当前稿增加了附录文字，但没有补齐决定性实验；同时引入了若干与代码冲突的数字。

### 2.2 物体表征链路

- 训练入口：`include/3d_semantic_train/semantic_field_release/scripts/train_semantic_field.sh`
- 正式训练逻辑：`include/3d_semantic_train/semantic_field_release/train/train_utonia_universal_field.py`
- 网络：`include/3d_semantic_train/semantic_field_release/models/universal_field/utonia_universal_field.py`
- 损失：`include/3d_semantic_train/semantic_field_release/models/universal_field/losses.py`
- PartNext 数据：`include/3d_semantic_train/semantic_field_release/my_datasets/partnext_universal_field.py`
- 导出：`include/3d_semantic_train/semantic_field_release/scripts/export_semantic_point_cloud.py`
- 开发版同名模型、损失和训练代码，以及现有 query-count 可视化工具。

### 2.3 策略链路

- 用户指定入口：`policy/DP3/train_semantic_pointwise_hybrid_eef_absolute6d_global.sh`
- 数据入口：`policy/DP3/process_data_semantic_pointwise_hybrid_eef_absolute6d_global.sh`
- 实际预处理：`policy/DP3/scripts/process_data_semantic_pointwise_hybrid_eef_absolute6d_global.py`
- 语义特征：`policy/DP3/scripts/semantic_feature_utils.py`
- EEF action：`policy/DP3/scripts/eef_action_utils.py`
- 部署：`policy/DP3/deploy_policy.py`
- 实物入口：`policy/DP3/real_infer_semantic_pointwise_hybrid_eef_absolute6d_global.sh`
- 点式 Utonia、DINO 可视化、Hydra task/policy 配置、dataset、PointNet encoder、训练/验证/评估脚本。

## 3. 三位审稿人的真实共识

| 共识 | 评审看到的优点 | 导致 Borderline/Reject 的缺口 |
|---|---|---|
| 问题有价值 | 语义、几何与控制之间存在实际接口问题 | 贡献像 Utonia、tri-plane、PartNext 和 DP3 的系统集成 |
| support/query 设计合理 | 比“语义附着在原始点上”更灵活 | 没有证明该设计本身产生了收益 |
| 仿真和实物结果有潜力 | 实物提升尤其明显 | trial 太少、无置信区间、无 seed、无失败统计 |
| 冻结同一 3D backbone 是好思路 | 有机会隔离表征差异 | 实际 baseline 的输入、encoder、action schema 并未完全匹配 |
| 论文方向可救 | 三人均给 4/Borderline | 缺三项 loss 消融、part supervision 对照、continuous field 对照和直接表征指标 |

审稿意见可归纳为五个必须回答的问题：

1. 三项损失各自是否必要？
2. continuous query/readout 是否真的比 point-attached feature 好？
3. 表征本身是否在 unseen instance 上更一致、更可对应，而不仅是 PCA 颜色好看？
4. 2D/3D baseline 是否足够强、公平且可复现？
5. 方法对部分观测、噪声、分割错误、类别扩展和语义歧义是否稳健？

## 4. 论文主张—证据—代码事实矩阵

| 当前主张 | 当前证据 | 代码事实/风险 | 下一稿处理 |
|---|---|---|---|
| continuous query 把语义从 noisy points 解耦 | shared-PCA 定性图、策略成功率 | DP3 中 query 是同一观测云的随机子集或 FPS，并非独立 clean/canonical queries | 收窄为“support-conditioned query readout”；新增固定 query、改变 support 的定量测试 |
| stability loss 固定 query、只改变 support | 无消融 | 当前 support 与 query 都做独立 SO(3) 旋转和 jitter；同 index 不等于同坐标 | 修实现或重写为 transformation consistency；优先比较 no/current/fixed-query 三种版本 |
| 仅表征发生改变 | 主表比较 DP3/2D/3D/ours | ours 为 A/B 各加独立 PointNet，raw DP3 没有同结构分支；参数量和输入模态不同 | 增加 capacity-matched XYZ control，并统一 action、输入和 encoder |
| Utonia point-wise 为公平 3D baseline | 同为 frozen Utonia | baseline 用真实 RGB+kNN normals+FPS；ours 默认常量颜色+radial normals+random；现有 baseline 多为 joint14 | 实现同一 EEF20/global、同预处理、同 query index/count 的 matched route |
| 2D feature 不适合精确控制 | 仿真表一行 2D baseline | 当前 DINO 仅 224 图像的 16×16 patch nearest lifting，无 z-buffer/visibility；无实物 policy route | 换成 visibility-aware DINOv2 lifting；最好增加 D3Fields/F3RM/GenDP 型 3D fusion |
| 学到跨实例 part semantics | 策略泛化和 PCA 颜色 | validation 只报 overall accuracy；无 mIoU、对应或真正稳定性 | 增加 held-out-instance mIoU、feature stability 和 correspondence |
| generalizable manipulation | 4 个任务、有限物体 | 实际主要是类别内未见实例；每类别依赖 PartNext 标签 | 全文改为 within-category cross-instance；跨类别只在新增统一模型实验后声明 |
| policy 使用 256 semantic points | 论文实现细节 | 已恢复的 Beat/Stir zarr meta 均为 128；实物 wrapper 默认 256，可能 train/test 不一致 | 以 manifest 固定并核对；主表全部同点数，另做 64/128/256 点数消融 |
| action/state 都是 14D | 论文表格 | global EEF 路线 state 14D、action 20D（xyz+continuous rot6d+gripper/arm） | 更正表格；仿真旧 joint14 与实物 EEF20 分开陈述或统一重训 |
| world-frame absolute action | 论文文字 | wrapper 使用 `reference_camera`；输出点云 frame 为 `source` | 改成准确坐标系，或显式改实现为统一 workspace frame |
| 语义 CE 权重为 0.5 | 当前稿表格 | CLI/config/checkpoint 均为 1.0；contrastive 0.2、consistency 0.1 | 改为 1.0/0.2/0.1，并用 manifest/checkpoint metadata 自动生成表格 |
| multi-task policy | 稿件措辞 | 指定入口一次训练一个 task | 改成 per-task policies；除非新增真正的 multi-task 实验 |

## 5. 额外的复现性与可信度风险

### 5.1 表征训练

- 模型是 frozen Utonia + 256D 两层 adapter + 三个独立 xy/xz/yz plane splat + fusion + 128D L2 embedding/logits；Fig. 2 目前容易让人误以为 plane 之间存在派生关系。
- tri-plane 坐标由 support 的 axis-aligned min/max 决定。support 被裁剪或部分缺失时，query 的归一化坐标也会变化；这是必须实测的鲁棒性来源。
- train/val 为默认 90/10，没有独立 test；best checkpoint 按 overall semantic accuracy 选择，而不是 macro mIoU。
- validation 的 view1/view2 都是 identity/no-jitter，当前记录的 `sem_consistency≈0` 是相同输入比较，不能作为稳定性证据。
- supervised contrastive 在整个 batch 上展开，同标签跨实例互为 positive；这是有意义的跨实例监督，但也混入大量同实例 positive，需要在方法中准确说明。
- pose/occupancy branch 被实例化并写入 checkpoint，但 semantic mode 下冻结，论文不应暗示联合训练。
- 现有命名存在 provenance 风险：例如 `Hammer_semantic/config.json` 实际是 Microwave_Oven；release hammer checkpoint 仅 epoch 1，是 smoke run，不是正式结果。
- 正式 checkpoint 恢复结果：Hammer best epoch 3714/val acc 0.857，Mug epoch 3964/0.834，Spoon epoch 54/0.665；Spoon 最终 epoch 的 val acc 降至约 0.470，存在明显过拟合/选择敏感性。

### 5.2 DP3 数据与训练

- Beat Cube meta 有 80/80 episodes；Stirring 文件名写 80，但实际只有 70 个 episode record。论文必须报告真实数目。
- 当前默认 `semantic_input_color_mode=debug_placeholder`、`semantic_forward_mode=reference`：对象 RGB 被替换为常量红/蓝，normal 为 radial fallback；这与 field 训练时 mesh color/normal 有 domain gap。
- reference 模式在离线 zarr 生成时只随机一次，在线部署却可能每帧重采样，形成随机 train/deploy mismatch。
- Utonia checkpoint 路径在工具中存在 hard-code/参数未完全生效的风险，必须把 SHA256 写入 manifest。
- validation ratio 仅 2%；50 demos 通常约 1 个 validation episode；每次验证最多 2 batches。它不足以支撑 checkpoint 选择。
- normalizer 在完整 replay buffer 上 fit，包含 validation episode，存在轻微泄漏。
- rollout 在训练脚本中关闭，而 top-k monitor 指向 `test_mean_score`；必须确认最终主表到底使用 last、EMA 还是 best checkpoint。
- run directory 只含 task/exp/seed，遗漏 checkpoint、query count、preprocessing 等字段，同时 `resume=true`，容易错误续训或覆盖。

### 5.3 现有结果不可追溯处

- 仿真表未记录 rollout 数量、training seeds、paired evaluation seeds、置信区间或逐 trial 文件。
- 实物每格 20 次，只有 aggregate success count。以 Wilson 95% CI 粗看，17/20 约为 [0.64, 0.95]，10/20 约为 [0.30, 0.70]，7/20 约为 [0.18, 0.57]；区间很宽。
- `eval_policy.py` 和实物入口未保存完整 trial-level success、初始条件、对象实例、失败阶段和模型 manifest，无法做配对统计或失败归因。
- 仿真结果很可能来自早期 joint14/non-hybrid，实物结果来自后期 EEF20/hybrid；下一稿不能把它们描述成完全同一 pipeline。
- DP3 引用编号有误：正文称 DP3 [5]，参考文献 [4] 才是 3D Diffusion Policy，[5] 是 3D Diffuser Actor。

## 6. 下一稿应如何重新定位

### 6.1 推荐核心命题

不再声称“我们提出了通用的 part-aware continuous 3D semantic field”。推荐改为：

> 在类别内跨实例操作中，冻结通用 3D backbone，并用少量 part supervision 学习一个 support-conditioned query readout，可把不稳定的点附着特征转换为可由策略一致消费的对象中心语义接口；其收益在 matched perception/control 设置下可由表征稳定性和闭环成功率共同验证。

这一定义把贡献落在当前实现真正可能防守的四点：

1. 冻结 backbone 的轻量适配，而不是从头训练大型 part foundation model。
2. support/query 接口与 feature readout，而不是宣称 continuous field 本身首次出现。
3. 直接测量 sampling/partial-observation stability，并连接到 policy gain。
4. 可插入点云 diffusion policy 的对象级语义分支，尤其适合双对象、工具使用和双臂任务。

### 6.2 可用标题

首选：

> **Support-Conditioned Part-Semantic Readouts for Cross-Instance 3D Manipulation**

备选：

> **From Point-Attached Features to Queryable Part Semantics for Cross-Instance Manipulation**

除非新增真正跨类别实验，不建议标题继续使用宽泛的 “Generalizable Manipulation”。

### 6.3 贡献段落必须与实验一一对应

建议只保留三条贡献：

1. support-conditioned、queryable 的 part-semantic readout，并明确 frozen backbone、adapter、tri-plane 和监督范围。
2. 一个定量 protocol，测量 held-out instance segmentation/correspondence 以及 support 扰动下固定 query 的特征稳定性。
3. 在 matched 3D diffusion policy 中证明表征稳定性与跨实例闭环收益相关，覆盖仿真与实物。

如果实验没有完成，就不能在 contribution 中出现对应主张。

## 7. 实验总矩阵

优先级定义：P0 为再次投稿前必须完成；P1 为 ICRA 强烈建议；P2 为期刊扩展。

### E0：实验 provenance 与事实校准（P0，所有实验的前置条件）

**目的**：确保以后每个数字能追到数据、代码和权重。

每个 run 自动保存：

- git commit、dirty diff 标识、hostname、时间、CUDA/PyTorch 版本；
- task、demo episodes、train/val object IDs、training/eval seeds；
- field/policy checkpoint absolute path + SHA256；
- support/query 点数与采样法、RGB/normal 模式、坐标系；
- state/action schema、encoder 参数量、EMA/last/best 选择；
- trial-level JSONL 和最终汇总 CSV。

**验收**：论文任意表格单元可以由一条 manifest + 一组 trial records 重算。

### R1：表征准确性与跨实例语义（P0）

**类别**：优先 Mug、Hammer、Spoon；它们已有权重，且分别覆盖容器/工具和当前策略任务。
**split**：固定 train/val/test object-instance IDs；test 实例从不参与 checkpoint 选择。
**指标**：macro mIoU、per-part IoU、balanced accuracy、precision/recall、confusion matrix；同时报告 mean±std/CI。
**对照**：raw Utonia linear probe、当前 field logits/embedding kNN、DINOv2 visibility-aware lifting；若可运行，加入 PA3FF。

**关键判断**：若 field 只提高 overall accuracy 而 macro mIoU/少数 part 无提升，就不能声称更好的 part semantics。

### R2：固定 query 的 support 稳定性（P0，最能回答核心 review）

对同一完整 mesh/点云固定一组 surface query coordinates，只改变 support：

- support count：128/256/512/1024/5000；
- random resampling：至少 10 次；
- dropout：0/25/50/75%；
- partial view/crop：完整、双视角、单视角、遮挡功能部件；
- Gaussian noise：按传感器尺度设置 3 档；
- outlier/segmentation contamination：5/10/20%；
- 坐标变换：旋转、平移与尺度扰动，分开报告。

**指标**：同 query 的 cosine similarity、L2 feature drift、label agreement、mIoU drop、worst-part drop，以及置信度校准。
**对照**：raw Utonia point feature、当前 field、no-consistency field、修正后的 fixed-query consistency field、visibility-aware DINO。
**图**：扰动强度—性能曲线，而不是只放 PCA 彩色点云。

**注意**：这项实验验证的是 field 能否在 support 变化下提供稳定 readout；它不能自动证明当前 DP3 已使用“clean arbitrary queries”。论文必须保持这一边界。

### R3：跨实例 correspondence（P1）

两级评测：

1. semantic correspondence：给定 source query，在 target instance 中检索同 part，报告 top-1、mAP、Recall@k；
2. within-part correspondence：在同 part 内用归一化局部坐标、人工关键点或 PartNext 对应标注，报告 normalized geodesic/Euclidean error。

对照 raw Utonia、DINOv2、field embedding、field logits/probability、PA3FF（若可用）。只做 part top-1 容易被粗标签饱和，不能单独作为 correspondence 结论。

### A1：三项损失与稳定性定义消融（P0）

最少五组：

| 组 | CE | SupCon | Consistency | 目的 |
|---|---:|---:|---:|---|
| Full-fixed | 1.0 | 0.2 | 0.1，固定 query/改变 support | 推荐修正后的完整方法 |
| No-CE | 0 | 0.2 | 0.1 | 回答 part classifier supervision 必要性 |
| No-SupCon | 1.0 | 0 | 0.1 | 回答跨实例 metric structure 必要性 |
| No-Cons | 1.0 | 0.2 | 0 | 回答稳定性 loss 必要性 |
| Current-transform | 1.0 | 0.2 | 0.1，当前双视图实现 | 区分变换一致性与 support 稳定性 |

**资源策略**：先 Hammer、1 seed 做 20–30% 训练进度的校准；只有 Full-fixed 在 R1/R2 上不劣于当前实现才启动完整训练。正式结论至少在一个代表类别做 3 seeds；Mug/Spoon 可先各 1 seed 作外部有效性，资源允许再补齐。

### A2：到底是 embedding、part label，还是 query readout 起作用（P0）

在完全相同的 A/B object branches、PointNet 容量、point count、preprocessing 和 action schema 下比较：

1. XYZ-only resampled object points；
2. raw frozen Utonia point-wise feature；
3. direct per-point part logits/probabilities；
4. learned 128D field embedding；
5. field embedding + logits（可选）；
6. current random observed-point queries vs FPS observed-point queries；
7. no-resampling/attached readout，即在原 support point 上输出而不另取 query subset。

这一矩阵能直接回答 R2 提出的“predicted part probabilities vs embedding”“query only original points/no resampling”问题。

### B1：matched 3D baselines（P0）

对 DP3、XYZ-only、Utonia point-wise、ours 强制统一：

- 同一 demonstrations 和 train/val episode split；
- 同一 EEF absolute6D global action（state14/action20）或全部 joint14，不能混用；
- 同一 object masks、RGB、normal、query indices、128/256 点数；
- 同样数量的 object branches 和 PointNet width；
- 同一 policy seeds、training steps、EMA/checkpoint 选择；
- 报告参数量、训练时间和在线 feature latency。

如果 raw DP3 只有 scene branch，则必须增加 `DP3 + object XYZ branches` 作为 capacity control，避免把额外网络容量误认为语义收益。

### B2：强 2D/3D 语义 baseline（P0/P1）

最低可接受 2D baseline：

- 固定并记录 DINOv2 model/version；
- 原始分辨率或明确 resize；
- 相机内外参、depth reprojection、z-buffer/visibility check；
- 多视角按可见性融合，不对遮挡点盲目平均；
- 无效点不能用 object mean feature 静默填充；
- 建立可复现 policy preprocess/train/eval route，而非只做可视化。

更强的 P1 baseline：GenDP、D3Fields/F3RM 风格的 persistent 3D fusion，或 PA3FF/PADP。若第三方代码不能及时稳定运行，论文必须说明未比较的原因，不能由弱 DINO lifting 推出“2D 方法本质不适合”的结论。

### P1：仿真闭环主表（P0）

**任务**：保留现有 4 个任务；完整 baseline 主表跑全部任务，昂贵消融优先 Beat Cube 和 Stirring 两个代表任务。
**训练**：每方法/任务至少 3 个 policy seeds。
**评测**：每 seed 100 个 paired initial-condition seeds；所有方法使用相同对象实例、初始姿态和环境随机量。
**报告**：每 task success、macro average、95% hierarchical bootstrap CI；同时给出对象实例分组结果。
**检验**：paired bootstrap 为主；同一 policy seed 下可用 McNemar，跨 training seeds 用分层/混合效应汇总，避免把 300 rollout 当成 300 个独立训练重复。

除 success 外，记录可解释的阶段指标：grasp/first-contact、functional-part contact、transport、final relation、collision/timeout。失败视频按这些类型汇总。

### P2：实物闭环（P0/P1）

最低做三组：capacity-matched XYZ、matched Utonia、Full-fixed field；若资源允许再加 visibility-aware DINO/PA3FF。
每个任务至少 20 次只是最低门槛，建议 30 次；采用 block-randomized paired protocol：同一对象、初始位姿和场景条件形成一个 block，各方法随机顺序执行。
报告成功率、Wilson 95% CI、对象实例分组、阶段失败类型和平均推理延迟。统计以 paired block bootstrap/条件 logistic regression 为主；若无法严格配对，使用 Fisher exact 并明确限制。

实物视频必须包含：任务名、train/test object、方法名、是否成功、关键失败原因；原视频的模糊、文字过密和无旁白问题需要彻底重做。

### S1：类别扩展与标注效率（P1/P2）

比较 per-category field 与统一 multi-category field：

- 2 类、3 类，逐步扩展；
- 100/50/25/10% part labels；
- seen category/unseen instance 与 unseen category 分开；
- 报告 mIoU、稳定性、policy success、参数量和训练成本。

当前多类别 run 只有很早的 incomplete checkpoint，不能作为 scalability 证据。若 ICRA 时间不足，正文把“scalable”改为 limitation；T-RO 扩展版再完成此实验。

### S2：语义歧义与失败边界（P2）

- 同一几何区域因任务而有不同功能语义；
- 标签层级/粒度冲突；
- 无明确 functional part；
- articulated/deformable parts；
- SAM2 mask 漏检、粘连和 distractor contamination。

这组实验决定该方法究竟是 category-part representation，还是可扩展为 task-conditioned functional semantics。当前方法属于前者，论文应明确承认。

## 8. ICRA 2027 最小可投稿实验包

ICRA 2027 官方截止为 [2026-09-15 23:59 PST](https://2027.ieee-icra.org/announcements/call-for-technical-papers/)，当前官方说明初稿完整内容限 [8 页](https://2027.ieee-icra.org/contribute/call-for-icra-2027-papers-now-accepting-submissions/)。建议只承诺以下闭环：

1. E0 manifest/logger 和论文事实修正。
2. Mug/Hammer/Spoon 的 held-out-instance mIoU；Hammer 完整 loss ablation。
3. 固定 query 的 support count/dropout/partial-view 稳定性曲线。
4. capacity-matched XYZ、Utonia、part probability、field embedding 四组。
5. 4 task 主表 × 3 policy seeds × 100 paired rollouts；消融只跑 2 个任务。
6. 至少 2 个代表实物任务，matched 三方法、block-randomized trials。
7. 重做 2–3 分钟清晰视频。
8. 正面讨论 PA3FF/PADP；能跑则加结果，不能跑则清楚划界和规模差异。

**ICRA 止损门**：若 2026-08-24 前尚未得到 matched Utonia/XYZ 的三 seed 仿真结果，或 fixed-query loss 未在 R2 上优于 no-consistency，则停止追赶 ICRA，转为滚动期刊路线，避免再次提交证据不足的稿件。

## 9. 期刊扩展包

### RA-L

[RA-L](https://www.ieee-ras.org/publications/ra-l/ra-l-information-for-authors/) 滚动投稿，6 页并最多 2 页付费超页，定位是及时、简洁的机器人创新。它不是“实验可以少一些”的出口：版面短意味着主张必须更窄、关键消融必须在正文。适合 ICRA 最小闭环完成但错过 deadline 后投稿。

### T-RO

[T-RO](https://www.ieee-ras.org/publications/t-ro/t-ro-information-for-authors/) 常规初稿最多 18 页，适合完整版本：统一多类别、标注效率、系统鲁棒性、更多实物任务、correspondence/segmentation 下游，以及 support/query 机制的更深入分析。只有完成 S1/S2 和更全面强 baseline 后才建议投稿；不能只是当前稿件加几张消融表。

RSS 2027 和 IROS 2027 的官方 CFP 当前未核实到，因此不使用往年日期安排关键路径。

## 10. 论文结构重写建议

### 10.1 八页 ICRA 结构

1. **Introduction（0.8 页）**：问题、严格范围、三条可验证贡献。
2. **Related Work（0.6 页）**：continuous fields、3D foundation features、part-aware manipulation；正面加入 PA3FF/PADP。
3. **Method（1.7 页）**：support/query 定义、frozen Utonia adapter、tri-plane readout、三损失、policy interface。
4. **Representation Experiments（1.3 页）**：mIoU、fixed-query stability、loss/readout ablation。
5. **Policy Experiments（1.8 页）**：matched baselines、仿真/实物、统计与失败分析。
6. **Limitations/Conclusion（0.4 页）**。
7. 剩余空间用于图表和参考文献；按官方 8 页完整限制预留。

### 10.2 Fig. 2 重画

只画一条清楚的数据流：

`support points → frozen Utonia → adapter → independent XY/XZ/YZ splat → query interpolation/fusion → embedding/logits`

旁边单独画两种训练 view，并明确：哪些变换作用于 support、哪些作用于 query、坐标如何对应。标注 `d_u=256`、`d_s=128`，说明 adapter 和 fusion 层。不要用容易被理解成“一个 plane 推导另一个 plane”的箭头。

### 10.3 Experimental Setup 必须进入正文的内容

- 精确 train/test object IDs 和数量；
- 每任务 demonstrations、实际 episodes（Stirring 是 70，不是 80）；
- state/action 维度和坐标系；
- semantic point count 及 train/test 是否一致；
- baseline 共同的 preprocessing；
- seeds、rollouts、checkpoint selection、CI；
- field 与 policy 参数量、训练/推理时间。

### 10.4 限制必须主动写

- 依赖 per-category PartNext supervision；
- 当前不是 open-vocabulary，也不是真正 unseen-category 泛化；
- support AABB normalization 对 partial observation 敏感；
- 对 segmentation/SAM2 质量有依赖；
- part taxonomy 未表达 task-dependent semantics；
- query 在当前 policy 中仍取自 observed surface points。

主动写清限制会提高可信度，也防止审稿人用更宽的主张攻击论文。

## 11. 工程实现顺序

### 第 0 步：冻结旧结果，不覆盖

- 现有 zarr/checkpoint 全部只读保留。
- 为每个旧主表数字建立 `legacy_result_provenance.csv`：能确认的写路径/hash，不能确认的标 `unknown`，不猜测。
- 修复/重命名 Hammer/Microwave、Stir70/80 等混淆；旧文件不删除，使用 manifest 加 alias。

### 第 1 步：表征 evaluator

建议新增：

- `include/3d_semantic_train/tools/evaluate_universal_field.py`
- `include/3d_semantic_train/tools/benchmark_universal_field_stability.py`
- `include/3d_semantic_train/tools/evaluate_universal_field_correspondence.py`

复用 `include/3d_semantic_train/semantic_field_release/myutils/training.py` 的 IoU 辅助，以及 `include/3d_semantic_train/tools/visualize_utonia_universal_field_query_count_stability.py` 的模型/数据加载；但固定 query，输出 JSON/CSV，而非只输出 PCA 图。

### 第 2 步：修正 consistency loss 与训练选择

- 在 dataset 显式产生 `support_view1/support_view2/fixed_query`；禁止依赖“相同 index”暗示坐标相同。
- validation 也实际做 resample/crop/noise，而不是 identity-vs-identity。
- checkpoint monitor 改为 val macro mIoU + stability composite，至少同时保存 best_miou、best_stability、last。
- 所有 loss weight 与 checkpoint metadata 写入 config，不从论文手填。

### 第 3 步：统一策略矩阵

建议新增 declarative 配置：

- `policy/DP3/experiments/semantic_field_matrix.yaml`
- `policy/DP3/scripts/run_semantic_field_matrix.py`

统一调用现有 `process_data_*`/`train_*`，run ID 必须包含 task、route、field SHA、point count、preprocess、action schema、seed。默认禁止模糊 `resume=true`；只有 manifest 完全一致才允许 resume。

### 第 4 步：matched baselines

- 为 Utonia 补齐 EEF absolute6D global route。
- semantic/Utonia/XYZ 共享 query indices 和 preprocessing helper。
- 在当前 DINO 可视化 `script/visualize_semantic_field_on_dataset.py` 基础上实现 visibility-aware lifting 和正式 zarr route。
- 报告每路 encoder 参数量与 latency。

### 第 5 步：结构化评估

修改 `script/eval_policy.py`、`policy/DP3/deploy_policy.py` 和实物入口，逐 trial 保存：

- run/field/policy manifest IDs；
- task、training seed、evaluation seed、object IDs、initial condition；
- success、stage successes、failure code、termination；
- inference timing、可选末端几何误差；
- video 文件映射。

只有在第 1–5 步 smoke test 和一个小规模 calibration 通过后，才启动完整 GPU 矩阵。

## 12. 建议时间表（ICRA 路线）

| 日期 | 交付物 | Go/No-Go |
|---|---|---|
| 8/05–8/08 | manifest、事实校准、表征 evaluator、旧 checkpoint 基准 | 能自动产出 mIoU/stability JSON |
| 8/09–8/12 | fixed-query consistency 修正；Hammer 小规模 loss calibration | Full-fixed 在 stability 上有清晰增益 |
| 8/13–8/18 | matched XYZ/Utonia/part-prob/field 数据与 2-task pilot | 输入/action/参数量审计通过 |
| 8/19–8/24 | 4 task × 3 seed 主训练/评估；representation 正式表 | 未达成则退出 ICRA 冲刺 |
| 8/25–8/31 | 实物 block trials、失败分析、补强 baseline | trial records 完整 |
| 9/01–9/06 | 八页初稿、图表、视频第一版 | 所有 contribution 有对应证据 |
| 9/07–9/11 | 内部 review、统计复核、复现实跑 | 表格可由原始记录重算 |
| 9/12–9/14 | 匿名化、格式、视频和最终检查 | 不在最后一天训练新模型 |
| 9/15 | 投稿 | 仅提交已验证版本 |

## 13. 第一轮实验的具体启动顺序

在正式补实验时，建议我们按下面顺序一起做：

1. 先实现 R1/R2 evaluator，用现有 Hammer/Mug/Spoon checkpoint 得到 baseline 数字。
2. 修 fixed-query consistency，Hammer 跑 Full-fixed/No-Cons/Current-transform 三组短训练。
3. 若 R2 证明 Full-fixed 有增益，再补 No-CE/No-SupCon 和正式 seeds。
4. 同时统一 XYZ/Utonia/field 的 EEF20 preprocess，并只在 Beat Cube 做一轮 policy pilot。
5. pilot 排除 pipeline 问题后再展开 4 tasks × 3 seeds。

这一顺序的好处是：最便宜的表征实验先验证论文最核心机制；如果机制不成立，可以及时重定位，不会先消耗数百 GPU 小时训练 policy。

## 14. 开始补实验前需要确认/找回的外部信息

这些信息在本地材料中无法可靠恢复，但不阻碍先实现 evaluator：

- 原论文仿真 Table 1 每个数字对应的 zarr、policy checkpoint、rollout 数量和 action route；
- Grasp Mug / Pour Water 的 semantic meta、权重和策略产物是否只在服务器；
- 真实实验使用的最终 query count 是 128 还是 256；
- 可并行使用的 GPU 数量、型号和每天预算；
- PartNext 训练/验证 object ID 列表是否有外部版本；
- 是否能拿到 PA3FF/PADP、GenDP 或 D3Fields 的可运行代码/权重；
- 实物机器人在 8/25 前可用的时间窗口。

## 15. 最终判断

当前工作仍值得救，而且三位 Borderline 已经说明它距离可接收不是“另起炉灶”的距离。但必须从一篇主要靠策略 success rate 支撑的系统稿，变成一篇**主张精确、实现一致、直接证明表征性质、baseline 完全 matched、统计可复现**的论文。

最优策略不是把当前 16 页稿继续润色，而是：先校准事实和实验协议；用 R1/R2/A1 证明 support-conditioned readout 的独立价值；再用 matched DP3 实验把该价值连接到闭环控制。若这条证据链成立，ICRA 2027 有现实机会；若 8 月下旬仍未成立，则以更完整的 RA-L/T-RO 版本为目标，比再次仓促投稿更合理。
