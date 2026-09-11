# RA-L 论文修改计划

本文档用于维护 **Beyond Point-Attached Semantics** 从 CoRL 投稿版本修改为 IEEE Robotics and Automation Letters（RA-L）稿件的全过程。后续关于论文叙事、证据需求、篇幅分配、实验占位和修改进度的决定，均以本文档为准。

## 1. 修改目标

在不改变论文核心立意的前提下重构全文：

> 真实 RGB-D 观测中存在深度噪声、遮挡、标定误差、视角变化和采样变化。这些扰动会使直接附着在观测点上的语义特征产生较大变化，从而增加下游策略学习的难度。我们构建对象中心语义场，将基于观测的对象条件编码与显式三维位置上的语义读取解耦，从而生成更加稳定的功能部件特征，并提升操作策略的鲁棒性。

论文不应宣称连续场、tri-plane 或部件监督本身是全新的技术。论文的主要贡献应定位为：**面向策略学习，利用可查询语义场降低观测扰动引起的特征变化，并通过完整实验说明表示稳定性与操作性能之间的联系。**

### 建议的一句话贡献

> We introduce a policy-oriented object-centric semantic field that decouples noisy observation samples from semantic readout, yielding stable functional-part representations under real-world point-cloud perturbations and improving cross-instance manipulation.

中文含义：

> 我们提出一种面向策略学习的对象中心语义场，将含噪观测样本与语义读取位置解耦，从而在真实点云扰动下生成稳定的功能部件表示，并提升跨实例操作性能。

### 确定标题

> **Beyond Point-Attached Semantics: Stable Object-Centric Semantic Fields for Robust Manipulation**

该标题已经确定。`Stable` 对应表示层贡献，`Robust Manipulation` 对应策略层结果，形成“稳定表示带来鲁棒操作”的主线，同时避免在依赖类别级 PartNext 监督的情况下过度强调广义泛化能力。

## 2. 写作中的基本原则

- [ ] 将观测扰动导致的特征变化作为论文的首要问题。
- [ ] 将功能部件语义写成需要在不同观测和对象实例之间保持稳定的信息，而不是独立的第二条动机。
- [ ] 将 support/query 解耦写成方法的核心机制。
- [ ] 不把 tri-plane 网络结构本身写成主要创新。
- [ ] 区分直接实验结果和作者解释。只有被实验直接测量的结论使用 `shows`，推断性解释使用 `suggests`。
- [ ] 不宣称超出实际对象类别和扰动范围的泛化能力。
- [ ] 尽早、准确地说明类别级 PartNext 监督要求。
- [ ] 充分描述每个基线，证明比较公平。
- [ ] 除下游策略成功率外，必须加入表示层面的定量指标。
- [ ] 仿真实验使用多个随机种子并报告均值和标准差。
- [ ] 正文和参考文献合计不得超过 8 页。

## 3. 审稿意见与修改动作映射

| 审稿意见 | 对应修改动作 | 修改位置 | 状态 |
|---|---|---|---|
| 相比 neural descriptor fields 和已有 semantic fields，技术创新不明确 | 将创新重新定位为面向策略的 support/query 解耦与观测鲁棒性；加入直接对比说明 | Introduction、Related Work | [ ] |
| 性能提升可能主要来自 PartNext 监督，而不是语义场 | 增加使用相同 PartNext 监督的 point-wise 基线，以及不使用独立 query 的消融 | Experiments：组件分析 | [ ] |
| 三个训练损失的作用没有被隔离 | 增加 `-L_part`、`-L_align` 和 `-L_stab` 消融 | Experiments：组件分析 | [ ] |
| 性能提升可能只来自重新采样较干净的点 | 比较独立控制的 query 与在原始观测点上直接读取特征 | Experiments：support/query 消融 | [ ] |
| 2D lifting 基线可能较弱或描述不充分 | 完整说明 backbone、输出层、分辨率、投影、融合、遮挡处理、归一化、特征维度、点数和策略接口 | Experimental Setup | [ ] |
| D3Fields/F3RM 和相邻工作也可以生成稳定且具有部件感知能力的特征 | 正面承认其能力，从任务、策略接口、监督和评价方式上说明区别，而不是笼统否定 | Related Work | [ ] |
| 缺少表示层定量指标 | 增加 feature consistency、part mIoU 和/或 cross-instance correspondence accuracy | Representation Evaluation | [ ] |
| 真实世界中的提升明显大于仿真，但缺少解释 | 增加受控噪声、遮挡和视角实验，并分析特征不稳定性与策略性能下降的关系 | Robustness Analysis | [ ] |
| 是否要求对象点云被完整观测不明确 | 明确是否支持部分观测，并说明训练和部署阶段的可见性假设 | Method、Limitations | [ ] |
| Adapter `g_phi` 和维度定义不清楚 | 说明网络结构、输入输出维度、归一化方式和作用 | Method | [ ] |
| 方法图信息过密 | 将图 2 简化为三个阶段，并清楚区分训练和部署路径 | Figure 2 | [ ] |
| 每类别训练和部件标签限制了可扩展性 | 收紧论文主张；报告实际类别范围；讨论标注与部件 taxonomy 的限制 | Introduction、Experiments、Limitations | [ ] |
| 对噪声、部分视角和分割错误的敏感性未知 | 增加受控输入扰动实验 | Robustness Analysis | [ ] |
| 补充视频不清晰且缺少说明 | 稿件稳定后制作高清、有标注和配音的视频 | Supplementary Material | [ ] |

## 4. 目标证据链

修改后的论文必须依次建立以下因果链：

1. **问题：**真实观测扰动会改变点云分布和附着在观测点上的语义特征。
2. **机制：**support/query 解耦允许模型在受控位置读取语义，而不是只能在原始传感器采样点上读取。
3. **表示结果：**语义场查询特征在扰动下变化更小，并能跨对象实例对齐功能部件。
4. **策略结果：**更稳定的表示提升操作成功率，特别是在真实噪声和未见对象条件下。
5. **归因分析：**通过公平基线和消融证明性能提升不能仅由 PartNext 标签、冻结 backbone 或重新采样解释。

正文中的每个主要图表都应支持这条证据链中的一个环节。

## 5. 论文结构与 8 页篇幅预算

页数预算包含参考文献。

| 页码 | 内容 | 目标 |
|---|---|---|
| 第 1 页 | Abstract、Introduction、teaser | 清楚说明问题、机制和主要证据 |
| 第 2 页 | Related Work、方法概览 | 与 point-wise features 和已有 fields 准确区分 |
| 第 3 页 | Field construction、训练目标、policy interface | 只保留必要公式和实现事实 |
| 第 4 页 | 实验问题、实验协议、基线、表示稳定性评价 | 证明公平性并直接测量表示稳定性 |
| 第 5 页 | 仿真、未见对象结果和主要消融 | 连接表示稳定性和策略性能 |
| 第 6 页 | 真实世界、受控扰动和失败分析 | 解释真实世界中更大的性能提升 |
| 第 7 页 | Discussion/Limitations、Conclusion、参考文献开始 | 限定论文主张并总结发现 |
| 第 8 页 | 参考文献 | 正文 8 页内不再编入附录 |

该分配是目标而不是硬性规定。每完成一次主要重写，都应重新编译并检查页数和版面。

## 6. 分章节修改计划

### 6.1 标题、作者信息和摘要

- [x] 标题确定为 `Beyond Point-Attached Semantics: Stable Object-Centric Semantic Fields for Robust Manipulation`。
- [ ] 统一作者姓名、单位、通讯作者标记和邮箱。
- [ ] 删除 `markboth` 中残留的 hyperspectral 旧论文标题。
- [ ] 除非投稿模板明确要求，否则删除 DOI 占位文字。
- [x] 按照以下五步重写摘要：
  1. 真实 RGB-D 扰动会导致 point-attached semantic features 不稳定。
  2. 这种不稳定性使策略无法可靠利用功能部件信息。
  3. 提出 support/query 解耦的对象中心语义场。
  4. 说明表示评价和操作评价的范围。
  5. 只报告最有代表性且已经确认的数值结果。
- [ ] 除非 query 生成过程能够保证，否则不要将重新采样的点描述为 `clean points`。

### 6.2 Introduction

- [ ] 在开头第一段就提出观测噪声和采样不稳定性，而不是先单独讨论部件语义。
- [ ] 解释功能部件信息为什么必须在不同观测和不同对象实例之间保持一致。
- [ ] 使用一个具体例子，例如同一个 mug handle 在不同视角和采样下获得不同特征。
- [ ] 描述 point-attached features 的限制时，不要暗示所有 2D lifting 或已有 field 方法都必然不稳定。
- [ ] 将核心 insight 写为：`observation for conditioning, controlled queries for readout`。
- [ ] 尽早说明 PartNext 类别级监督及其适用范围。
- [ ] 将贡献列表改成有实验支持的三类贡献：
  - 表示机制；
  - 表示稳定性的直接证据；
  - 下游策略鲁棒性和跨实例结果。
- [ ] 不要将三个 loss 的简单罗列作为 contribution。

### 6.3 Related Work

建议压缩为三个主题：

1. **3D visuomotor policy learning：**DP3 及其他点云策略。
2. **Semantic 3D representations for manipulation：**feature lifting、GenDP、G3Flow、HeRO，以及 point-wise part/affordance features。
3. **Queryable object-centric fields：**NDF 类 descriptor fields、D3Fields/F3RM、feature-SDF/semantic fields 和其他连续表示。

必须完成的修改：

- [ ] 明确认可 D3Fields/F3RM 能产生一致且具有部件感知能力的 descriptor。
- [ ] 从输出用途和证据类型上说明区别：本文输出冻结的、直接面向策略的语义表示，并在受控观测扰动下评价其稳定性。
- [ ] 加入 Twin-DP3，说明部分观测和视角变化已经被证明是 3D 策略的重要瓶颈。
- [ ] 加入 HeRO，作为紧密相关的 semantic representation baseline；若无法复现，应说明任务设定或接口差异。
- [ ] 未逐篇核实前，不使用“已有 field 方法只用于 planning”这类宽泛表述。
- [ ] 只有在对比表确实能替代较长文字且不超页时，才加入方法对比表。

### 6.4 Method

方法部分保留四个逻辑模块：

1. 问题定义和 support/query 解耦。
2. 对象条件语义场构建。
3. 语义场训练目标。
4. 生成 semantic point cloud 并接入策略。

必须澄清的技术事实：

- [ ] 定义 support 和 query 使用的坐标系。
- [ ] 说明是否要求完整对象点云。
- [ ] 准确说明训练和部署阶段如何生成 query points。
- [ ] 解释部署阶段的 query 分布为什么比原始采样更稳定。
- [ ] 如果 query 仍然直接从 noisy observed cloud 中采样，需要相应收紧“解耦”的论述。
- [ ] 定义 `g_phi` 的层结构、输入输出维度、激活函数、归一化和作用。
- [ ] 说明 `d_u` 和 `d_s` 是否相同。
- [ ] 说明 support features 如何分别聚合到三个 tri-plane。
- [ ] 明确三个 plane 是并行生成的。
- [ ] 分别说明 field training、policy training 和 deployment 中哪些模块冻结、哪些模块训练。
- [ ] 说明策略最终接收 raw scene points、semantic object points，还是两者同时接收。

篇幅压缩：

- [ ] 保留总损失函数。
- [ ] 除非复现必须，否则压缩 supervised contrastive loss 的完整展开公式。
- [ ] 删除每个公式后重复表达同一含义的说明文字。
- [ ] 将低层超参数移到简短 implementation paragraph 或 supplementary material。

### 6.5 方法图

将 Figure 2 重构为三个明确阶段：

1. **Noisy observation as support：**部分、含噪对象点云与冻结 backbone。
2. **Object-conditioned semantic field：**三个并行 plane 和 query decoder。
3. **Policy readout：**受控 query points、semantic point cloud 和 DP3。

图中还应展示：

- [ ] field training 与 policy training/deployment 的清晰分界。
- [ ] part labels 和 logits 只出现在 field training 分支。
- [ ] 输入策略的是 embeddings，而不是 part logits。
- [ ] `XY`、`XZ` 和 `YZ` plane 使用并行箭头。
- [ ] 原始观测点和查询语义点之间的直观对比。

### 6.6 Experiments

实验章节开头明确提出四个研究问题：

- **RQ1：**point-attached features 和 field-based features 对观测扰动分别有多敏感？
- **RQ2：**哪些方法组件带来了表示稳定性？
- **RQ3：**更加稳定的表示能否提升标准设置和未见对象设置下的操作性能？
- **RQ4：**表示稳定性是否能够解释本文方法在真实世界中更明显的提升？

#### 实验设置

- [ ] 说明机器人、相机、标定、分割、点云构建和坐标系。
- [ ] 说明每个类别的训练/测试对象数量和准确划分规则。
- [ ] 说明每任务 demonstrations 数量、策略训练 epoch/step、batch size 和 checkpoint 选择方式。
- [ ] 说明随机种子数量和 evaluation episodes 数量。
- [ ] 仿真实验报告 `mean ± standard deviation`。
- [ ] 使用紧凑表格或图注定义各任务成功标准。
- [ ] 说明所有基线在适用情况下共享 demonstrations、masks、calibration、point counts、policy backbone 和训练协议。

#### 基线实现说明

对于 **2D Feature Lifting**，正文必须说明：

- [ ] DINOv2 具体型号和输出层。
- [ ] 图像分辨率和预处理。
- [ ] 相机投影与多视角融合方式。
- [ ] 可见性、遮挡和重复投影处理。
- [ ] 特征归一化和降维方式。
- [ ] 点数以及接入策略的方式。
- [ ] 哪些模块冻结、哪些模块训练。

对于 **3D Point-wise Features**，正文必须说明：

- [ ] Utonia checkpoint 和输出层。
- [ ] 冻结和训练的模块。
- [ ] 特征维度和归一化。
- [ ] 与策略的准确连接方式。

需要增加或讨论：

- [ ] 使用相同 PartNext 监督的 point-wise Utonia。
- [ ] G3Flow、GenDP、HeRO 或另一个强连续/语义场基线。
- [ ] 直接将 part probabilities 输入策略的基线。

#### RQ1：表示层直接评价

固定对象身份和姿态，每次只改变一种因素：

- camera viewpoint；
- 点云下采样随机种子或点密度；
- synthetic depth noise；
- occlusion ratio；
- segmentation mask 的腐蚀、膨胀或 outliers。

候选指标：

- [ ] 匹配或 canonical queries 上的 feature consistency：cosine similarity 或 L2 distance。
- [ ] 使用训练阶段 part logits 计算 part mIoU。
- [ ] Cross-instance correspondence accuracy 或 nearest-neighbor part accuracy。
- [ ] Within-part 与 between-part embedding separation。

建议主图：横轴为扰动强度，纵轴为表示一致性，对比 2D lifting、3D point-wise 和 ours。

#### RQ2：归因分析和消融

最低要求的消融表：

| Variant | Independent query | Part loss | Alignment loss | Stability loss | Embedding input | Representation metric | Policy success |
|---|---:|---:|---:|---:|---:|---:|---:|
| Full model | Yes | Yes | Yes | Yes | Yes | `TBD` | `TBD` |
| Read at observed points | No | Yes | Yes | Yes | Yes | `TBD` | `TBD` |
| Point-wise + PartNext | No | Yes | Optional | Optional | Yes | `TBD` | `TBD` |
| Part probabilities only | Yes | Yes | N/A | N/A | No | `TBD` | `TBD` |
| Without alignment | Yes | Yes | No | Yes | Yes | `TBD` | `TBD` |
| Without stability | Yes | Yes | Yes | No | Yes | `TBD` | `TBD` |
| Without part anchoring | Yes | No | Yes | Yes | Yes | `TBD` | `TBD` |

如果篇幅不足，正文至少保留前四项，以及最有解释力的两个 loss ablation。

#### RQ3：策略评价

使用三个实验协议：

1. **Standard simulation：**训练和测试对象遵循 benchmark 标准设置。
2. **Unseen-object simulation：**按类别留出部分对象实例，仅用于测试。
3. **Real-world unseen objects：**训练对象和测试对象完全不重合。

主结果表要求：

- [ ] 增加 Average 列。
- [ ] 仿真结果使用 `mean ± std`。
- [ ] 在 caption 或正文中说明每任务 episode 数量。
- [ ] 分开报告 standard 和 unseen-object results，不混合解释。
- [ ] 从任务所需功能部件的角度分析任务间差异。
- [ ] 不单独依赖成功率推断机制，机制解释必须引用 RQ1/RQ2 的证据。

#### RQ4：真实世界鲁棒性分析

- [ ] 定量测量真实 RGB-D 观测上的 feature inconsistency。
- [ ] 将失败分类为 perception/part localization、grasp、contact/alignment 和 motion/policy failures。
- [ ] 使用实际测量到的扰动解释为什么真实任务中的提升更大。
- [ ] 如果可能，分析不同条件下 representation inconsistency 与 policy success degradation 的相关性。
- [ ] 将 segmentation/tracking failures 与 policy failures 分开报告。
- [ ] 用一个紧凑段落或表格行报告 runtime、latency 和 hardware。

### 6.7 Discussion、Limitations 和 Conclusion

- [ ] 如能节省篇幅，将独立 Limitations 合并为实验后的 `Discussion and Limitations`。
- [ ] 讨论类别级训练和 PartNext 标注成本。
- [ ] 讨论 ambiguous、continuous、deformable 和 task-dependent functional regions。
- [ ] 讨论严重遮挡、分割失败和不完整观测。
- [ ] 不暗示方法具有 open-vocabulary 或 category-free generalization。
- [ ] Conclusion 压缩为一个段落，重述证据链，不重复方法细节。

## 7. 正文图表规划

建议正文最终保留：

1. **Figure 1：**问题和核心 insight；point-attached instability 与 stable queryable field 的对比。
2. **Figure 2：**简化后的三阶段方法图。
3. **Figure 3：**任务和真实平台总览，包括训练/测试对象划分。
4. **Figure 4：**受控扰动下的表示一致性曲线，并附一个小型 qualitative panel。
5. **Table I：**standard 和 unseen-object simulation results。
6. **Table II：**real-world success rates；空间允许时加入 failure breakdown。
7. **Table III：**组件和 loss ablations。

应从正文移除或转入 supplementary material 的内容：

- [ ] 独立的对象分割流程图。
- [ ] 占用整栏或双栏的额外特征可视化。
- [ ] 能够合并到 Figure 3 的重复任务图。
- [ ] 冗长的逐任务文字定义。

## 8. 附录与补充材料处理

当前附录不应继续编入 8 页 RA-L 正文。

| 当前附录内容 | 处理方式 |
|---|---|
| 八个任务的长篇文字定义 | 改为紧凑的任务/成功标准表格或图注 |
| 仿真任务图 | 合并到正文任务总览图 |
| 真实世界平台图 | 合并到正文任务总览图 |
| 对象分割流程图 | 在实验设置中用一段文字描述；图片仅放 supplementary material |
| 额外语义特征可视化 | 放入 supplementary material |
| 详细超参数 | 放入 supplementary material；复现必需的信息可保留为紧凑表格 |

## 9. 重写前必须确认的技术事实

以下问题属于事实确认，而不是写作偏好，不能依靠推测补全。

- [ ] 部署阶段的 query points 是从 observed object points、重建表面、canonical template、bounding volume，还是其他分布中采样？
- [ ] support 和 query points 在哪个坐标系中归一化？
- [ ] 训练或部署阶段是否要求完整对象点云？
- [ ] 当一个功能部件完全不可见时，模型如何处理部分观测？
- [ ] `g_phi` 的准确网络结构是什么？
- [ ] `d_u`、`d_s`、`d_t`、`d_g` 和 policy input dimension 分别是多少？
- [ ] 真实世界实验中如何融合多相机点云？
- [ ] SAM2 是在每个视角独立运行，还是包含跨视角关联？
- [ ] query coordinates 如何变换回 robot/world frame？
- [ ] 每个对象类别使用了多少 PartNext instances 和 part categories？
- [ ] 每个类别单独训练一个 semantic field，还是多个类别共享同一个 field？
- [ ] 每项结果具有多少 policy seeds、demonstrations 和 evaluation episodes？
- [ ] 在现有代码和算力范围内，可以公平复现哪个强 field baseline？

## 10. 执行顺序

### Phase A：建立论文骨架

- [ ] 确认一句话贡献；标题已经确定。
- [ ] 确认第 9 节中的技术事实。
- [ ] 重写 Introduction。
- [ ] 重写 Related Work，并加入两篇近期 RA-L 论文。
- [ ] 压缩和澄清 Method。
- [ ] 使用 RQ1-RQ4 和 `TBD` 数值占位重建 Experiments。
- [ ] 从主文件中移除附录，仅将必要实验设置融入正文。
- [ ] 加入实验结果前先编译，确认骨架能够控制在 8 页内。

### Phase B：填入实验证据

- [ ] 加入表示层评价。
- [ ] 加入受控输入扰动结果。
- [ ] 加入 PartNext-controlled point-wise baseline。
- [ ] 加入 support/query 和 loss ablations。
- [ ] 将单次仿真百分比替换为多随机种子统计结果。
- [ ] 加入真实世界失败分析和 runtime。

### Phase C：最终写作检查

- [ ] 确保每个主要 claim 都对应表格、图片或已经引用的前人结论。
- [ ] 确保所有 caption 可以独立理解，并说明实验协议和样本量。
- [ ] 删除缺少证据的因果性表述。
- [ ] 检查 `support`、`query`、`semantic field`、`semantic point cloud` 和 `functional part` 等术语是否一致。
- [ ] 核查全部 citation 和 BibTeX 条目。
- [ ] 对完整 8 页稿件进行视觉版面检查。
- [ ] 制作高清、带标注和配音的 supplementary video。

## 11. 决策记录

所有影响论文叙事或结构的决定都记录在这里，以保证后续修改一致。

| 日期 | 决定 | 原因 |
|---|---|---|
| 2026-08-19 | 保留“语义场提高真实观测鲁棒性”的核心立意 | 这是论文最强、且得到审稿人认可的出发点 |
| 2026-08-19 | 将实验作为本轮修改的最高优先级 | 审稿人一致认为当前方法描述过多，而实验证据不足 |
| 2026-08-19 | 论文总长度按包含参考文献在内 8 页规划 | 作者给出的 RA-L 篇幅约束 |
| 2026-08-19 | 不再将当前附录编入正文，仅合并必要设置 | 为表示指标、消融和公平基线描述留出空间 |
| 2026-08-19 | 使用中文维护修改计划，保留必要英文技术术语 | 便于作者团队讨论，同时避免论文术语翻译歧义 |
| 2026-08-19 | 标题确定为 `Beyond Point-Attached Semantics: Stable Object-Centric Semantic Fields for Robust Manipulation` | `Stable` 对应表示层贡献，`Robust Manipulation` 对应策略结果，并避免过度声称广义泛化 |
