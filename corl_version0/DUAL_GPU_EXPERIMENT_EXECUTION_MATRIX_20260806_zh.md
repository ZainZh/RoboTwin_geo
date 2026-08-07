# 双机补实验执行矩阵

日期：2026-08-06
资源：本机 RTX 4090 24GB；远程 RTX 6000 Ada（待阶段41只读审计确认具体显存/环境）
目标：以最短墙钟时间补齐审稿人要求的直接表征证据、因子化消融、公平baseline、鲁棒性和闭环结果。

## 1. 状态定义

- `完成`：协议、数据、统计和产物足以进入论文。
- `诊断完成`：已有可靠工程结果，但split或比较边界不足，不能直接放主表。
- `进行中`：代码或训练已启动。
- `阻塞`：缺数据、机器人或第三方资源，不能靠增加GPU解决。

## 2. 总实验矩阵

| ID | 审稿问题 | 正式实验 | 当前状态 | 主要缺口 | 首选机器 |
|---|---|---|---|---|---|
| R1 | 缺直接表征指标 | held-out instance part mIoU、per-part IoU、confusion、balanced accuracy | 诊断完成 | 旧checkpoint只见过val；Mug val无Closure；需独立test split | 4090评测，6000 Ada重训 |
| R2 | 缺表征稳定性 | 固定query，只改support count/resample/dropout/noise/crop/viewpoint；cosine/L2/label agreement/mIoU drop | 诊断完成 | 显式single-view相机模型、segmentation contamination/outlier尚缺 | 4090 |
| A1 | 三项训练loss无消融 | Full-fixed、No-CE、No-SupCon、No-Consistency | 待启动 | 统一训练入口、3 seeds、R1/R2复评 | 6000 Ada |
| A2 | 收益来自field还是PartNext/重采样 | observed-point/no-resampling、resampled XYZ、part-prob、128D field | 部分完成 | XYZ/part-prob/field e3000已完成；attached/no-resampling未完成 | 6000 Ada训练，4090评测 |
| A3 | part语义是否为因果变量 | correct part-prob、channel shuffle、point shuffle、uniform、random | 待启动 | 生成严格matched zarr并训练最小代表seed/多seed | 4090或6000 Ada |
| B1 | 3D point-wise baseline是否公平 | same-query raw Utonia、point-wise part supervised、XYZ、field | 工程smoke | Beat raw object support缺失；context-nearest禁止进论文 | 数据优先，恢复后6000 Ada |
| B2 | 2D lifting是否过弱 | visibility-aware DINOv2 + depth consistency + multiview fusion | 审计中 | 正式RGB/depth/calibration/query接口与模型缓存待核对 | 4090预处理，6000 Ada策略 |
| B3 | 相近连续3D方法缺失 | D3Fields/F3RM风格persistent fusion，或可复现PA3FF/PADP | 待可行性判断 | 代码/权重/数据接口与时间成本 | 6000 Ada |
| G1 | 实物提升大于仿真原因不清 | clean→realistic sensing controlled ladder；同policy逐级加noise/occlusion/seg error | 待实现 | 必须保持task/policy不变并记录阶段失败 | 4090仿真评测 |
| G2 | 噪声/不完整点云鲁棒性 | single view、部件遮挡、5/10/20% outlier与mask漏检/粘连 | 部分完成 | R2已有noise/crop，但没有显式mask/outlier协议 | 4090 |
| P1 | policy证据缺多seed统计 | 每方法/任务≥3 training seeds，paired initial conditions，success+CI | Beat offline完成 | offline loss不是success；其他任务与rollout未完成 | 6000 Ada训练，4090 rollout |
| P2 | 实物主表不可配对重算 | block-randomized paired trials、Wilson CI、失败阶段、延迟 | 阻塞 | 需要机器人、初始条件协议和逐trial logger | 实物系统 |

## 3. 已完成且需要保留的证据

### R1旧validation诊断

- Hammer：mIoU 0.8580；Handle 0.8539，Head 0.8622。
- Mug：mIoU 0.9636，但Closure无GT点，只覆盖Container/Handle。
- Spoon：mIoU 0.9641。
- 产物含逐实例/重复/部件CSV、confusion JSON和checkpoint/Utonia provenance。

这些结果可以用于设计正式split和展示问题，但正文主表应标为validation diagnosis，直到独立test协议完成。

### R2固定query诊断

- 三类别完整运行共3640个condition forwards。
- 5000/1024 support通常较稳；512以下出现类别相关退化。
- Gaussian noise相对稳定；single-sided partial crop是最主要弱点。
- Hammer校准改善稀疏support和中等crop，但support-normalized极端crop仍未解决。

### A2策略输出消融

- Beat Cube、80条实机demonstrations、相同query/action/state/DP3、3 seeds、3000 epochs。
- Raw offline loss：part-prob `0.009016±0.000194`，XYZ `0.009914±0.000495`，field `0.010136±0.001990`。
- part-prob对XYZ为3/3更低，平均约低9.05%；field方差最高且没有建立稳定优势。
- 结论边界：只支持part-aware interface的offline信号，不是task success。

## 4. 首批并行任务

### 远程 RTX 6000 Ada

1. 只读审计代码、Conda、PartNext、Utonia、semantic checkpoints、Beat/Stir数据和磁盘。
2. 运行Hammer loss ablation smoke，验证四组真正改变loss而非仅改变run名。
3. profile物理batch；正式四组使用同一effective batch、样本暴露数、optimizer steps与seed。
4. 启动Full-fixed/No-CE/No-SupCon/No-Consistency多seed训练。
5. 若远端存在Beat raw support，立即解锁formal Utonia生成与策略矩阵。

### 本机 RTX 4090

1. 扩展R2：outlier、mask contamination、功能部件定向遮挡、显式single-view/view direction。
2. 为每个loss ablation checkpoint自动运行R1/R2并生成统一JSON/CSV。
3. 实现part-prob corruption数据与最小策略pilot。
4. 完成visibility-aware DINO几何/深度一致性工具、单测和小样本smoke。
5. 维护汇总文档、manifest和统计脚本。

## 5. Batch与公平性规则

- 不直接把远端batch增大后的结果和本机旧模型混成同一消融表。
- 先测试batch 6/12/24或显存允许的档位，再为四个loss组固定同一effective batch。
- 若使用gradient accumulation，记录physical batch、accumulation steps和effective batch。
- 固定每个方法看到的样本数与optimizer updates；不能只固定epochs而让大batch少更新。
- Full-fixed必须在相同新协议下重训，作为所有loss消融的共同control。
- 所有正式run保存seed、split IDs、resolved config、代码diff、data/checkpoint SHA、wall time、peak VRAM和安全权重。

## 6. 验收标准

### R1正式表

- train/val/test instance IDs不交叠；test不参与checkpoint选择。
- 每个split检查part coverage；缺失part明确N/A，不能静默从macro平均中消失。
- 报告pooled confusion、object-level mean±SD/CI、per-part IoU。

### R2正式表

- query坐标逐bit固定，只改变声明的support/observation因素。
- 同时报告feature cosine/L2和label/mIoU；不能只靠平均cosine掩盖分类边界退化。
- 至少跨实例、跨扰动方向、跨随机trial汇总，并给分位数/CI。

### A1/A2消融

- 每组代码路径、数据、split、seed、预算、point count与encoder容量matched。
- loss组必须从训练日志验证对应项为0或非0。
- offline指标只作诊断；论文策略主张最终由paired closed-loop success决定。

### Baseline

- DINO必须有depth/visibility check、明确invalid mask和多视角融合权重。
- Utonia必须使用raw独立object support、相同query/action/DP3；context-nearest工程数据永不进入正式表。
- 第三方强baseline若无法运行，记录具体接口/许可/数据阻塞，并收窄相关主张。

## 7. 决策门

1. 若part-prob corruption显著退化，支持收益来自part semantics；否则重新检查容量/数值通道解释。
2. 若No-Consistency在R2不差，不能把稳定性归因于当前consistency loss。
3. 若No-CE或point-wise part-prob与full field相当，论文主线转为PartNext-supervised part interface，不强调continuous embedding。
4. 若visibility-aware DINO接近或超过ours，改写动机为接口/计算/3D-native tradeoff，禁止声称2D lifting本质不稳定。
5. 若closed-loop排序与offline loss不同，以闭环为主，并分析接触阶段和轨迹分布原因。

## 8. 需要用户快速协调的条件

- 远端没有PartNext/Utonia/checkpoint且数据同步量过大或路径未知。
- Beat raw object support在本机和远端都不存在。
- 需要实物机器人时间、测试对象清单和可复现initialization protocol。
- 强baseline需要未公开权重/代码或额外许可证。
- 双机存储预计不足以保留安全final weights与必要中间表征。
