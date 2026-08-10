# RoboTwin策略比较实验状态（2026-08-08）

## 目的

本轮实验把审稿人关心的“语义表征是否真正改善闭环策略”从实机少量结果扩展为可复现的RoboTwin仿真比较。当前先在Beat Block Hammer完成严格matched的三training-seed矩阵，再在统一100个测试seed上比较任务成功率与失败阶段；Hanging Mug随后复用同一协议。

## 冻结协议

- 任务：Beat Block Hammer。
- 动作：RoboTwin原生14D joint action；不把实机20D EEF配置错误接入仿真。
- 数据：50 demonstrations、5718 policy frames、scene point cloud 1024×6、semantic query 128点。
- 数据划分：dataset split seed 424242、validation ratio 0.1。
- 训练：seeds 20260805/20260806/20260807、300 epochs、batch 256、BF16、TF32、EMA。
- 公平性：六条路线共享逐元素相同的action、state、scene cloud和query XYZ；只改变query点附加表征。
- 评测：所有模型使用同一组100个RoboTwin test seeds，报告success rate、置信区间、耗时和失败阶段。

## 六条正式路线

| 路线 | query维度 | 作用 |
|---|---:|---|
| XYZ | 3 | 不使用语义的几何下界 |
| Uniform probability | 5 | 保留输入维度但移除语义信息 |
| Shuffled probability | 5 | 保留概率分布统计但破坏点-语义对应 |
| Part probability | 5 | 仅使用部件概率 |
| Semantic field | 131 | 论文方法：XYZ + part probability + 128D feature |
| Utonia | 579 | 使用完整raw support的3D表征比较 |

visibility-aware DINOv2暂不进入正式策略表。全量审计确认现有50条演示的100个camera-episode均缺dense depth；在确定性重渲染或补采同步depth之前，不能用1024点稀疏cloud冒充遮挡判断。

## 当前运行状态（2026-08-08 01:19 HKT）

- 本机RTX4090负责seed 20260805六路线。XYZ、part-prob、uniform与shuffled已完成e300并通过安全加载、metadata和SHA256归档；field已到225/300，随后自动接Utonia。与此同时Hanging Mug field构建与训练并行，占用约1.1GB显存；4090短时采样SM均值约87%。
- 远端RTX6000 Ada负责seeds 20260806/20260807全部12个run。seed20260806的XYZ、uniform与field已完成并有v2 manifest；part-prob和Utonia正在双lane运行，当前3/12完成，预计04:20--04:30 HKT完成训练队列。
- 两卡按实测总吞吐均冻结为双lane；4090约99% GPU/23.8GB，RTX6000 Ada平均约92%、峰值100%。第三lane在实测中引入CPU/显存竞争，不提高总推进率。
- 远端结果每120秒增量回传。每个run必须先生成不可覆盖的v2 completion manifest，本机再重新安全加载并交叉核对EMA、完整shape meta、checkpoint/Hydra/code SHA后生成v2 receipt；旧v1 marker不作为释放依据。
- matched-Utonia正式在线链已增加fail-closed门禁：必须使用safe checkpoint，原始对象support不可缺失，输出必须严格为`[128,579]`、finite且query XYZ逐值不变。相关CPU回归31/31通过，真实fixed-2仿真服务已排队。

正式输出根：`outputs/paper_revision/robotwin_sim_policy_beat_hammer_e300_bf16_v1`。

## 结果进入论文前的硬门禁

1. 18/18 checkpoint均通过epoch、seed、task、zarr、precision与TF32 metadata校验。
2. 12/12远端run均有本机receipt，且两条remote lane completion存在。
3. 每个模型先做少量固定seed闭环smoke；通过后运行完整100-seed评测。
4. 汇总三training-seed的success rate均值、方差/置信区间，并保留逐rollout JSONL和失败视频索引。
5. 不根据结果好坏挑seed、删失败或修改数据；任何运行异常必须单独标记而不是计为成功。

RTX6000实例只能在12/12本地v2 receipt、两条lane成功完成、最终checksum dry-run无差异、无rsync partial、训练代码快照与六份数据指纹均归档后释放；运行中的monitor计数本身不构成释放条件。

## 后续顺序

- 完成Beat Hammer 18-run训练与fixed-100闭环表。
- 对输入点云加入Gaussian noise、dropout/crop、outlier和不完整视角，测量success degradation。
- 在Hanging Mug复现实验主路线，验证结论是否跨任务成立。
- 结合逐阶段失败统计解释“实物提升大于仿真”的来源，并更新论文主表、消融表和失败案例。

Hanging Mug只读审计已确认50条数据完整，按T-1预计16942帧；field/XYZ/part-prob/Utonia构建器均可复用，六份matched zarr构建后会先做逐值一致性验证。跨任务18-run矩阵已排队：本机seed20260805六路线，远端seeds20260806/07十二路线；只有Beat训练/smoke及Hanging数据门禁全部通过才自动启动。其全部50条演示同样缺dense depth，因此DINO仍须重渲染/补采。
