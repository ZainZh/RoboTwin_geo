# 调查发现

用于持续记录鞋子 SE(3) 放置对比实验的代码事实、推断和待确认项。

## 初步事实
- 比较训练入口接受 `task_name/task_config/expert_data_num/seed/gpu_id/route` 等参数；`route=baseline` 不加入额外低维观测，其余 route 加入形状为 11 的 `se3_relation_token_A_to_B`。
- 所有路线仍使用 `robot_dp3_objpc.yaml` 和物体点云；实验核心不是再次直接拼接高维 NDF，而是在 baseline 点云策略上额外条件化一个紧凑的物体间 SE(3) 关系 token。
- 输出数据以 `-objpc-placement-only-baseline` 或 `-objpc-placement-only-se3-relation-<route>` 隔离，非 1024 点还带 `-pcN`，训练输出也按 route/seed 隔离。
- 脚本会在目标 zarr 不存在时自动调用 `process_data_shoe_se3_placement_comparison.sh`，之后启动 DP3 训练；默认 resume、EMA、batch 256、encoder 输出 128。
- 工作区当前已有 `.codex` 删除状态，属于用户既有改动，本任务不会触碰。
- PCA 对齐脚本总体是：加载两个类别的 NDF 与 PCA 模型，从示范物体/查询物体的描述子建立对应，再通过可微 SE(3) 优化迁移双物体相对姿态；需要继续读其损失与输出部分。

## 比较协议
- 四组 matched routes：`baseline`、`oracle`、`ndf_no_direction`、`ndf_direction`。
- 数据仅保留 `relation_phase > 0` 的放置阶段帧；状态为当前机器人 vector，action 为下一帧 vector。评估则用专家先完成抓取与抬升，再从 placement 阶段启用 policy，避免抓取能力掩盖放置精度。
- A 明确是鞋，B 是旋转斜块/ramp。原始数据必须包含逐帧 `object_pose_A/B`、oracle 功能相对位姿、`shoe_id`、`relation_phase`。
- 11 维 token = 3D 平移纠偏 + 6D 旋转纠偏 + solver energy + 有效门控。纠偏在 B 坐标系表达；token 会乘以 `phase * confidence`，无解/非有限能量/无效门控时全零。
- `oracle` 直接读取该 episode/task state 的真值目标相对位姿，energy=0/confidence=1；两条 NDF 路线按 `shoe_id` 从离线 goal table 查目标相对位姿。
- NDF goal table 由鞋-斜块 SE(3) 验证器生成：同一示范鞋向 10 个鞋实例迁移，分别比较 direction weight 0 与 5，多次随机 trial 后按每鞋最低 energy 选解。
- `baseline` 与三种几何路线使用相同 placement-only 样本、相同点云与训练配置，主要变量仅为关系 token 的来源，因此可分解判断：(1) 显式真值关系是否能帮助 DP3；(2) NDF 回归关系是否足够准；(3) 方向约束是否有增益。
- 这些比较文件已被提交，相关历史至少在 `9ab8921`、`f8fd3aa`，不是当前未提交草稿；文件时间为 2026-07-20 晚间。

## 与用户指定 PCA 方法的关系（重要）
- 当前比较链路**没有直接调用** `ndf_demo_dual_object_align_pca.py`，`validate_ndf_shoe_ramp_se3.py` 也没有加载 PCA 模型。
- 它借用了相同的核心思想——固定示范关系、在查询实例的 NDF 描述子场上对 SE(3) 做多起点梯度优化——但为鞋-斜块任务改成了单向版本：只加载鞋 NDF，把斜块表面 11x5 probe grid 映射进鞋坐标系，匹配 demo 鞋的 reference features。
- `ndf_no_direction` 只最小化 NDF feature L1 energy，可能因鞋的近似对称性产生鞋头/鞋跟 180° 翻转；`ndf_direction` 再加入“斜坡上坡轴→鞋头方向”和“斜坡法向→鞋底法向”的显式方向损失，默认权重 5。
- 原始 PCA 双物体脚本更一般：A/B 各自加载 NDF+PCA，利用 PCA 聚类/锚点及双向 probe 描述子能量优化相对 SE(3)。因此现在的 comparison 是其任务特化的简化实现，而不是严格复用用户点名的 PCA 回归管线。

## DP3 注入方式
- 训练时 Hydra 动态把 token 注册为 `low_dim`；DP3Encoder 会把 `agent_pos(14)` 与 token(11) 拼成 25 维低维输入，经 state MLP 编码后，再与 PointNet 的物体点云特征拼接。它不是把 token 拼到每个点，也不是原先的 256 维 NDF latent 拼接。
- 部署时每帧从真实 task state 计算当前相对误差 token；NDF 本身不在线运行，NDF 优化结果已离线按 `shoe_id` 固化到 goal table。
- 环境成功判定要求功能点 XY 误差分别小于 5cm/3cm、四元数对齐绝对内积 >0.98，且双夹爪打开。

## 历史
- 主要功能在提交 `9ab8921` 一次性加入；`f8fd3aa` 随后统一了数据路径。该提交还删除过旧的根目录 planning 文件，所以聊天虽丢失，Git 中仍保留了实现边界。

## 当前实物状态与风险
- 原始 collection 已生成 50 个 `_traj_data/episode*.pkl`，但只有 `episode0..8` 共 9 个 HDF5；采集/回放转 HDF5 在第 9 个之后中断。目前 comparison 预处理若用 50 episodes 会在 episode9 缺文件。
- 还没有发现任何 `placement-only-*` comparison zarr，说明四组比较训练尚未真正开始。
- NDF checkpoint `/home/zheng/model/ndf/shoe.pth` 存在，离线验证与 `outputs/ndf_shoe_ramp_se3/comparison/goal_table.json` 已存在，且两条 NDF route 都覆盖 10 个 shoe IDs。
- 离线验证总体：无方向 30 trials success 46.7%，方向约束 90%；方向约束把中位旋转误差从约 145.8° 降到 3.35°，平移中位误差均约 6mm。
- 但 goal table 对每鞋仅按最低 total energy 选解，不检查 `success`。最终选中的 `ndf_no_direction` 有 4/10 失败（shoe 0/2/5/7）；`ndf_direction` 有 1/10 失败（shoe 5，平移误差约 12.5cm、energy 10.3），仍以 confidence=1 写入。这会把明显错误目标喂给 DP3，是实验解释必须记录的已知失败，而不能只报整体训练成功率。
- episode0 的 task_state schema 正确，共 177 帧，其中 placement active 50 帧（28.2%）；comparison 会只用这 50 帧。
- 当前 `geo-utonia` Python 环境缺 `zarr` 和 `diffusers`，导致 2 个集成测试在 import 阶段失败；其余 16 个所选 relation tests 通过。另有环境静态集成 11 tests 通过、相关 Python 编译与四个 shell 的 `bash -n` 通过。训练脚本本身不激活环境，因此正式运行前要切到具备 DP3 依赖的环境。
- 当前无 collection/train 进程运行。

## 接手后的建议顺序
1. 用相同 collect 命令续跑，让转换逻辑从 episode9 自动继续，补齐 50 个 HDF5；先校验每个 episode 的 task_state 与 placement 帧数。
2. 保留现有四组作为 v1：placement baseline / oracle / NDF no-direction / NDF direction，先形成可解释的控制实验。
3. 在训练前修正或明确记录两个公平性问题：baseline 最好也使用同结构的全零 11D token；NDF energy/confidence 目前可能充当 shoe-ID 泄漏，且失败解 confidence 仍为 1。
4. 把用户指定的 PCA 双物体回归正式增加成独立 route（而非把当前单向 NDF route 改名），这样可以直接比较“简化单向 NDF”和“原始 PCA 双向几何关系”。
5. 统一 seed/训练超参后训练多 seed，并以 success rate 之外的 XY、旋转、释放后稳定性误差做统计。

## 实验解释边界
- 当前 token 的“目标关系”来自 NDF 或 oracle，但“当前关系”由模拟器真值 object poses 计算；所以 v1 验证的是在完美状态估计下，显式几何目标误差能否改善策略放置，不等价于完整视觉系统端到端地从点云估姿。
- baseline 没有 11D 分支，而其他组有，参数结构并非完全等同；增加 zero-token baseline 可消除这一混淆。
- goal table 按同一批固定 10 个 shoe IDs 查表，主要验证跨实例目标迁移+控制收益，而不是未见鞋实例上的在线泛化。
