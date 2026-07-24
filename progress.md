# 工作进度

## 2026-07-21
- 开始接手 `train_shoe_se3_placement_comparison.sh` 对比实验。
- 已读取 `planning-with-files` 技能并建立调查计划。
- 已初读比较训练入口、PCA 对齐程序以及 DP3 中相关文件索引。
- 已还原四条实验路线、11 维 token schema、placement-only 数据配对方式与 goal-table 生成流程。
- 已确认当前鞋任务验证器是 PCA 双物体对齐思想的单向任务特化版，并未直接使用 PCA；已追踪 token 在 DP3 Encoder 与在线部署中的注入路径。
- 已检查数据与产物：50 个轨迹仅转换出 9 个 HDF5，goal table 已生成，comparison zarr/训练尚未开始。
- 已检查 NDF 验证结果与每鞋最优解，识别 shoe5 的方向路线错误解仍以 confidence=1 入表。
- 验证：4 个 shell 语法通过；16 个 relation 单测通过、2 个因当前环境缺依赖无法导入；任务静态集成 11 tests、相关 py_compile 均通过。
- 调查阶段完成；已形成继续补数据、完善公平对照、增加严格 PCA route、训练评估的接手顺序。
- 服务器复现发现 `include/geometry_awareness_manipulation` 是指向本机外部仓库的绝对 symlink，服务器因此找不到 `ndf_robot`；开始依赖自包含整改。
- 完成依赖自包含：baseline 解耦 NDF import、最小 NDF runtime vendoring、requirements/preflight 检查与 wrapper 接入。
- 验证完成：RoboTwin 环境 23 tests、1-episode baseline zarr、真实 checkpoint CPU forward、NDF validator smoke、shell/Python syntax 全部通过。
- 开始修复推理成功率恒为 0：审计 rollout 终止条件与必须松爪的旧成功判定。
- 已将鞋任务成功标准改为持物也可满足的完整功能位姿对齐，并用纯 NumPy 回归测试验证；16 tests、py_compile、diff check 通过。
- 开始审计已有 100 次评估产物能否支持逐 seed/鞋型/姿态误差与配对统计。
- 审计完成：现有 `_result.txt` 只含聚合成功率，视频不含结构化种子/姿态元数据；完整失败分析需加 logger 后重跑评估，但无需重训。
- 2026-07-23：定义去除 `shoe_id` 的无身份泄漏协议；明确 ID 只用于 evaluator 分组，query metadata/资产路径禁止进入 estimator，并要求 held-out-shoe reference split。
- 已将 GeometryRelationEstimator、演示 reference bank、纯几何 benchmark、observation token、DP3 对照和最终移除 simulator pose 追加为后续阶段。
- 开始实现无 ID 第一阶段：统一 estimator API、成功演示 reference bank 和 observation-only 几何 benchmark；DP3 接入将在几何达标后进行。
- 已确认现有 NDF validator 的 query 路径仍依赖完整资产 mesh、functional/toe metadata，不能通过替换 `shoe_id` 参数实现真正 observation-only；开始拆分 estimator 与 evaluator。
- 决定统一 prediction 以 observation-only `correction_T_world` 为主；simulator pose 仅在 estimator 外部适配到旧 relation token，避免要求真实点云具有 actor-local 坐标系。
