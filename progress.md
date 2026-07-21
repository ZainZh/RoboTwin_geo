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
