# 接手鞋子 SE(3) 放置对比实验

## 目标
从 `policy/DP3/train_shoe_se3_placement_comparison.sh` 及其依赖代码还原实验设计、各对照组、数据与训练流，并确认当前实现状态和下一步工作。

## 阶段
- [complete] 1. 检查仓库状态、比较脚本和关联入口
- [complete] 2. 追踪各实验配置、模型与几何表征数据流
- [complete] 3. 对照 PCA/SE(3) 姿态回归脚本，归纳研究假设
- [complete] 4. 检查昨天的改动、缺口与潜在问题
- [complete] 5. 向用户汇报理解并提出接手后的执行顺序
- [complete] 6. 审计服务器缺失的 NDF 源码与第三方 import 链
- [complete] 7. 解耦 baseline 预处理对 NDF 的非必要依赖
- [complete] 8. 将真正需要的 NDF 运行时代码纳入仓库或明确安装项
- [complete] 9. 添加干净环境依赖检查并验证 baseline/NDF 入口
- [complete] 10. 追踪鞋子推理 rollout 终止与成功率判定
- [complete] 11. 将成功判定改为适配“持物到位但不松爪”的 placement 指标
- [complete] 12. 添加回归测试并验证评估入口
- [complete] 13. 审计评估产物是否足够恢复逐 episode 失败统计
- [complete] 14. 明确可恢复字段、必须重跑字段与服务器检查命令
- [complete] 15. 定义移除 shoe_id/资产 metadata 的无身份泄漏实验协议
- [pending] 16. 统一 NDF/PCA 为 GeometryRelationEstimator 接口与预测数据结构（暂缓，先完成论文重投审查）
- [pending] 17. 从成功演示末端 object_pointcloud_A/B 构建 reference bank
- [pending] 18. 实现 observation-derived 纯几何 benchmark 与 held-out-shoe 划分
- [pending] 19. 保留 simulator current pose，离线生成 observation-derived token zarr
- [pending] 20. 训练 observation-goal DP3 并与 baseline/oracle/goal-table NDF 配对比较
- [pending] 21. 由观测估计 current correction，移除 simulator object pose


## CoRL v0 论文重投审查（2026-08-05 起）

### 目标
完整核对 `corl_version0` 稿件与审稿意见、`include/3d_semantic_train` 的物体表征训练实现，以及 `policy/DP3/train_semantic_pointwise_hybrid_eef_absolute6d_global.sh` 下游策略链路，形成证据可追溯的论文问题诊断、修改方案和优先级实验计划，并据此继续补实验。

### 阶段
- [complete] 22. 盘点并逐页/逐条阅读论文、审稿意见及相关已有笔记
- [complete] 23. 还原语义场训练目标、数据、模型结构、输出表征与 DP3 注入链路
- [complete] 24. 建立“论文主张—当前证据—审稿质疑—代码事实”对照矩阵
- [complete] 25. 设计最小可投稿实验包、扩展实验包、统计协议与消融优先级
- [complete] 26. 核对当前候选会议/期刊的范围和时间窗口，给出投稿路线建议
- [complete] 27. 写成 `corl_version0/REVISION_AND_EXPERIMENT_PLAN_zh.md` 并自检完整性
- [complete] 28. 与用户确认实验优先级和首轮本机资源（单张 RTX 4090）
- [complete] 29. 审计 GPU、Conda 环境、PartNext 数据、现有 checkpoint 与 evaluator 可复用接口
- [complete] 30. 实现并测试 R1 held-out-instance part metrics evaluator（mIoU/per-part/confusion）
- [complete] 31. 在现有 Hammer/Mug/Spoon checkpoint 上运行 R1，保存 manifest/JSON/CSV
- [complete] 32. 实现并测试 R2 fixed-query/support-perturbation benchmark
- [complete] 33. 在现有 checkpoint 上运行 R2，记录精度、稳定性、显存与耗时
- [complete] 34. 基于 R1/R2 结果决定并启动 fixed-query consistency Hammer calibration
- [complete] 35. 完成 matched field/XYZ/part-prob e300 pilot 与 Utonia 工程 smoke
- [complete] 36. 安全导出 raw/EMA 权重并完成固定 split 配对离线评测
- [complete] 37. 将单 seed pilot 扩展到至少 3 policy seeds
- [pending] 38. 恢复 Beat 原始 object support 并补 formal matched Utonia
- [pending] 39. 在相同初始条件上运行三路线实机闭环 trial
- [complete] 40. 在本机 RTX 4090 上完成三路线 × 三 seeds 的 e3000 full-budget matrix 与固定split评测
- [complete] 41. 审计本机4090与远程RTX 6000 Ada的代码、数据、环境和存储，建立双机可复现实验分工
- [complete] 51. 在远程RTX 6000 Ada的144GB非持久实例上部署Hammer消融最小包，校验SHA并完成1-epoch full-fixed smoke
- [in_progress] 52. 远程并行运行seed20260805/06的Hammer四变体e4000，每variant完成即回传并校验
- [complete] 53. 在不干扰seed20260805/06的前提下采样远端GPU/CPU与checkpoint推进率，审计第三/第四并发进程的吞吐风险
- [complete] 54. 运行隔离的第三进程10-epoch profile，以同步GPU/CPU/checkpoint数据验证三并发加速门槛并自然退出额外任务
- [in_progress] 42. 将R1从旧validation diagnosis升级为独立held-out test split并形成正式mIoU/per-part/confusion表
- [in_progress] 43. 扩展R2到显式viewpoint、single-view、segmentation contamination/outlier并形成鲁棒性正式表
- [in_progress] 44. 完成Hammer三项field loss消融：Full-fixed/No-CE/No-SupCon/No-Consistency，多seed评测R1/R2
- [complete] 54. 实现隔离的semantic-only等价fast path，证明实际semantic输入、loss/gradient与RNG流一致并完成吞吐/break-even profile
- [in_progress] 55. 本机从全新目录并发完成seed20260807四变体正式e4000可恢复训练，逐run写入锁、full-state与SHA completion manifest
- [pending] 45. 完成support/query与readout消融：observed-point/no-resampling、part-prob、field embedding和XYZ
- [pending] 46. 恢复raw object support并完成formal matched Utonia策略对照
- [complete] 56. 将semantic-only fast path部署到远端并完成4/6正式进程严格吞吐profile；按aggregate门槛保留两路No-Consistency、停止无净增益的两路No-SupCon
- [complete] 57. 远端safe入口默认自动安装semantic fast path并完成SHA、CLI与唯一e1产物验证，使Full完成后的后续variants无需抢停即可自动加速
- [complete] 58. 审计当前loss矩阵的独立test可声明性，完成legacy/independent双协议prepare、CPU validate、汇总门禁与stage42最小重训成本文档
- [pending] 47. 实现visibility-aware DINOv2 lifting并评估D3Fields/F3RM风格强baseline的可运行方案
- [complete] 70. 补全DINOv2 visualization summary的配置provenance与逐物体visibility/fusion统计，保持旧ndarray backend兼容并完成CPU回归
- [pending] 48. 用受控感知扰动、相同policy和阶段失败统计解释sim-real gap
- [complete] 58. 为safe/fast trainer实现weights-only可读的原子full-state resume checkpoint，严格校验run/config/seed identity，并保留v1评测checkpoint兼容
- [complete] 59. 保全归档远端两seed被停止的No-SupCon partial，并部署严格max=4、No-CE优先、incomplete fail-closed的持久supervisor slot-filler
- [complete] 60. 诊断本机systemd-oomd，完成四路workers4低内存e200 profile、linger/watcher持久化与可恢复正式fresh重启
- [complete] 61. 将full-state resumable semantic stack原子同步至远端，完成CPU-only回归、CLI/dry-run验证并保持现有4路训练进程不重启
- [complete] 62. 诊断本机第二次oomd整会话登出，允许仅workers operational漂移续跑，部署持久限内存四路服务并恢复双机并行实验
- [complete] 63. 按用户速度优先决策，将当前及后续Hammer loss/stage42矩阵统一从固定e4000迁移为e1750，保持断点连续并避免变体间预算不一致
- [in_progress] 64. 验证e1750停止、完成标记、回传与评测协议，更新剩余队列及最终ETA
- [in_progress] 65. 审计当前工作树并仅提交代码、测试、脚本与必要文档，排除模型/数据/训练产物后推送现有Git云端
- [complete] 66. 回传并校验RTX6000上的4组Stage42训练产物，确保远端释放前模型与完成元数据安全落盘
- [complete] 67. 将CE-only纳入Stage42协议并在双机完成三training-seed统一e1750训练
- [complete] 68. 对Full/No-CE/No-SupCon/No-Consistency/CE-only统一运行independent-test R1与稳定性/鲁棒性R2
- [complete] 69. 汇总三seed统计、决定辅助损失定位并更新重投稿实验表述
- [pending] 49. 完成多任务/实物paired closed-loop评测、统计与失败分析
- [pending] 50. 汇总所有实验为可重算表格、图、视频和重投稿正文
## 约束
- 将比较实验改为可从单一 `RoboTwin_geo` checkout 运行，不依赖本机其他源码仓库。
- 保留用户及其他 AI 已有改动。
- `shoe_id` 只允许用于数据集划分和评测分组，不得进入 estimator、token builder 或 policy。
- estimator 输入不得包含 query 资产路径、`functional_matrix`、`orientation_point` 或预计算 per-shoe goal。
- 训练/参考实例和 held-out 测试实例必须按鞋划分，避免通过 observation 最近邻隐式恢复实例身份。

## Beat Block Hammer formal-fixed 闭环协议（2026-08-08）

### 目标
在不启动仿真/GPU的前提下，实现可审计的标准锤 fixed-seed RoboTwin 闭环协议：精确评估 seeds 100000..100099，关闭视频与 expert screening，保存逐 candidate JSONL 及 summary JSON，并保持 `_result.txt` 兼容。

### 阶段
- [complete] 71. 完整审阅 eval loop、task config 与现有 CPU 测试边界
- [complete] 72. 新增标准 020_hammer formal-fixed task config
- [complete] 73. 实现逐 candidate JSONL 与 summary JSON 落盘，保持旧结果文件
- [complete] 74. 新增不依赖 SAPIEN/GPU 的 seed 序列与结果格式回归测试
- [complete] 75. 运行定向测试、语法检查并记录精确产物

## Matched Utonia 在线闭环路由（2026-08-08）

### 目标
让 semantic policy 显式支持 `matched_utonia` 在线观测：先由 semantic query 路径生成与训练 Zarr 一致的 128 个 query XYZ，再在原始 `object_pointcloud_A` support 上查询 Utonia 576D 特征，输出 `[128,579]` 到 `utonia_point_cloud_A`；保持其他 semantic 输出模式及 legacy Utonia 路由不变，并以 CPU mock 单测验证契约。

### 阶段
- [complete] 76. 完整审阅 deploy/config/semantic query/Utonia helper/checkpoint shape_meta 链路
- [complete] 77. 实现显式 matched_utonia 配置、模型加载与在线输出路由
- [complete] 78. 添加 key/query/shape/配置路由及旧模式兼容 CPU mock 单测
- [complete] 79. 运行定向测试、语法检查并形成实现报告

## 错误记录
| 错误 | 尝试 | 处理 |
|---|---:|---|
| formal-fixed 协议首次两次 `apply_patch` 被 bwrap loopback 错误阻断 | 2 | helper 在任何写入前失败；改用同一 `apply_patch` 程序的审批 PTY/stdin 入口，后续补丁正常落盘。 |
| matched Utonia 首轮定向测试在 `RoboTwin` 环境导入 semantic release 时缺 `utonia` 包 | 1 | py_compile 已通过但测试未执行；改用仓库现有 `geo-utonia` CPU 环境运行，不安装依赖、不启动 GPU。 |
| matched Utonia 第二轮在 `geo-utonia` 环境导入 DP3 时缺 `diffusers` | 1 | helper 的4项CPU测试通过、deploy两模块未加载；定位到 `/home/zheng/github/Utonia`，改用具备DP3依赖的 `RoboTwin` 环境并显式加入该只读源码路径。 |
| accepted/skipped 测试 fixture 的无上下文补丁命中首个 `accepted=True` | 1 | 5项CPU测试立即以断言失败捕获；改用 candidate_seed 完整上下文分别修正 successful/unstable 记录。 |
| relation 集成测试缺 `zarr` / `diffusers` | 1 | 记录为当前 Python 环境依赖缺失；其余可运行测试继续验证，不擅自安装依赖。 |
| vendored NDF CPU forward 使用硬编码 CUDA device | 1 | 将 graph index tensor 改为跟随输入 device；CPU checkpoint forward 与 validator smoke 通过。 |
| 直接导入完整任务做 runtime 单测时 Curobo 强制初始化 CUDA | 1 | 将成功指标抽成独立纯 NumPy 模块，在无仿真/GPU依赖下完成真实逻辑单测。 |
| 读取 planning skill 时 bwrap 无法配置 loopback | 1 | 按沙箱规则改用已审批的只读命令；成功读取并恢复现有计划。 |
| `apply_patch` 工具及审批后的同名命令均受 bwrap loopback 错误影响；首次 `git apply` 补丁行数有误 | 4 | 核对精确行号后使用最小 `git apply --unidiff-zero` 补丁，不使用脚本重写文件。 |
| 默认 Python 环境解析 `.pt` 时缺少 `torch` | 1 | 已确认仓库现有 `RoboTwin`/`geo-utonia` Conda 环境；后续改用具备 torch 的项目环境做只读 checkpoint metadata 提取。 |
| 尝试以 `torch.load(weights_only=False)` 解析 checkpoint metadata 被安全审查拒绝 | 1 | 未规避；改用项目环境中的 `weights_only=True` 安全模式，成功读取基础 metadata。 |
| 一次向 `pdfinfo` 传入多个 PDF 只返回 usage | 1 | 改为逐文件调用或并行独立调用；不重复同一错误命令。 |
| `RoboTwin` 环境直接 import release semantic field 找不到 `utonia` | 1 | 不重复同命令；先检查仓库 Utonia checkout/导入辅助，再测试已有 `geo-utonia` 环境或显式项目路径。 |
| 尝试在 `include/3d_semantic_train` 新增 evaluator 被 Git 拒绝 `beyond a symbolic link` | 1 | 该目录是指向外部 checkout 的 symlink；不重试写入外部树，改在本仓库 `policy/DP3/scripts` 放 evaluator/tests，并把 semantic field root 作为显式只读依赖。 |
| `python -m unittest policy/DP3/...` 触发 `policy.DP3.__init__` 并因 geo-utonia 缺 `diffusers` 失败 | 1 | 测试本身未加载；不安装无关DP3依赖，改为直接执行独立测试文件或指定 scripts 为 discovery top-level。 |
| Conda run默认捕获python stdin导致heredoc无输出 | 1 | 改用no-capture-output模式；同一只读检查成功，不再使用被捕获形式。 |
| `utonia_feature_utils` 优先导入外部非 release tree，缺少其声明的模型文件 | 1 | 改为显式优先使用 `include/3d_semantic_train/semantic_field_release`，`geo-utonia` 中 weights-only 加载 576 维 Utonia point feature 成功。 |
| `RoboTwin` 环境运行 Utonia 单测时找不到 `/home/zheng/github/Utonia` | 1 | 不重复裸环境命令；后续先验证显式 `UTONIA_ROOT/PYTHONPATH`，若依赖不兼容则把 Utonia 特征提取与 DP3 训练解耦。 |
| Beat Cube 80 条原始演示符号链接的外置盘目标目录不存在 | 1 | formal raw-support builder 在任何写入前安全失败；保留严格模式，另用明确标记 `formal_comparison_eligible=false` 的 context-nearest 数据只打通工程管线，并请用户恢复原始数据。 |
| 完整 EEF wrapper 测试集引用已删除的非-global脚本并保留旧suffix断言 | 1 | 本次新增的global实机特征契约2项定向测试通过；3个既有失效单独记录，不误归因于新改动。 |
| 旧DP3 `.ckpt` 混合OmegaConf/dill/optimizer，`weights_only=True`拒绝；非安全反序列化被安全审查阻止 | 2 | 不扩展dill白名单、不绕过；新增tensor-only raw/EMA companion格式，按原Hydra配置安全重跑e300后再评测。 |
| `--epochs` 参数的zero-context补丁首次落在 `return parse_args()` 之后，CLI拒绝该参数 | 1 | 核对精确行号后移到return之前；两组单测与e300重汇总通过。 |
| Hammer loss smoke由启动器默认系统`python`执行，缺少`torch` | 1 | 不重复默认启动；改为显式项目Conda解释器，先跑唯一目录1-epoch smoke再启动正式矩阵。 |
| `apply_patch`再次受bwrap loopback错误阻断，组合`git apply`及文档追加曾因hunk行数或上下文错误失败 | 10 | 已改为每文件单独的精确最小补丁并先核对实际新增行数；不再复用损坏补丁。 |
| 远端`pgrep`首轮把未转义的`|`当shell管道，`grep`首轮把含空格pattern拆成参数 | 2 | 改用无管道的单模式`pgrep`与无空格多`-e`异常pattern；两次均为只读失败，训练未受影响。 |
| 2026-08-06 19:55 `systemd-oomd`同时终止本机4路Hammer训练，旧partial只有weights-only权重而无optimizer/scaler/RNG | 1 | 保留旧partial作诊断证据但禁止伪精确恢复；新增独立原子`resume.pt`协议与显式/auto resume强校验。 |
| 新helper首次新文件补丁声明行数不足，尾部restore/CLI函数未落盘 | 1 | 首次CPU测试以ImportError捕获；核对`wc/tail`后补齐尾部并重新py_compile/测试，不启动GPU。 |
| RoboTwin torch版本不支持序列化`torch.uint32` NumPy RNG keys | 1 | 改为无损`int64` tensor编码，恢复时显式转换为`np.uint32`；weights-only round-trip与后续随机数逐值测试通过。 |
| slot-filler搜索命令包含不存在的顶层scripts目录，首次unittest在scripts工作目录仍传仓库相对模块名 | 2 | 搜索改为现有目录；测试改用当前目录模块名，13/13项实际加载通过。 |
| 新增controller/archive文件首次patch低估新文件行数导致尾部被截断，随后一次追加hunk计数错误 | 2 | py_compile立即捕获；用wc/tail核对并以git apply --recount补齐，远端同步前语法/测试全通过。 |
| 远端真实slot dry-run硬编码不存在的/usr/bin/supervisorctl | 1 | fail-closed且starts为空；改为实例实际/usr/local/bin/supervisorctl并回归。 |
| supervisorctl status因存在STOPPED/EXITED服务正常返回3，被check=True误判 | 1 | 仅接受status约定的0/3，其他返回码仍fail-closed；新增测试并通过真实快照。 |
| 事件日志零上下文补丁首次把--event-log插到parse_args之后、global插到while内部 | 1 | py_compile在同步前捕获IndentationError；移动到解析前/main入口并重跑13/13测试。 |
| supervisor配置使用stdout_logfile=/dev/stdout导致supervisorctl tail无独立日志 | 1 | 保留stdout并新增fsync JSONL事件日志，controller单独重载后连续快照可审计。 |
| fast入口裸--help因未显式--train-mode semantic按契约退出1 | 1 | 未启动训练；改用--train-mode semantic --help，隐藏CUDA后成功验证resume CLI。 |
| 首次只读审计误用不存在的`/home/zheng/miniconda3/envs/utonia/bin/python` | 1 | 未写入任何文件；查询实际Conda环境后改用`/home/zheng/miniforge3/envs/geo-utonia/bin/python`安全读取断点。 |
| 从仓库根以`python -m unittest policy/DP3/...`运行独立测试时触发DP3包导入并缺`diffusers` | 1 | 不安装无关依赖；切到`scripts`目录按独立模块名运行，11/11实际测试通过。 |
| 持久服务首次恢复使用`utonia_checkpoint=auto`，与断点记录的resolved绝对路径不一致 | 1 | 严格身份门禁在训练前拒绝四路；改回原绝对权重路径后四路均从正确epoch恢复。 |
| 恢复后短时内存正常但workers2运行数小时后四cgroup均陷入MemoryHigh reclaim，服务仍显示active | 1 | 结合resume mtime、memory.events/PSI和wchan定位；从原子断点改workers0重启，52秒epoch吞吐恢复且PSI归零。 |
| 复核恢复日志时误用rg -E，被解释为encoding参数 | 1 | 只读命令未影响训练；改用rg默认正则alternation后成功读取四路连续epoch。 |
| 在scripts工作目录仍给sed/bash传仓库相对policy/DP3/scripts路径，首轮文件读取失败 | 1 | 测试命令实际按当前目录名称成功运行12/12；随后用当前目录相对路径完整审阅并bash -n/SHA验证。 |
| planning-with-files文档补丁三次因bwrap helper或缺失hunk行号失败 | 3 | 停止重复失败入口；读取精确行号后使用`git apply --unidiff-zero --recount`应用最小补丁。 |
| e1750整批补丁首次因JS模板中的反引号解析、随后因上下文偏移未应用 | 2 | 没有部分写入；移除模板歧义并读取精确行号，拆为零上下文原子补丁后成功。 |
| 首次真实e1750恢复被save_every 4000→1750身份差异拒绝 | 1 | 训练在加载断点前安全退出；确认save_every仅影响额外文件写入后纳入operational字段，并重跑18/18回归。 |
| 远端staging的13项编排测试缺两个未修改辅助文件 | 1 | 正式代码未受影响；补传wrapper/conf后同一套件13/13通过。 |
| 本机No-Consistency在人工停机前已推进至1779，超过新目标29轮 | 1 | 立即停止并以target_epoch=1750的显式完成标记保全；不得将其表述为精确e1750，正式统一矩阵使用stage42。 |
| CE-only首个多文件`git apply`使用了缺少范围的hunk头，被判定为garbage | 1 | Git在写入前整体拒绝；改为核对`nl -ba`精确行号并使用`--unidiff-zero --recount`，补丁成功且无部分旧写入。 |
| CE-only首轮语法门禁在scripts工作目录仍传仓库相对路径，且测试夹具把CE-only虚构成legacy guard | 1 | 路径错误未执行对应检查；改用当前目录文件名并让fixture读取真实CURRENT_UNITS，随后py_compile/bash-n和26/26回归通过。 |
| Stage42严格审计补丁三次因zero-context行位移插入函数实参内部 | 3 | 每次均由py_compile或真实audit在发布/评测前捕获；最终替换完整resume审计块并在本机项目环境重跑26/26通过。 |
| 首轮聚焦回归误用缺torch的系统Python | 1 | py_compile已通过但测试collection失败；改用训练实际RoboTwin环境，26/26实际加载通过。 |
| 初次远端正式评估未显式设置Utonia checkout路径 | 1 | 审计通过但evaluator在模型加载前ImportError；将`PYTHONPATH=/workspace/Utonia`固化进Supervisor配置，独立import验证后评估正常运行。 |
| 远端评估Supervisor首次start在连续失败重试期返回spawn error | 1 | 检查独立日志确认根因同为Utonia import，不是CUDA/模型；修正环境后update process group，服务稳定RUNNING。 |
| final matrix prepare把历史best_sem的e4000 metadata误当作训练未统一 | 1 | 保持config/last严格e1750，仅允许best_sem的epochs=save_every属于{1750,artifact_tag4000}且best epoch不超过1750；新增非法3999与last4000拒绝测试，27/27通过。 |
| 通用Stage42本机CLI按build_tasks默认workers4审计真实workers0产物 | 1 | 不改launcher/queue默认；CLI新增显式`--num-workers`且role默认local0/remote8，真实本机Full audit通过。 |
| 远端旧四组inner completion完整但outer completion/manifest缺失 | 1 | 根因是完成时旧严格审计拒绝历史best metadata；修复后重新进入wrapper，run-lock验证现有产物并只生成outer SHA/marker，四组均EXITED且无重训。 |
| DINO provenance搜索命令包含不存在的shell glob `test*` | 1 | `rg`已返回其余有效路径且未写文件；后续改用明确的`script/`与`policy/DP3/scripts/`测试路径。 |
| `apply_patch`受bwrap loopback错误阻断DINO provenance补丁 | 1 | helper在写入前失败；沿用已验证的`git apply --recount`最小补丁入口。 |
| 系统Python运行DINO visualization测试缺少torch | 1 | 测试未collection；切换已有RoboTwin环境并保持`CUDA_VISIBLE_DEVICES=''`，32项CPU测试实际执行。 |
| 新增summary落盘断言首次缺少json import | 1 | CPU测试以NameError捕获；补充单行import后27项visualization与5项visibility测试全部通过。 |
| 合并outer metadata时整根rsync意外包含仍在写入的CE-only resume.pt | 1 | 识别本机临时partial后立即SIGINT接收端，不影响远端训练且不发布临时文件；改为精确同步completed/manifests与已完成evaluation目录，checksum dry-run一致。 |

## RoboTwin 双GPU仿真比较（2026-08-07 起）

- [complete] 77. 冻结Beat Block Hammer/Hanging Mug仿真训练、held-out资产、固定测试seed与公平输入协议
- [complete] 78. 修复并验证uniform/shuffled part-prob构建器，确保query/action/state与概率分布不变量
- [complete] 79. 构建Beat Block Hammer matched XYZ/part-prob/field/Utonia及两条概率控制数据并完成训练smoke；DINO因原始数据缺dense depth按Phase85独立阻塞
- [in_progress] 80. 将正式六路线×三training-seed矩阵拆分到本机4090与远端RTX6000 Ada并行训练
- [in_progress] 81. 对完成模型先运行统一fixed-2闭环门禁，通过后扩展到同一100个RoboTwin测试seed的成功率、耗时和失败阶段评测
- [in_progress] 82. 在Hanging Mug重复主路线，并补clean/randomized/held-out instance与感知扰动对照
- [pending] 83. 冻结仿真模型与结果SHA，生成论文主表、置信区间、失败案例和可复算图表
- [complete] 84. 审计Beat Hammer原始HDF5的同步多视角RGB-D与相机标定，确认visibility-aware DINOv2 matched数据可行性
- [pending] 85. 复用现有投影/深度可见性/多视角融合工具实现matched-query DINOv2 zarr构建器与provenance门禁
- [pending] 86. 用合成数据和真实max-frames smoke验证bitwise invariants、visibility/fusion统计及失败关闭语义

### 本轮新增错误

| 错误 | 尝试 | 处理 |
|---|---:|---|
| apply_patch再次因本机bwrap loopback失败 | 1 | 读取精确文件结尾后改用此前验证的unidiff-zero最小git补丁，不重试同一失败入口。 |
| semantic control无上下文补丁位置漂移 | 2 | 编译门禁捕获参数错位；核心文件、mode guard与7项构建器测试均已修复并通过。 |
| 本地首次DP3 smoke误用geo-utonia解释器，缺diffusers/zarr | 1 | 训练在模型构建前退出且未写权重；改用依赖完整的RoboTwin环境后field e1成功。 |
| 五路线validator首次使用错误参数别名 | 1 | argparse在读取数据前拒绝；按真实参数重跑后全部约束通过。 |
| apply_patch再次因bwrap loopback阻断DINO子阶段记录 | 1 | helper在写入前失败；切换到精确git apply，不重复失败入口。 |
| 首个DINO规划多文件补丁因progress尾部上下文已变化而整体拒绝 | 1 | 无文件被部分修改；拆成基于当前尾部的独立补丁。 |
| findings git diff误含apply_patch结束标记而被判corrupt | 1 | Git在写入前拒绝；移除非diff标记后同一内容成功应用。 |
| DINO审计器增强补丁再次被apply_patch的bwrap错误阻断 | 1 | helper写入前失败；切换到精确git apply。 |
| 零上下文shape门禁补丁插入多行valid表达式中导致SyntaxError | 1 | py_compile在真实审计前捕获；替换完整局部表达式后10/10测试通过。 |
| 归档器首次smoke使用了不存在的`diffusion_policy/config`路径 | 1 | checkpoint已安全验证但manifest在写入前fail-closed；改为实际`diffusion_policy_3d/config`，真实manifest与source-receipt smoke均通过。 |
| rsync断点续传首次组合`--append-verify --whole-file` | 1 | rsync在传输前拒绝；移除互斥参数，不影响远端训练和已完成文件。 |
| rsync断点续传第二次组合`--append-verify --partial-dir` | 1 | rsync在传输前拒绝；恢复已验证的`--partial --partial-dir`模式，保留110MB部分文件并持续回传。 |

### Phase 79 当前门禁（2026-08-07 23:45 HKT）

- [complete] Beat Hammer field数据：50 episodes、5718 frames、joint14，finite检查通过。
- [complete] 从同一field数据派生XYZ、part-prob、uniform-prob与shuffled-prob。
- [complete] 五路线action/state/scene/query XYZ逐值一致与概率控制验证，JSON证据已落盘。
- [complete] field单路e1与XYZ/part-prob本地双路e3真实训练smoke。
- [complete] BF16/TF32、原子weights-only、safe deploy、远端数据同步和并发profile门禁。
- [complete] 正式Utonia matched数据构建并通过六路线一致性门禁。
- [blocked] visibility-aware DINOv2 matched数据：现有50条演示的100个camera-episode全部缺dense depth，须确定性重渲染/补采后继续。
- [in_progress] 六条可运行路线×三seed e300训练；之后执行固定100-seed闭环评测。

## RoboTwin 闭环评测性能优化（2026-08-08）

### 目标
在不改变固定候选seed、策略输入、物理控制频率、成功判定和方法公平性的前提下，定位并减少RoboTwin闭环推理中的场景初始化、相机渲染、点云生成和冗余数据收集开销；以相同seed的基准/优化计时与观测契约验证后再用于正式评测。

### 阶段
- [complete] 87. 对当前formal-fixed单回合做分段计时并审计相机、RGB、third-view、点云与对象点云依赖
- [complete] 88. 实现显式fast-eval配置/计时接口，保持默认协议向后兼容并添加CPU回归测试
- [in_progress] 89. 用少量固定seed运行基准/优化A-B smoke，验证观测shape/finite/query契约和成功判定一致性
  - 2026-08-09：四次首帧digest一致；planner-free setup错误和远端同步错位均产生 `evaluated_count=0`，已判无效。安全 fast smoke 为89.9s（旧baseline 138.7s）；尾部观测虽到53.2s，但 fixed-100 前34 seeds 为0/34、旧协议同区间5次成功，故已撤销。现本机 validation3 与远端六个互斥 route-group 均用安全配置从新时间戳重跑。
- [pending] 90. 达到安全加速门槛后更新后续队列；若协议不等价则保留原配置并记录瓶颈
- [pending] 91. 汇总速度、显存、成功一致性和论文结果可比性证据

## Beat Hammer matched baseline 扩展（2026-08-09）

### 目标
补齐当前六路线之外的三个关键公平对照：严格 Vanilla DP3（仅主点云与机器人状态）、GT part one-hot oracle（仅在可审计真值存在时）和 Utonia-128（不拟合任何数据、冻结随机正交投影的固定降维），随后接入同一三 training-seed、e300、joint14 与 fixed-100 闭环协议。

### 阶段
- [complete] 92. 审计 strict Vanilla 数据/配置、Hammer GT part 标签来源和 Utonia-128 无泄漏降维方案
- [complete] 93. 实现数据构建、在线部署路由、shape/provenance 门禁与 CPU 回归测试
- [complete] 94. 构建可成立的 matched 数据并验证 action/state/main-cloud/query 不变量；标准020_hammer无逐面part真值，GT oracle已按协议fail-closed
- [complete] 95. 旧路线训练已结束或按用户指示停止，已有checkpoint与结果保留；GPU队列转入语义-策略接口搜索
- [complete] 96. 旧fixed-100余项按用户指示停止，不再占用双GPU；现有有效结果仅作为新接口筛选基线

## PA3FF相似路线与当前效果退化分析（2026-08-10）

### 目标
完整阅读《Learning Part-Aware Dense 3D Feature Field for Generalizable Articulated Object Manipulation》，对照其表征、数据、策略接口和评测协议与本项目的真实实现及现有结果，定位“思路相近但下游效果不佳”的主因，形成按证据强弱排序的诊断与最小验证实验。

### 阶段
- [complete] 97. 完整提取并阅读PA3FF正文、附录、表格、训练与策略细节
- [complete] 98. 建立PA3FF与本方法在监督、场表示、query/support、规范化、策略集成和泛化协议上的逐项对照
- [complete] 99. 将现有R1/R2和RoboTwin fixed-100结果映射到候选失败机制，区分已证实问题与待验证假设
- [complete] 100. 写成可执行中文分析文档，给出修复优先级、止损条件与论文重定位建议

## 双GPU语义表征-策略稳定接入搜索（2026-08-10）

### 目标
停止旧训练/评测队列，在不篡改数据与正式协议的前提下，用本机RTX 4090和远端RTX 6000 Ada并行定位随机query、额外分支和blind maxpool的影响；实现至少一种不会稳定劣于strict Vanilla、并能在真正需要部件信息的任务上提供增益的语义策略接口。

### 阶段
- [complete] 101. 精确识别并停止两机旧GPU任务与自动接力，保留checkpoint、日志和同步服务，确认两张GPU空闲
- [complete] 102. 审计现有dataset、query构建、encoder与checkpoint兼容边界，定义可复用的paired筛选协议
- [complete] 103. 实现确定性query、zero-feature/zero-gate matched control和训练/在线逐值一致性门禁
- [complete] 104. 实现三类候选接口：零初始化残差门控、按部件soft pooling几何token、task/part-token Transformer或cross-attention
- [in_progress] 105. CPU单测和e1真实GPU smoke后，双卡并行短预算训练；用paired fixed seeds闭环淘汰明显差于Vanilla的方案
- [pending] 106. 对入选方案运行至少3个training seeds、统一checkpoint选择和fixed-100闭环，并报告均值、置信区间、失败阶段与计算成本
- [pending] 107. 在至少一个更需要语义的多实例/跨任务RoboTwin设置复验，更新论文方法、消融与限制

### 本轮接口搜索错误记录

| 错误 | 尝试 | 处理 |
|---|---:|---|
| 首次用JavaScript模板字符串生成临时launcher编辑器时，Bash变量`${...}`被解释为JS模板表达式并在任何工具执行前SyntaxError | 1 | 未产生文件修改；改用shell单引号保护的Perl q-string精确替换临时副本，bash -n和完整diff通过。 |

| RoboTwin环境直接运行两组旧semantic测试未显式加入现有Utonia checkout，import阶段失败 | 1 | 未触发GPU或写产物；按仓库既有环境边界设置PYTHONPATH=/home/zheng/github/Utonia后两组16/16通过。 |

| warm-start toy测试错误假设frozen参数量必大于adapter参数量 | 1 | helper行为正确；改为分别断言两者均大于0并核对精确trainable names，4/4通过。 |
| 临时launcher位于/tmp时用相对zarr路径，repo_root解析为/导致dry-run找不到数据 | 1 | 未启动GPU；用绝对zarr路径重跑后完整Hydra命令通过。 |
| warm-start首次把配置raw_task_name显式null解释为任务名 | 1 | 权重加载前fail-closed；加入null fallback后语法通过。 |
| legacy配置实际把raw_task_name写为字符串none，第二次仍得到none前缀 | 1 | 权重加载前fail-closed；统一将空/none/null sentinel回退到task_name。 |
| Vanilla训练后checkpoint含目标模型在dataset初始化前尚无的normalizer buffers | 1 | 只白名单跳过normalizer.*并计数，其余missing/shape仍fail-closed；测试4/4与真实e1通过。 |
