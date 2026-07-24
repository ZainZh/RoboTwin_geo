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
- [in_progress] 16. 统一 NDF/PCA 为 GeometryRelationEstimator 接口与预测数据结构
- [pending] 17. 从成功演示末端 object_pointcloud_A/B 构建 reference bank
- [pending] 18. 实现 observation-derived 纯几何 benchmark 与 held-out-shoe 划分
- [pending] 19. 保留 simulator current pose，离线生成 observation-derived token zarr
- [pending] 20. 训练 observation-goal DP3 并与 baseline/oracle/goal-table NDF 配对比较
- [pending] 21. 由观测估计 current correction，移除 simulator object pose

## 约束
- 将比较实验改为可从单一 `RoboTwin_geo` checkout 运行，不依赖本机其他源码仓库。
- 保留用户及其他 AI 已有改动。
- `shoe_id` 只允许用于数据集划分和评测分组，不得进入 estimator、token builder 或 policy。
- estimator 输入不得包含 query 资产路径、`functional_matrix`、`orientation_point` 或预计算 per-shoe goal。
- 训练/参考实例和 held-out 测试实例必须按鞋划分，避免通过 observation 最近邻隐式恢复实例身份。

## 错误记录
| 错误 | 尝试 | 处理 |
|---|---:|---|
| relation 集成测试缺 `zarr` / `diffusers` | 1 | 记录为当前 Python 环境依赖缺失；其余可运行测试继续验证，不擅自安装依赖。 |
| vendored NDF CPU forward 使用硬编码 CUDA device | 1 | 将 graph index tensor 改为跟随输入 device；CPU checkpoint forward 与 validator smoke 通过。 |
| 直接导入完整任务做 runtime 单测时 Curobo 强制初始化 CUDA | 1 | 将成功指标抽成独立纯 NumPy 模块，在无仿真/GPU依赖下完成真实逻辑单测。 |
