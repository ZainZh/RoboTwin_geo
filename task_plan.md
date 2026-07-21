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

## 约束
- 将比较实验改为可从单一 `RoboTwin_geo` checkout 运行，不依赖本机其他源码仓库。
- 保留用户及其他 AI 已有改动。

## 错误记录
| 错误 | 尝试 | 处理 |
|---|---:|---|
| relation 集成测试缺 `zarr` / `diffusers` | 1 | 记录为当前 Python 环境依赖缺失；其余可运行测试继续验证，不擅自安装依赖。 |
| vendored NDF CPU forward 使用硬编码 CUDA device | 1 | 将 graph index tensor 改为跟随输入 device；CPU checkpoint forward 与 validator smoke 通过。 |
| 直接导入完整任务做 runtime 单测时 Curobo 强制初始化 CUDA | 1 | 将成功指标抽成独立纯 NumPy 模块，在无仿真/GPU依赖下完成真实逻辑单测。 |
