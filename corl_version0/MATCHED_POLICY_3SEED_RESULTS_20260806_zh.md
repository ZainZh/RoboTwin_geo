# Matched Policy Pilot：三训练 Seed 固定划分 Raw/EMA 结果

日期：2026-08-06
状态：单任务、3个独立训练 seed、e300 早期诊断；不是论文闭环主表。

## 1. 为什么做这个实验

原稿没有回答一个关键公平性问题：策略提升究竟来自 semantic field，还是来自额外对象点分支、更多参数或对象坐标本身。本实验固定 DP3 主干、action route、query、训练预算和数据划分，只替换对象分支每个 query 点携带的特征：

1. XYZ only；
2. XYZ + part probability；
3. XYZ + 128D semantic field embedding。

formal matched Utonia 尚未纳入，因为 Beat 原始 `{A}` object support 数据目录缺失；现有 context-nearest Utonia zarr 只用于工程 smoke，manifest 明确禁止作为论文对照。

## 2. 严格匹配协议

三条路线都来自同一份 Beat Cube 实机 ZED 数据：80 episodes、2774 frames、20维 EEF absolute-6D action。

- XYZ 与 part-prob 从 field zarr 派生；
- action、state、scene point cloud、episode boundaries 逐元素不变；
- semantic query xyz 完全一致；
- policy seeds：20260805、20260806、20260807；
- 每个 seed、每条路线训练300 epochs，每 epoch 11 updates，共3300 updates；
- batch size：256；
- 同一 DP3 主干、同一128维 point encoder output；
- validation：dataset seed 0、val ratio 0.02；episode 50、67，共73个 sequence samples；
- 每个训练结果使用相同的20个 diffusion-noise seeds 离线评测；raw 与 EMA 使用成对噪声；
- 指标：held-out diffusion imitation loss，越低越好；
- 跨模型统计单位是独立的 policy training seed，而不是 diffusion-noise repeat。

旧 workspace `.ckpt` 将 OmegaConf、dill、optimizer 和权重混合在同一个 pickle 容器中，不能被 `weights_only=True` 安全读取。本轮按完全相同配置重跑，同时输出仅含 raw/EMA state_dict 和 primitive metadata 的安全 companion；9份 companion 均由独立评测进程用 `weights_only=True` 回读。

## 3. 三 Seed 结果

### 3.1 Raw model：逐训练 seed

| policy seed | semantic field | XYZ only | XYZ + part probability | 从好到坏 |
|---:|---:|---:|---:|---|
| 20260805 | 0.00977856 | 0.01066560 | **0.00954789** | part-prob < field < XYZ |
| 20260806 | 0.00940211 | 0.01042274 | **0.00834129** | part-prob < field < XYZ |
| 20260807 | 0.01132413 | **0.00944482** | 0.00963765 | XYZ < part-prob < field |

### 3.2 跨训练 seed 汇总

下表的 `±` 是3个独立训练 seed 的 sample standard deviation，不是20个 diffusion-noise repeats 的波动。

| 表征 | 输入形状 | 参数量 | Raw mean ± seed SD | EMA mean ± seed SD |
|---|---:|---:|---:|---:|
| semantic field embedding | `[128,131]` | 273.7122M | 0.01016827 ± 0.00101855 | 0.01016575 ± 0.00101710 |
| XYZ only | `[128,3]` | 273.7040M | 0.01017772 ± 0.00064622 | 0.01017567 ± 0.00065325 |
| XYZ + part probability | `[128,5]` | 273.7041M | **0.00917561 ± 0.00072394** | **0.00918120 ± 0.00072455** |

以三 seed Raw 均值计算：

- part-prob 比 field 低9.76%，并且3/3 seeds 都更低；
- part-prob 比 XYZ 低9.85%，但逐 seed 只在2/3 seeds 更低；
- field 比 XYZ 仅低0.09%，两者跨 seed 均值实际上接近相同，field 逐 seed为2胜1负。

Raw model 的逐训练 seed paired difference：

| 差值（左减右） | seed 20260805 | seed 20260806 | seed 20260807 | mean ± seed SD | 左侧更低 |
|---|---:|---:|---:|---:|---:|
| field - XYZ | -0.00088704 | -0.00102063 | +0.00187931 | -0.00000945 ± 0.00163708 | 2/3 |
| field - part-prob | +0.00023067 | +0.00106082 | +0.00168648 | +0.00099265 ± 0.00073029 | 0/3 |
| XYZ - part-prob | +0.00111771 | +0.00208145 | -0.00019283 | +0.00100211 ± 0.00114154 | 1/3 |

EMA 的逐 seed 排序与 Raw 完全一致；两者差异远小于训练 seed 带来的波动。因此目前没有证据表明 EMA 选择能改变路线结论。

## 4. 单 Seed 结论为什么必须修正

seed 20260805 下，field 相对 XYZ 的 `field - XYZ = -0.00088704`，在20个固定 diffusion-noise seeds 上均更低。那个结果只证明了同一个已训练模型对采样噪声的稳定性，不能证明换一次训练初始化仍然成立。

扩展到3个独立训练 seeds 后，seed 20260807 上差值反转为 `+0.00187931`，而且反转幅度大于前两个 seed 的优势。最终 field 与 XYZ 的跨 seed 平均差只有 `-0.00000945`，比 seed 间标准差 `0.00163708` 小两个数量级。因此：

- 不应在论文中声称完整128D field 已稳定优于 capacity-matched XYZ；
- 不能把20个 diffusion-noise repeats 当作20个独立模型，或据此给路线差异计算论文置信区间；
- 当前最稳定的信号是 part-prob 对 field 的3/3优势，而不是 field 对 XYZ 的优势。

## 5. 对论文贡献的含义

当前证据更支持一个收窄后的假设：**part-aware query interface 有潜力优于只输入对象坐标**。它还不支持“高维连续 semantic field embedding 对策略最有价值”。完整 field 在当前设置中可能受到以下问题影响：

- e300 只达到原配置3000 epochs 的10%，131维输入可能比5维概率更难优化；
- 128维 embedding 中存在对当前动作预测无关或有噪声的自由度；
- 三个通道的尺度、归一化或 encoder 容量可能不公平地影响收敛；
- 仅2个 validation episodes 会放大偶然性，且离线 loss 不一定和闭环成功率同序。

因此重投稿时应把“完整 embedding 优势”降为待验证假设。若 full-budget 与闭环实验仍显示 part-prob 最好，更合理的主方法应是由3D semantic field 产生稳定的 per-point part distribution，再供策略使用；128D embedding 可作为消融，而不应硬写成最优表征。

当前仍不能支持：

- 不能把 imitation loss 写成 task success rate 或物理轨迹误差；
- 不能从2个 validation episodes 推断跨实例泛化；
- 不能把 context-nearest Utonia smoke 放入论文表格；
- 不能仅凭本实验决定最终实机部署路线。

## 6. 下一轮实验决策

按信息增益和审稿风险排序：

1. **原始 full budget：** 三路线、3 seeds 统一训练到3000 epochs，仍只保存安全 final raw/EMA weights。这一步用于判断 field 的高 seed 方差是不是早停造成的，并生成可用于实机 rollout 的候选模型。
2. **闭环主指标：** 用三路线在相同初始条件做实机 paired trials，保存逐 trial 成败、初始状态与失败阶段；论文主结论必须以 task success 为准。
3. **semantic corruption：** 对 part probability 做 point-wise channel shuffle / uniform / random-label control，验证收益确实来自 part semantics，而不是额外两个数值通道。
4. **formal Utonia：** 恢复 Beat 原始 `{A}` object support 后生成严格 matched Utonia；缺少这一行仍无法完整回应 reviewer 的公平 baseline 质疑。
5. **更多验证 episode：** 固定当前 test split，不用它调参；另建立覆盖更多 episodes 或新采集 demonstrations 的 validation/test protocol，降低2个 episode 带来的偶然性。

full-budget 若仍出现 `part-prob < field ≈ XYZ`，论文应明确转向“field-supervised part-aware policy interface”；若 field 在3000 epochs 后稳定超越 part-prob，才可以把连续 embedding 的额外价值恢复为主张。

## 7. 产物与复现

实现：

- `policy/DP3/scripts/safe_dp3_checkpoint.py`
- `policy/DP3/scripts/evaluate_dp3_checkpoint_offline.py`
- `policy/DP3/scripts/summarize_matched_policy_offline_eval.py`
- `policy/DP3/scripts/run_matched_policy_pilot_safe_exports.sh`
- 对应3组 unit tests。

训练与评测：

- `outputs/paper_revision/policy_pilot_safe/`：9份 e300 safe weights；
- `outputs/paper_revision/policy_offline_eval/beat_cube_{field,xyz,partprob}_e300_seed{20260805,20260806,20260807}.json`：9份逐模型报告；
- `outputs/paper_revision/policy_offline_eval/beat_cube_e300_three_seed_summary.json`：跨 seed 完整汇总；
- `outputs/paper_revision/policy_offline_eval/beat_cube_e300_three_seed_summary.csv`：逐 seed 可分析表。

每份模型评测 JSON 包含20次完整 loss、raw/EMA paired delta、validation episode IDs、checkpoint SHA256、运行环境、耗时与峰值显存。汇总脚本会拒绝 validation split 或 diffusion protocol 不匹配的输入。
