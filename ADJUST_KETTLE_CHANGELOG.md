# adjust_kettle 任务开发记录

## 概述

在 RoboTwin_geo 项目中新增 `adjust_kettle` 任务：机器人抓起桌上的茶壶，放置到目标位置。任务逻辑参考 `adjust_bottle`。

茶壶模型来源：用户自定义数据集 `~/Datasets/Teapot/`（60 个 Objaverse GLB 文件）。

---

## 一、新增文件

### 1. `envs/adjust_kettle.py`
- 任务主文件，继承 `Base_Task`
- 当前配置：
  - 模型：`092_teapot`，可用 model_id = [4, 9]
  - 朝向：`qpos = [0.707, 0.707, 0, 0]`（GLB Y-up → SAPIEN Z-up，绕 X 轴旋转 90°）
  - 放置高度：`zlim = [0.78]`
  - 目标位置：沿用 adjust_bottle 的值（左 [-0.25, -0.12, 0.95]，右 [0.25, -0.12, 0.95]）

### 2. `assets/objects/092_teapot/` （整个目录）
- 从 `~/Datasets/Teapot/` 导入的茶壶资产，共 10 个变体（base0 ~ base9）
- 目录结构：
  ```
  092_teapot/
  ├── collision/base{0-9}.glb    # CoACD 多部件凸分解碰撞网格
  ├── visual/base{0-9}.glb       # 原始 Objaverse 视觉网格
  ├── model_data{0-9}.json       # 模型元数据（center, extents, scale, functional_matrix）
  └── points_info.json           # 抓取点/功能点描述
  ```
- **稳定性测试结果**（50 seed 全环境测试）：
  - model_id 4: 100% 稳定
  - model_id 9: 100% 稳定
  - model_id 5: 50% 稳定
  - 其余 (0,1,2,3,6,7,8): 0% 稳定
- 因此 `adjust_kettle.py` 中只使用 model_id [4, 9]

### 3. `script/import_glb_assets.py`
- 批量导入 GLB 文件到 RoboTwin 资产格式的工具脚本
- 功能：
  - 加载 GLB，过滤掉面数 < 50 的碎片（Objaverse 模型常有 2000+ 断开碎片）
  - 用 CoACD 做多部件凸近似分解（阈值 0.05）
  - 计算 AABB 包围盒的 center 和 extents
  - 按目标尺寸计算 scale_factor
  - 生成 visual GLB、collision GLB、model_data JSON、points_info JSON

### 4. `script/debug_stability.py`
- 单独测试模型在地面上的稳定性（不依赖完整任务环境）

### 5. `script/debug_stability_full.py`
- 在完整任务环境中测试每个 model_id 的稳定性（遍历 50 个 seed）
- 通过 monkey-patch `load_actors()` 记录每次选中的 model_id

### 6. `description/task_instruction/adjust_kettle.json`
- 任务语言指令模板，含 10 条 seen + 5 条 unseen 描述
- 模式：`{A}` = 茶壶模型路径，`{a}` = 操作手臂

---

## 二、修改的原有文件

### 1. `task_config/_eval_step_limit.yml`
- 新增一行：`adjust_kettle: 400`

---

## 三、之前的尝试记录（已废弃）

以下文件在早期尝试中被修改，后来方案改为使用 092_teapot，这些修改已不再需要：

### 废弃：091_kettle 方案
- `assets/objects/091_kettle/model_data{0-5}.json` — 添加了 functional_matrix
- `assets/objects/091_kettle/points_info.json` — 添加了 functional_points
- 原因：091_kettle 模型全部不稳定，改用自定义导入的 092_teapot

### 废弃：053_teanet 方案（更早期测试）
- `assets/objects/053_teanet/model_data{1,4,5,6,7}.json` — 添加了 functional_matrix
- `assets/objects/053_teanet/points_info.json` — 添加了 functional_points
- 这些是早期测试时修改的，与 adjust_kettle 任务无关

---

## 四、遇到的问题及解决过程

### 问题 1：GLB 坐标系不匹配
- **现象**：茶壶躺倒在桌面上
- **原因**：GLB/glTF 使用 Y-up，SAPIEN 使用 Z-up。`qpos=[1,0,0,0]`（单位四元数）不做转换
- **解决**：设置 `qpos=[0.707, 0.707, 0, 0]`，绕 X 轴旋转 90°，将 Y-up 转为 Z-up

### 问题 2：所有茶壶都不稳定（第一轮）
- **现象**：所有 seed 都报 UnStableError
- **原因**（三个叠加问题）：
  1. Objaverse 网格有 2059 个断开碎片 → 碰撞网格质量差
  2. 使用了 OBB（有向包围盒）而非 AABB（轴对齐包围盒）→ center 和 extents 不准
  3. `zlim=0.752` 太低，物体部分嵌入桌面
- **解决**：
  1. 过滤掉面数 < 50 的碎片再做分解
  2. 改用 `mesh.bounds`（AABB）代替 `mesh.bounding_box_oriented`（OBB）
  3. 提高 `zlim` 到 0.78

### 问题 3：单凸包碰撞网格不够好
- **现象**：茶壶底部不平，无法稳定站立
- **原因**：单个凸包把壶嘴、把手等都包进去了，底面变得不规则
- **尝试**：pybullet V-HACD 分解 → 只产生 1 个部件，无效
- **解决**：使用 CoACD 库（`pip install coacd`），阈值 0.05，产生 9~133 个凸部件

### 问题 4：CoACD 之后仍然大部分不稳定
- **现象**：全 10 个模型测试，仅 30% seed 通过
- **分析**：不同茶壶形状差异大，有些形状确实不适合物理仿真
- **解决**：逐个测试 10 个 model_id 的稳定性，筛选出 100% 稳定的 model_id [4, 9]

### 问题 5：抓取点在模型中心而非把手
- **现象**：机器人从茶壶顶部/中心抓取，抓起后甩出去
- **原因**：`contact_points_pose` 放在几何中心，不在把手上
- **状态**：⚠️ 待解决 — 需要用 `python script/create_object_data.py 092_teapot` 交互标注把手位置

---

## 五、当前状态与下一步

### 当前状态
- ✅ 茶壶模型已导入（092_teapot，10 个变体）
- ✅ 稳定变体已筛选（model_id 4 和 9）
- ✅ 茶壶朝向正确（壶底朝下）
- ✅ 任务代码、配置文件、语言指令已创建
- ❌ 抓取点位置不正确（在几何中心，应该在把手上）

### 下一步
1. **标注把手抓取点**：
   ```bash
   cd ~/github/RoboTwin_geo
   conda activate RoboTwin
   python script/create_object_data.py 092_teapot
   ```
   在交互界面中为 model_id 4 和 9 标注把手位置的 contact_points_pose

2. **重新测试**：
   ```bash
   rm -rf data/adjust_kettle && CUDA_VISIBLE_DEVICES=0 python script/collect_data.py adjust_kettle demo_clean
   ```

3. **（可选）扩充模型**：从 60 个原始 GLB 中导入更多并测试稳定性，增加可用变体数量

---

## 六、2026-05-04 继续修改：放置到红色目标物块

### 修改目标
- 原任务的目标位姿是悬空点，且 `check_success()` 直接返回 `True`，不适合采集可用于算法对比的数据。
- 新目标改为：抓起桌面上的茶壶，并放置到桌子中央固定红色物块上。

### 已实现内容
- 在 `envs/adjust_kettle.py` 中新增静态红色目标物块 `target_block`。
- 任务固定使用左手；红色目标物块放在左侧。
- 将红色目标物块设置为略大于 `model_id = 13` 茶壶 footprint 的高台，当前尺寸为 `0.24m x 0.14m x 0.10m`。
- 放置目标位姿改为显式使用红台中心 `TARGET_BLOCK_XY`，不再从 box functional point 间接推导，避免目标 pose 和红台视觉位置不一致。
- Expert 放置逻辑改为让茶壶 actor 中心 XY 对齐红台中心，Z 高度使用“红台顶面 + 茶壶初始中心离桌面高度”；不再用茶壶 functional point 对齐目标。
- 成功判定同步改为检查茶壶 actor 中心是否位于红台中心附近，符合“茶壶底面放到目标支撑面、茶壶中心在目标 XY 点”的任务定义。
- 将茶壶初始位置移到桌面左后方，避免与中央红块重叠。
- 新增初始化 seed 过滤：若茶壶初始状态接触红块，或茶壶 actor/功能点 XY 距离红块过近，直接抛 `UnStableError` 并跳过该 seed。
- `script/collect_data.py` 会在任务失败时打印 `adjust_kettle` 的 seed 诊断信息，方便观察初始距离、接触状态和最终判定误差。
- 当前实现保持使用已标注把手抓取点的 `model_id = 13`。
- 放置目标高度改为动态计算：红块顶面高度 + 茶壶功能点相对桌面的初始高度。
- `place_actor()` 改为放置后打开夹爪，并在释放后原地等待稳定，避免夹爪上移时带偏茶壶。
- `check_success()` 改为真实判定：
  - 茶壶功能点 XY 接近红块中心；
  - 茶壶功能点 Z 接近目标高度；
  - 茶壶接触红块；
  - 茶壶不接触桌面；
  - 双夹爪均打开。
- 更新 `description/task_instruction/adjust_kettle.json`，语言指令明确要求把茶壶放到红色物块上。

### 后续验证建议
```bash
python -m py_compile envs/adjust_kettle.py
python -m json.tool description/task_instruction/adjust_kettle.json
CUDA_VISIBLE_DEVICES=0 python script/collect_data.py adjust_kettle demo_clean
```

### 直接验证结果
- GPU 空闲后用无渲染 probe 实测 seed 0：
  - 初始化过滤有效，茶壶初始状态不再压到红块。
  - `model_id = 13` 能规划，但释放后茶壶滑落/甩到桌面，最终不满足成功判定。
  - `model_id = 9` 在 seeds 0-5 中均未找到规划成功样本，不能直接替代。
  - 临时壶身中心抓取也规划失败。
- 当前结论：任务的 seed 观测和失败诊断已可用，但 expert 轨迹本身还不可靠，不能直接大规模采集数据。
- `demo_clean.yml` 增加 `max_seed_tries: 500`，避免没有有效 seed 时无限搜索。

---

## 七、2026-05-04 expert 修正：使用壶底中心作为放置点

### 背景
- 对比 `place_empty_cup.py`、`place_phone_stand.py`、`place_container_plate.py` 后确认：RoboTwin 的“放到某支撑物上”通常不是对齐 actor root，而是对齐对象的稳定放置 functional point。
- `092_teapot/model_data13.json` 现有 `functional_matrix[0]` 描述为茶壶中心/轴线点，不是壶底接触点；直接用 actor root 或该 functional point 会导致可视壶体和红色高台错位。

### 已实现内容
- `envs/adjust_kettle.py` 在任务加载时给当前 kettle actor 追加一个临时 functional point：壶底中心点。
  - 位置来自 `model_data13.json` 的 `center` 和 `extents`：`bottom_center = center; bottom_center[1] -= extents[1] / 2`。
  - 不修改全局 asset json，避免影响其它实验。
- Expert 保持手柄抓取方式不变，只把 `place_actor()` 改为用新增的壶底中心 functional point 对齐红色高台顶面。
- `check_success()` 改为检查壶底中心点：
  - 壶底中心 XY 接近红色高台中心；
  - 壶底中心 Z 接近红色高台顶面；
  - 茶壶接触红色高台；
  - 茶壶不接触桌面；
  - 双夹爪均打开。
- 初始化 seed 诊断新增 `initial_place_point_xy`，用于判断真实壶底位置是否靠近或压到红色高台。
- 为了提高采集成功率，任务初始化收敛为左手可达设置：
  - 固定使用左手；
  - 茶壶初始 actor root 在左侧小范围 `x=[-0.29,-0.27]`, `y=[0.09,0.12]`；
  - 固定 `model_id=13` 的手柄可达朝向 `KETTLE_QPOS=[-0.6018,-0.6026,-0.3631,-0.3779]`；
  - 仍然抓手柄，不改为抓壶身。

### 验证结果
- `python -m py_compile envs/adjust_kettle.py script/collect_data.py` 通过。
- 无渲染 probe 实测 seeds 0-9：
  - 成功 seeds：0, 3, 5, 7, 9。
  - `UnStableError` seeds：1, 2。
  - 规划失败 seeds：4, 6, 8。
- 成功样本最终诊断均满足：
  - `final_contact_block=True`
  - `final_contact_table=False`
  - `final_place_point_xyz` 接近 `target_xyz=[-0.08,-0.12,0.841]`

### 当前限制
- 仍有 seed 会因为物理不稳定或规划失败被跳过；`max_seed_tries: 500` 用于批量采集时自动搜索足够有效 seed。
- 如果后续需要更高成功率，应继续优化左手 IK 路径或为手柄补更多 contact points，而不是改为抓茶壶壶身。
