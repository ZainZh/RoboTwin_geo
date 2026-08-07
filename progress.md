# 工作进度

## 2026-08-06（远程RTX 6000 Ada最小部署）
- 新SSH端口 `112.69.3.12:43678` 已连通；实例为RTX 6000 Ada 49140MiB，GPU空闲，PyPI出口正常。
- 根overlay已扩容到144GB，当前可用144GB；`workspace_is_volume=false`，并非持久volume，recycle/destroy会丢失。
- 镜像现有Python 3.12.13 + torch 2.11.0/cu128，但缺Utonia需要的spconv/torch-scatter/flash-attn及semantic trainer依赖；决定建立隔离Python3.11环境对齐本机geo-utonia，不污染镜像main。
- 最小源码、Hammer 140MB数据和549,168,522-byte Utonia权重已同步；目录经`rsync --checksum --dry-run`逐文件校验，权重SHA256为`a0a2e5e234ce943082ae9ba26bcff74e285f16ed8d04fed5c36ffe8dc30ea353`。
- 远程隔离环境已对齐Python3.11/torch2.7+cu126/spconv2.3.8/torch-scatter2.1.2/flash-attn2.8.3；Utonia weights-only GPU加载通过。
- 首次训练smoke因缺rtree失败；补rtree1.3后发现无Embree时e1需2m57.9s，补齐本机`embreex2.17.7.post6`后3 epochs总约37s。
- threads1精确profile的steady train/val约4.6--4.8s/1.7s；双进程e3墙钟38.7--40.5s，aggregate throughput约1.85x，无OOM/NaN。
- 双进程e1显存profile的每进程train peak allocated约1459MiB、val约853MiB。seed20260805/06四变体e4000已由远程supervisor启动。
- 本机`hammer-remote-sync-watcher-20260806.service`每30s检查远程marker，每个variant完成即回传run/log/manifest并做checksum dry-run。

## 2026-08-05
- 接到 CoRL v0 被拒稿后的完整论文/代码审查与补实验规划任务。
- 按 `planning-with-files` 技能恢复已有鞋子 SE(3) 计划；无未同步会话内容。
- 将旧实现阶段暂缓，新增 CoRL 重投审查阶段 22–28。
- 初步盘点确认：原稿与审稿意见位于未跟踪的 `corl_version0/`；物体表征目录含开发版和 release 版；DP3 目录含指定训练入口及完整预处理/配置/评估链。
- 首次读取技能说明时遇到 bwrap loopback 权限错误，改用审批后的只读命令解决。
- `apply_patch` 工具及审批后的同名命令均受该错误影响；首次等价补丁行数错误，核对行号后应用最小补丁。
- 完整读取三位 borderline review、meta-review、10页实际匿名审稿版、16页当前修订版和13页视频稿；确认当前版主要补文字附录，核心实验缺口仍在。
- 逐层审计 semantic field：PartNext split/sampling、双视图变换、tri-plane、三项 loss、validation/checkpoint及开发版/release一致性。
- 恢复真实 field 权重：Hammer epoch3714/0.857、Mug epoch3964/0.834、Spoon epoch54/0.665；确认全部 CE=1.0而非论文0.5，release hammer best仅epoch1。
- 恢复 Beat/Stir meta：Beat 80/80、Stir实际70/80；均用128点、debug colors、radial normals、random observed-point queries，与论文256点描述不同。
- 审计指定 absolute6d-global DP3：state14/action20、reference-camera frame、每对象独立PointNet、validation仅2 batches、2% episode split、run-dir/resume污染风险。
- 审计 Utonia point-wise：输入预处理不matched且现有route为joint14，无matched EEF20 wrapper；审计DINO lifting：无depth visibility、nearest 16x16 patch、等权多视角和mean fallback。
- 确认当前无可复现2D policy baseline和主表逐trial records；已有query-count工具仅定性，可快速扩展为mIoU/consistency/correspondence benchmark。
- 阶段22–23完成，进入论文主张—证据—审稿质疑—代码事实矩阵与优先实验设计。

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

## 2026-08-05（投稿核查续）
- 核对官方投稿窗口：ICRA 2027 截止 2026-09-15，RA-L 滚动投稿，T-RO 适合作为更完整扩展路线；RSS/IROS 2027 尚未以官方信息确认。
- 检索到 2026-02 的 PA3FF/PADP，与当前稿件核心叙事高度重叠；将修稿策略从“补原审稿消融”升级为“收窄可防守贡献 + 正面对照新竞品 + 建立定量表征证据”。
- 完成 465 行中文修订文档 `corl_version0/REVISION_AND_EXPERIMENT_PLAN_zh.md`：包含 reviewer 共识、主张/证据/代码矩阵、事实冲突、重定位、P0–P2 实验、统计协议、ICRA/期刊包、工程路径、止损门和时间表。
- 文档已通过 `git diff --check`；阶段 24–27 完成。阶段 28 保持 pending，下一步从 R1/R2 表征 evaluator 与 E0 manifest/logger 开始实施。
- 本轮未启动训练或改动模型逻辑，避免在实验定义与用户资源信息确认前占用 GPU 或污染旧结果。

## 2026-08-05（本机补实验启动）
- 用户授权在本机单张 RTX 4090 上开始实验；若计算资源不足再汇报申请扩容。
- 继续按文件化计划记录所有配置、结果和错误；已恢复完整 task/progress/findings，session catch-up 无未同步输出。
- 阶段 28 完成，阶段 29 开始：先做只读资源/数据/环境审计，再实现 R1/R2；不直接启动大规模重训练。
- 资源初检：RTX 4090 24GB，约 22.9GB 空闲；发现 RoboTwin/geo-utonia 环境和本机 semantic checkpoint 集合。
- PartNext 配置默认指向 `/home/zheng/Datasets/PartNext_mesh`；进入数据完整性、依赖与 evaluator API 细查。
- 数据/API 初审确认 R1 可直接用 dataset 的 surface labels/model IDs；旧 checkpoint 只能得到 held-out val diagnosis，不是独立 test 结论。
- R2 不能直接重复 `dataset[index]`，因为 query 和 support 都会重采样；需新增固定 mesh query 的准备层，并将 normalization drift 作为单独控制变量。
- 锁定 R1 复用路径：checkpoint args + canonical labels + release dataset/model forward；新 loader保持 `weights_only=True` 安全边界。
- 锁定 R2 不能复用 query-count循环的根因：每轮 query也重采样；将从 annotated record建立一次固定 query、多次独立 support的 builder。
- CUDA smoke：RoboTwin(torch2.4/cu121)与geo-utonia(torch2.7/cu126)均可用4090；安全恢复三权重hash/labels/split args。
- 首次 release import 在 RoboTwin 因 `utonia` 不在 Python path失败，已记录；转查导入 helper与geo-utonia环境。
- 替代方案成功：`geo-utonia` 可原生 import Utonia + release dataset/model，确定为 evaluator 运行环境。
- 校验 legacy/local alias配置逐字节一致，并记录 Utonia 524MiB 权重 SHA256；R1 provenance条件齐备。
- 完成 R1 CLI/输出设计，准备新增 evaluator与纯指标单测。
- 首次新增 evaluator因目标在 symlink外被Git拒绝，未产生文件；已调整为在本仓库 `policy/DP3/scripts` 实现并只读导入 semantic field。
- 已在仓库内新增 R1 evaluator、共享 confusion/mIoU/provenance工具和4个纯单测；semantic field外部树保持只读。
- evaluator实现安全 weights-only加载、legacy config hash校验、逐实例/逐重复/逐部件CSV与JSON、split/checkpoint/Utonia/git/runtime/VRAM manifest；进入语法与单测验证。
- R1新增文件 py_compile 与 diff check通过。
- 第一次 unittest命令被DP3包级副作用/缺diffusers阻断，测试未实际运行；已记录并切换到独立脚本执行方式。
- 改用独立测试文件后4个 R1 metrics/provenance单测全部通过；evaluator CLI完整import/`--help`通过。
- 阶段29完成、阶段30进入端到端GPU smoke；下一步先跑Hammer 1个val实例，检查数据、模型、输出与显存。
- Hammer 1-instance GPU smoke通过：accuracy 0.9355、mIoU 0.8787，约0.46s/实例，peak allocated约612MiB；输出manifest/CSV完整。
- 阶段30完成，证明代码/data/model/output端到端可运行且4090资源充足。
- 阶段31开始：三类别完整旧val split各5 repeats，均使用FP32与原始5000/2048点。
- R1 Hammer完整5×5完成：accuracy 0.9237、mIoU 0.8580；发现一个稳定困难实例。
- R1 Mug完整16×5完成：accuracy 0.9823、mIoU 0.9636；无资源或数据错误。
- R1 Spoon 5×5完成：accuracy 0.9817、mIoU 0.9641。
- 统一R1汇总发现Mug旧val没有Closure GT点；正式split必须加part coverage验证。
- 阶段31完成：三类别结果/manifest/CSV均已保存，4090峰值约616MiB、无需额外算力。
- 阶段32开始，先在Hammer实现有GT的fixed-query/support perturbation benchmark。
- 已新增 R2 fixed-query benchmark：对每个annotated mesh只采样一次query+GT，随后独立改变support count/dropout/noise/view crop。
- 同时输出真实support-normalization与固定reference-normalization两种模式、embedding cosine/L2、label agreement、query坐标漂移、mIoU下降和逐trial provenance。
- 新增normalization/embedding纯函数单测；进入语法、单测和1-instance最小条件GPU smoke。
- R2新增代码通过6单测、py_compile和diff check；1-instance全14条件×双normalization GPU smoke通过。
- smoke显示5000→512 support较稳，256/128明显下降；random dropout比同数量独立resample稳，partial crop方向敏感。
- 阶段32完成、阶段33开始，先跑Hammer 5 models×14 conditions×5 trials×2 modes。
- Hammer完整R2完成：700次condition forwards、35.94s、peak allocated 612MiB，无资源压力。
- 完整聚合确认1024--5000 support随机重采样稳定，512以下快速下降；random dropout/noise较稳，单侧partial-view crop是最显著薄弱项。
- 归一化对极端crop有放大效应但不是唯一原因；下一步按同一协议运行Spoon与Mug，验证结论是否跨类别成立。
- Spoon完整R2完成：baseline 0.9653，噪声/dropout稳定，但单侧crop50%/25%降至0.5309/0.3121。
- Mug完整R2完成：baseline 0.9649（Closure缺失），对稀疏独立重采样尤其敏感，1024/512/128降至0.7822/0.6485/0.2330。
- 阶段33完成：三类共3640 condition forwards，峰值显存均低于635MiB，单4090充足。
- 阶段34开始：依据R2把Hammer calibration聚焦于fixed-query的partial-view crop与512/1024 support resampling。
- 完整审计原训练数据流：旧second-view仅SO(3)+jitter，没有稀疏/partial-view support augmentation。
- 发现原resume入口会把输出固定到源checkpoint父目录，存在覆盖风险；决定实现隔离的repo内calibration入口。
- 完成校准数据配对/loss/验证设计，下一步实现纯函数单测和1-epoch smoke。
- 新增fixed-query support perturbation纯函数与9项单测，全部通过。
- 新增隔离的Hammer calibration训练入口；语法、CLI、1-epoch GPU smoke全部通过。
- smoke峰值约916MiB且原checkpoint哈希不变；下一步先做calibrated checkpoint的R2兼容性回归，再启动完整pilot。
- Hammer 30-epoch calibration完成：内部robust mIoU +0.0237，clean mIoU +0.0066，约273s/937MiB。
- 原始R1/R2 apples-to-apples复评完成：clean保持，128--1024稀疏support与中等crop明显改善；support-normalized极端crop仍未解决。
- 新增完整结果文档corl_version0/R1_R2_CALIBRATION_RESULTS_20260805_zh.md；阶段34完成，开始matched policy pilot审计。
- 找到现有Beat/Stir 3000-epoch semantic policy checkpoint（约4.4GB/个）及其真实zarr/meta；Stir实际70 episodes而非文件名80。
- 审计DP3多点云编码/归一化/保存逻辑；确定先用Beat同一zarr派生XYZ与part-prob，再做短预算matched smoke/pilot。
- 记录磁盘余量约102GB：当前pilot足够，正式多任务多seed矩阵需要checkpoint瘦身或额外存储。
- 新增并通过4项semantic ablation zarr单测；生成Beat严格matched的XYZ `[2774,128,3]` 与part-prob `[2774,128,5]` 数据，所有非目标数组逐元素不变。
- field/XYZ/part-prob三路同seed的一步训练+验证smoke均通过，参数差异仅约8k；当前数值只作工程验证。
- 修复Utonia helper误导入非release源码的问题；`geo-utonia`已能加载576维point feature模型。
- `RoboTwin`默认缺Utonia import，下一步验证显式Utonia根或实现跨环境特征提取，再生成完全相同query的Utonia zarr。
- 显式 `UTONIA_ROOT` 后RoboTwin环境可直接在4090加载Utonia；无需安装依赖或跨环境传输。
- 完成固定query Utonia helper/builder及8项相关单测；一帧GPU前向和4帧zarr smoke均通过，query xyz与非表征数组不变。
- formal raw-support模式确认被损坏的Beat外置盘链接阻塞，并在写入前失败；开始生成明确标为非论文结果的完整context-nearest数据以继续策略工程验证。
- 完整2774帧Utonia context-nearest工程数据完成（183s/553MiB/426MiB），manifest禁止作为正式对照。
- 第四路Utonia同EEF配置一步训练+验证通过（273.7409M参数，val loss 0.2323）；四路策略工程链路现已全部打通。
- 完成batch256完整epoch profile并确定e300/3300-update pilot预算；field与XYZ同seed pilots均完成并各保存一个约4.1GiB最终checkpoint。
- field最低/末次可见val loss 0.00914/0.00974，XYZ为0.00932/0.01064；下一步同设置跑part-prob，再做checkpoint/rollout接口验证。
- part-prob e300完成（最低/末次可见val loss 0.00881/0.00958）；三路formal early-learning checkpoints齐备。
- 确认Beat pilot来自实物ZED数据、无匹配模拟task config；不制造伪simulation success，转向部署表征模式修正与固定split EMA离线评测。
- 补齐部署端 `embedding/xyz/part_prob` 特征模式、已有color/forward参数透传及checkpoint setting覆盖；新增定向回归测试通过。
- 发现legacy DP3 checkpoint不是weights-only安全格式；新增并测试tensor-only raw/EMA companion保存/加载协议，不采用不安全dill反序列化。
- 新增固定validation split离线evaluator：配对扩散噪声、raw/EMA、重复均值/标准误、数据split/checkpoint hash/显存provenance；3项单测与CLI导入通过。
- 三条formal路线按原Hydra配置安全重跑e300；所有可见validation点与旧pilot逐位一致，生成3个约2.1GiB的raw/EMA weights-only companion并独立回读通过。
- field/XYZ/part-prob在相同validation episodes 50/67、73 samples上完成20个配对diffusion seeds的raw/EMA评测。
- raw mean分别为field 0.00977856、XYZ 0.01066560、part-prob 0.00954789；排序为part-prob < field < XYZ，EMA排序相同。
- 写成 `corl_version0/MATCHED_POLICY_PILOT_RESULTS_20260806_zh.md`；明确该结果只是单seed imitation diagnostic，不是闭环成功率。

## 2026-08-06（三训练 seed 与 full-budget）
- 完成 seeds 20260806/20260807 的 field、XYZ、part-prob 共6个e300安全重跑；连同seed 20260805共9个模型，固定validation曲线均可复现。
- 9个weights-only companion全部完成固定episodes 50/67、73 samples、20个相同diffusion-noise seeds的raw/EMA评测。
- 新增跨训练seed汇总脚本与3项单测；统计单位明确为独立policy training seed，并拒绝split/protocol/epoch不匹配的报告。
- e300 Raw跨seed均值±SD：field 0.01016827±0.00101855，XYZ 0.01017772±0.00064622，part-prob 0.00917561±0.00072394。
- part-prob在3/3 seeds优于field；field对XYZ为2胜1负且均值差仅-0.00000945，单seed的field>XYZ结论不稳健。
- 写成 `corl_version0/MATCHED_POLICY_3SEED_RESULTS_20260806_zh.md`；旧单seed文档已加superseded提示。
- 当前9份e300安全权重占19GB，磁盘剩70GB；4090计算够用，存储成为更紧的资源约束。
- 新增full-budget断点/磁盘保护wrapper，只保存最终raw/EMA安全权重，40GiB以下拒绝继续启动。
- 已以user systemd service启动3 routes×3 seeds×e3000矩阵：`dp3-full-e3000-3seed-20260806.service`。
- 启动后确认field/seed20260805正常训练，显存约12.3GB、GPU利用率约79%，无其他计算任务冲突。
- 已调度`dp3-full-e3000-eval-20260806.service`等待训练结束，自动安全评测9个checkpoint并生成e3000三seed JSON/CSV汇总。
- evaluator/summarizer泛化到任意epoch预算；两组各3项单测、shell syntax和e300重汇总全部通过。
- e3000矩阵9/9训练与9/9固定split raw/EMA评测全部以exit code 0完成；跨seed raw loss为part-prob 0.009016±0.000194、XYZ 0.009914±0.000495、field 0.010136±0.001990。
- part-prob对XYZ为3/3更低且平均低约9.05%；field只在1/3 seed最优、policy训练方差最大，完整预算未建立continuous embedding优势。

## 2026-08-06（双机加速实验计划）
- 用户新增授权远程RTX 6000 Ada服务器（SSH host `112.69.3.12:43562`），允许与本机4090并行运行大量实验，并可按显存提高batch size。
- 第一要务改为快速完成审稿所需实验；新增阶段41--50，覆盖独立R1 test、扩展R2、三loss消融、support/query消融、formal Utonia、visibility-aware DINO、sim-real gap和闭环评测。
- 下一步先做远程只读环境/数据/代码/存储审计，再按可比性与数据依赖拆分双机任务；所有远端run必须保存resolved config、seed、checkpoint/data hash、runtime和输出manifest。

## 2026-08-06（新远端144GB与双机并行）

- 用户重建RTX 6000 Ada实例，节点磁盘上限144GB；按实测最小Hammer训练包评估足够，禁止同步208GB完整本机工作树。
- 并行分工：远端复查/最小部署；本机4090运行seed20260807 loss消融；R2三类别扩展结果独立审计成文档。
- 首次Hammer 1-epoch smoke因启动器落到系统Python、缺`torch`而在训练前失败，未产生模型；改为显式RoboTwin解释器后使用新唯一目录重试。
- 新远端连续9次连接均返回`Connection refused`；需要确认实例Running状态和重建后的SSH端口，当前未做数据同步。

## 2026-08-06（R2 robustness extension审计）
- 完成三类别正式产物审计：Hammer/Mug/Spoon分别300/960/300条trials，均与对象数×6 conditions×5 trials×2 normalizations完全一致，未发现缺失或NaN/Inf。
- 完成outlier 5/10/20%与normal-facing keep75/50/25%的reference/support聚合和统计边界复核；明确normal-facing只是法向筛选proxy，不是真实camera single-view。
- 新增`corl_version0/R2_ROBUSTNESS_EXTENSION_RESULTS_20260806_zh.md`，记录类别表、跨类别等权macro、完整性/provenance和不可夸大的论文表述。

## 2026-08-06（本机 Hammer loss ablation 启动）

- 审计确认首次smoke目标目录没有可见产物；保留该路径、不删除、不复用，以新唯一目录重试。
- launcher新增显式`CONDA_ENV`数组展开，同时拒绝带空格的`PYTHON_BIN`；5项单测、bash语法、dry-run与diff check通过。
- 两次1-epoch Full-fixed smoke均exit 0；profile smoke耗时12秒，峰值显存3371MiB、GPU利用率77%、温度44°C、功耗202W。
- smoke的best/best_sem/last/epoch四份checkpoint均可由`torch.load(weights_only=True)`读取，格式为`semantic_field_weights_only_v1`，不含optimizer/scaler。
- 本机正式seed20260807四变体e4000已由user service `hammer-loss-ablation-local4090-seed20260807.service`顺序启动；输出根为`outputs/paper_revision/hammer_semantic_loss_ablation_local4090_seed20260807_e4000_20260806`，日志为其`training.log`。
- 正式Full-fixed已进入训练，配置为batch6/split42/CE1.0/SupCon0.2/Consistency0.1；启动时服务active、MainPID 1365234、主训练PID 1365239，GPU约3.4GiB，无OOM/NaN/异常退出。

## 2026-08-06（Hammer loss ablation统一评测准备）

- 新增12-run CPU严格审计/等待/计划工具与跨3 training seeds的R1/R2汇总器；默认audit，wait不改变mode，GPU运行需双显式开关。
- 统一计划为12个best_sem checkpoints各跑R1 val×5与R2 val×5，共24条FP32命令；R2锁定20 conditions×2 normalizations。
- 审计检查config、loss权重、split、非消融协议、last epoch4000、best epoch范围、weights-only格式和checkpoint SHA；缺36个文件的CLI验证fail-closed且不生成manifest。
- 新增4项CPU-only单测并通过；RoboTwin解释器下R1/R2 help导入通过，未加载模型或占用GPU，正在运行的本机训练服务保持active。
- 2026-08-06：对远端两个正式Hammer进程完成约63秒无侵入资源采样；未停止、修改或重启seed20260805/06。
- GPU 60个1秒样本的利用率均值29.25%、中位22.5%、p95 98%，其中17/60为0%、10/60不低于90%；显存恒定4300MiB，功耗均值131.82W、p95 197.82W。GPU只在短时计算burst饱和，整段不属于compute或显存饱和。
- CPU cgroup额度为7.68核；窗口内平均使用5.48核，但630个period中308个被throttle（48.9%），累计新增148.58 CPU秒节流。瓶颈是数据/CPU阶段的突发限流与GPU空洞，而非GPU容量。
- checkpoint在63.421秒内由epoch 409/404推进到417/413，合计17 epochs，真实aggregate吞吐0.2681 epoch/s（等效3.73秒/aggregate epoch；seed05约7.93秒/epoch、seed06约7.05秒/epoch）。
- 结论：第三进程显存和GPU算力空间充足，可能通过阶段错峰获得非线性的小幅吞吐增益；第四进程在当前7.68核quota下不应盲开。保持当前两个正式进程不动，优先增加CPU quota/vCPU或把额外seed分发到独立节点；若要扩并发，先以唯一输出目录做短3/4进程profile。
- 2026-08-06：完成独立`no_consistency/e10/seed20999999`第三进程profile，wall 98.358秒、RC=0；completion marker与SHA manifest存在，额外进程已退出，两个formal始终保持运行。
- 严格前60秒按checkpoint mtime排除边界后写入的epoch：两个formal分别+7/+6，profile +5，共18 epochs，aggregate 0.3000 epoch/s；相对双进程0.2681仅提升11.9%，低于预设15%门槛，而formal自身合计吞吐下降19.2%。
- 完整profile严格边界为三个进程各+10 epochs，30/98.358=0.3050 epoch/s，相对基线提升13.8%；同样未达门槛。GPU均值30.12%、p95 95.2%、峰显存6107MiB，几乎没有填平原GPU空洞。
- 三并发CPU平均6.775/7.68核，444/624 periods被throttle（71.15%），较双并发48.9%明显恶化；无OOM、NaN或traceback。因此不启动长期第三variant，后续优先提高CPU quota/vCPU或使用第二节点。

## 2026-08-06（本机 semantic-only fast path 与正式并发）

- fast path仅对semantic mode生效：保持真实Hammer semantic query/batch逐位一致，跳过与semantic loss无关的occupancy/probe/pose计算；零权重CE/SupCon不再执行昂贵kernel。
- No-SupCon仍按release实现对每个view消费一次相同大小的`torch.randperm`；实际release loss的total、semantic gradients和调用后Torch RNG state均有单测逐位比较。
- 真实Hammer采样、post NumPy/Torch RNG、loss/gradient、installer identities、run lock、completion SHA和launcher共17项测试在RoboTwin环境全部通过。
- 吞吐profile：formal-alone为0.2666 epoch/s；formal+一个fast No-SupCon为0.3592 epoch/s；formal+两个fast variants为0.4052 epoch/s，相对formal-alone aggregate提高52.0%，峰值显存7262MiB且两个profile均正常完成。
- 不停止原Full-fixed的前提下，同时启动三个正式user services：`hammer-fast-no-ce-local4090-seed20260807.service`、`hammer-fast-no-supcon-local4090-seed20260807.service`、`hammer-fast-no-consistency-local4090-seed20260807.service`。
- 启动后四个formal services均active；三个fast日志分别确认目标loss项为0，GPU采样为96%利用率、9269/24564MiB、187.5W，无OOM/NaN/traceback。
- safe入口现对未来显式`--train-mode semantic`进程默认安装并验证fast path；`SEMANTIC_FAST_PATH=0`可显式回退。实现采用运行时lazy import，避免dedicated fast入口导入safe checkpoint helper时形成循环；19项完整测试、py_compile与diff check通过，已运行四服务不受文件更新影响。

## 2026-08-06（远端 semantic-only fast path 与并发门槛）

- 远端SSH端口由实例重配后的`43562`更新为`43678`；将fast-path源码/测试、safe/fast训练入口和matrix launcher按SHA同步到`/workspace/RoboTwin_geo`，远端真实Hammer/RNG/bitwise、run-lock/SHA与launcher共17/17项测试通过。
- 独立`no_consistency/e1/seed20999801` fast smoke完成：wall 25.608秒（含启动），到epoch 1/global_step 8；weights-only checkpoint、原子`completion.json`和文件SHA均验证通过。
- 在不停止两个原始Full-fixed进程的前提下启动seed20260805/06两路正式fast No-Consistency。四路严格窗口60.728秒：Full两路各推进5 epoch；No-Consistency因结束快照分别晚于边界1.98/0.61秒，按逐epoch原子覆盖恢复为约+10/+11，总吞吐约0.5105 epoch/s，约为双Full基线0.2681的1.90倍。
- 四路GPU 60样本利用率均值49.67%、p95 99%，显存峰7689MiB，功耗均值147.78W/峰值228.43W；CPU平均6.995核，447/607 periods被throttle（73.64%）。四路有显著aggregate增益，故保留长期运行。
- 随后新增seed20260805/06两路正式fast No-SupCon，并用checkpoint mtime事件监控进行严格90.188秒六路profile：各epoch增量为Full +5/+5、No-Consistency +11/+11、No-SupCon +7/+7，总46，aggregate 0.510048 epoch/s。
- 六路相对四路约0.5105没有净吞吐提升；GPU均值49.16%、p95 99%、显存峰11885MiB，CPU平均7.594/7.68核且827/902 periods被throttle（91.69%）。按预设“总吞吐提升才保留”门槛，停止两路No-SupCon，保留其epoch 16/15附近的checkpoint与日志且不删除；两个Full和两个No-Consistency继续RUNNING。
- 新增并同步的远端No-SupCon supervisor配置SHA为`8a57df9b0bad9f910395e6f361587b5aeb3cf3a2fe77844f72c0dc5a1e7998cb`；停止前日志未检出OOM、NaN、traceback或error，run lock确保不同variant目录不会相互覆盖。

## 2026-08-06（远端 safe 入口自动 fast 验证）

- 更新后的`train_semantic_field_loss_ablation_safe.py`在semantic模式默认安装等价fast path，`SEMANTIC_FAST_PATH=0`可显式回退；远端同步SHA为`622333cfca86382ee9c8640a9ca5d0b52791c1cf353affed8bab41198c911a4f`，测试文件SHA为`c06d0099eceb9b5365472514f26b11c127decbf520ae0e59750c392ff33fe466`，core/fast-entry SHA保持不变。
- 显式设置远端`UTONIA_ROOT=/workspace/Utonia`后safe入口`--help`成功；第一次未设置该变量时仅在导入检查阶段因默认checkout不存在而失败，没有启动训练或产生目标目录。
- 唯一`no_ce/e1/seed20999802` safe-entry smoke日志确认`semantic-fast-path enabled`及marker/identity验证；完成epoch1/global_step8，weights-only格式正确，run `completion.json`与outer `.complete`/SHA manifest齐全，last checkpoint SHA为`6f483526043d31adafb9141071f14e7a332a843af78d9aeb3c71d3c0a4ea9485`。
- 更新只影响未来启动的Python进程；当前2×Full+2×No-Consistency均未停止或重启。Full结束后原supervisor继续启动No-CE/No-SupCon时会从safe入口自动使用fast path，不需要在variant边界人工抢停。
- 19:35 HKT只读快照：四个supervisor均RUNNING；Full05/06为epoch 997/986，No-Consistency05/06为175/178；GPU即时44%、7689/49140MiB、133.08W、56°C，磁盘136GB可用（使用6%），四份日志未检出OOM、NaN或traceback。
- 2026-08-06 14:30：远端只读复核确认4个CUDA主进程仍为2×Full（PID 13211/13216）+2×fast No-Consistency（PID 42103/42118），supervisor均RUNNING；No-SupCon两服务保持STOPPED。
- 2026-08-06 14:30：远端safe/fast/lock四文件SHA与本机冻结版本逐字一致；最新日志epoch为Full 1910/1890、No-Consistency 2048/2043，loss有限且No-Consistency项严格为0。
- 2026-08-06 14:30：远端即时资源为7689/49140MiB显存、55°C、115.7W；CPU配额仍7.68核，磁盘仅用8.1/144GB（余136GB）。开始持续监控窗口与slot-filler安全设计审计。

## 2026-08-06（Hammer stage42与统一评测安全审计）

- 全程设置`CUDA_VISIBLE_DEVICES=''`运行fake-checkpoint CPU测试；未加载真实模型、未启动R1/R2 GPU evaluator、未修改或重启任何训练服务。
- 代码核对确认当前12个训练run的`val_ratio=.1,test_ratio=0`，以及evaluator只从checkpoint args重建split；现有12 checkpoints不能直接提供独立test。
- 已把统一prepare工具扩展为显式`legacy_val`和`independent_test`协议；independent模式要求positive test ratio，计划自动生成`r1_test/r2_test`路径及`--split test`命令。
- 新增`--mode validate`，只读验证24个results的checkpoint SHA、experiment、evaluation status、split name/seed、val/test ratio、evaluation seed、R1 repeats/R2 trials和FP32；stale result拒绝skip。
- manifest与evaluation plan改为原子、不覆盖写出；相同内容可幂等复查，不一致的已有文件fail-closed。汇总器同步支持independent test，并用positive val/test和held-out provenance做二次paper-claim门禁。
- CPU fake matrix测试由4项扩至8项；`git diff --check`、三文件`py_compile`和8/8 unittest均通过，CLI `--help`在禁用CUDA时通过。
- 已更新`corl_version0/HAMMER_LOSS_ABLATION_EVAL_PROTOCOL_20260806_zh.md`，记录现有checkpoint边界、48实例的推荐33/5/10 split、三档最小额外训练成本和stage42 prepare/validate命令。

## 2026-08-06（本机OOM后full-state恢复协议）

- 19:55 `systemd-oomd`终止本机四路Hammer后，确认旧partial仅有v1 evaluator权重，缺optimizer/scaler/RNG；保留证据但禁止伪精确恢复。
- 新增`semantic_field_training_state.py`：`last/best/best_sem`继续写`semantic_field_weights_only_v1`，仅`last.pt`旁路原子写`resume.pt`，包含model/optimizer/scaler、epoch/global_step/best、canonical labels、Python/NumPy/Torch CPU+CUDA RNG及规范化run identity SHA。
- 所有artifact先同目录临时文件写入、flush+fsync、`weights_only=True`回读，再`os.replace`并fsync目录；release trainer的legacy unsafe fallback在wrapper内被strict loader替换。
- safe/fast均新增`--resume-from`与`--auto-resume`；run-lock只允许同一run_dir、同run_name/seed/完整config SHA恢复，direct `--resume`、identity漂移、旧v1 partial和带损坏completion marker均fail-closed。
- Hammer matrix默认追加`--auto-resume`，`AUTO_RESUME=0`可回退；NUM_WORKERS=4 dry-run确认命令正确。没有启动正式GPU任务。
- CPU测试：training-state 2/2、run-lock 4/4、fast等价11/11、Hammer launcher 5/5全部通过；safe真实parser `--help`显示新CLI，py_compile、bash -n与聚焦diff check通过。
- 冻结SHA：training-state `f2cbfe77...4429`、safe `f3a4cb0d...b430`、fast `f451f4e7...12ab`、run-lock `c925a213...16f`、launcher `7aef83bd...16a`。
- 恢复保证是epoch边界的model/optimizer/scaler与主进程RNG连续；PyTorch persistent DataLoader worker内部随机流未序列化，恢复后的后续样本不承诺逐位相同，但不影响optimizer-continuous续训。

## 2026-08-06（远端 interrupted partial 归档与动态补槽）

- 两seed No-SupCon partial在归档前均通过双门槛：无匹配训练PID、nonblocking flock成功、无inner/outer completion或外层manifest；weights-only metadata均为epoch20/global_step160/target4000，配置权重为(1,0,.1)。
- 归档前关键SHA：seed05 config/last/log为330703...392b/e4c996...3472/9e0c5e...7725；seed06为9c9342...af82/38d468...6a09/53a6a4...dbb。
- 两份partial与对应日志在持有同一run flock期间，以os.replace原子移入各自interrupted_archive/...20260806T144708Z；原标准run/log路径释放，归档后SHA逐字一致，未删除任何产物。
- 新增remote slot-filler、原子归档工具、4个one-shot supervisor任务和两组测试；本地模拟状态/TOCTOU/blocked/active-PID/held-flock/SHA/shell共13/13通过。
- controller只经supervisor启动one-shot任务；占槽数为所有CUDA compute PID，加上STARTING任务及无CUDA descendant的原顺序supervisor切换reservation，严格不超过4。
- pending优先级为两seed No-CE后两seed No-SupCon；outer/inner completion会skip，任意非活跃不完整run目录均fail-closed，EXITED/FATAL任务不自动重试，最终Python flock处理最后TOCTOU竞态。
- 远端workspace与/etc/supervisor SHA一致：controller 998a9f...865f、archive f181a3...b75d、task config 916db0...f088、filler config 51c6d0...2cf8。
- hammer_slot_filler持久RUNNING（部署后PID83102）；4个one-shot任务保持STOPPED Not started，原四个CUDA PID 13211/13216/42103/42118未中断或重启。
- 独立JSONL连续6轮均为occupancy=4,max_active=4,starts=[]，无slot_error；45秒资源窗SM为33--99%、显存恒7689MiB、温度54--61°C、功耗83--189W。
- 部署后四路推进至Full seed05/06 epoch2057/2036、No-Consistency epoch2355/2350；四日志未检出Traceback/OOM/RuntimeError/NaN。

## 2026-08-06（本机低内存profile与正式重启）

- 19:55 HKT四个旧本机服务被`systemd-oomd`同时SIGKILL；最后完整weights-only epoch分别为Full 2283、No-CE 230、No-SupCon 229、No-Consistency 355。四份日志未出现CUDA OOM、NaN或Traceback，但旧服务峰值主存合计约58GB并产生swap压力。
- 旧partial完整保留在原输出根且没有`completion.json`；因缺少optimizer/scaler/RNG的`resume.pt`，不参与正式结果，也不伪装为可恢复训练。
- 为避免SSH/登录会话退出终止持久任务，已为`zheng`启用systemd linger。首次恢复watcher直接执行无执行位脚本导致status 126并自动重试；停止旧unit后改为显式`/bin/bash`的`hammer-remote-sync-watcher-v2-20260806.service`，随后active且0次重启。
- 使用全fast、`NUM_WORKERS=4`、`OMP/OPENBLAS/MKL_NUM_THREADS=1`及每unit `MemoryHigh=9G/MemoryMax=12G/MemorySwapMax=1G`完成四变体e200恢复profile；四个服务均exit 0，无数值异常。
- profile从22:35:51--54启动；No-Consistency于22:46:35完成，其他三路于22:52:35--36完成。800 aggregate epochs约用1002秒，吞吐约0.7984 epoch/s；峰值主存Full/No-CE/No-SupCon/No-Consistency为8.5/8.8/8.2/7.5GB，全部0 swap，GPU持续采样约93--100%。
- 以同一资源边界从全新输出根`outputs/paper_revision/hammer_semantic_loss_ablation_local4090_seed20260807_e4000_fastw4_resumable_restart20260806`并发启动四个seed20260807正式e4000服务；旧partial未移动、删除或覆盖。
- 四个正式unit均设置`Restart=on-failure`并默认`--auto-resume`：`hammer-formal-restart-{full-fixed,no-ce,no-supcon,no-consistency}-seed20260807.service`。启动后均active、NRestarts=0，fast marker与目标loss置零正确，4090即时99%、约8.9GB显存。
- 22:59安全回读确认四路`last.pt`均为`semantic_field_weights_only_v1/training_resume_supported=false`，配套`resume.pt`均为`semantic_field_training_state_v1/training_resume_supported=true`且包含optimizer；Full/No-CE/No-SupCon/No-Consistency分别推进至28/28/29/50，run identity含完整config SHA。
- 部署后连续观察超过7分钟：slot-filler PID83102始终RUNNING，最近两轮15:01:28/15:01:58 UTC仍为4个原CUDA PID、occupancy4、starts空；4个one-shot从未启动。
- 最终训练快照为Full seed05/06 epoch2080/2060、No-Consistency epoch2404/2397；异常扫描仍无匹配，45秒GPU窗保持原有burst形态且温度/显存正常。
- 两份interrupted archive已回传本机outputs/paper_revision/remote_rtx6000/interrupted_archive；config/last/log六个关键SHA与归档前、远端归档后完全一致。

## 2026-08-06（远端resumable semantic stack原子部署）

- 远端subset runner调用matrix launcher并保留父环境；新launcher默认AUTO_RESUME=1，因此现有slot task无需兼容补丁，未来No-CE/No-SupCon自然获得--auto-resume。
- stage42可对同一launcher设置SPLIT_SEED=424242、VAL_RATIO=.1、TEST_RATIO=.2及新OUTPUT_ROOT；dry-run确认run name为split424242并同时带--auto-resume。
- 本机冻结SHA：training-state f2cbfe77...4429、safe f3a4cb0d...b430、fast f451f4e7...12ab、run-lock c925a213...16f、launcher 7aef83bd...16a；4个测试SHA亦冻结。
- 9个文件先传至同目录唯一staging，逐一SHA匹配后用os.replace原子发布并fsync目录；正式路径发布后9个SHA再次匹配。
- 同步前后CUDA主进程均为13211/13216/42103/42118，显存分别约2142/2144/1704/1674MiB；4个训练supervisor uptime连续，未重启或中断。
- 远端显式CUDA_VISIBLE_DEVICES空、BLAS单线程运行training-state/run-lock/launcher/fast共22项：全部通过，1项按设计skip；测试日志实际写出临时resume.pt。
- safe --help与fast显式--train-mode semantic --help均通过并显示--resume-from/--auto-resume；未运行GPU smoke。
- 实际FAST_PATH=1的No-CE/No-SupCon dry-run命令使用fast入口且同时包含--auto-resume；stage42 dry-run使用新split/test ratio且包含--auto-resume。
- 最终slot JSONL仍为occupancy4/starts空，4个future one-shot均STOPPED Not started；最终epoch为Full 2163/2143、No-Consistency 2576/2568，日志无Traceback/OOM/RuntimeError/NaN。

## 2026-08-07（本机第二次oomd登出诊断与双机续训）

- 系统journal确认23:30:03先杀终端scope、23:30:19再杀`user@1000.service/init.scope`：内存压力67.18%持续超过50%阈值，用户会话峰值50.7GB、swap峰值6.1GB；这次登出与19:55事件同属主存压力，不是CUDA OOM。
- 四份`semantic_field_training_state_v1`断点均安全可读：Full/No-CE/No-SupCon/No-Consistency为491/468/486/531 epoch，optimizer/scaler/RNG/run identity完整；GPU曾空闲但无结果丢失。
- resume identity新增仅`num_workers`可漂移的operational例外；旧stored config SHA、stored args、run_name、seed仍先自洽验证，batch size/seed/loss等scientific config变化继续fail-closed。聚焦回归11/11通过。
- 新增repo内system/user两套持久unit及单variant wrapper；当前因本机`sudo -n`需要密码，先部署已verify的user fallback，四路统一workers2、BLAS1、MemoryHigh/Max=7G/9G、SwapMax=512M、Restart=on-failure，远端sync watcher亦持久启用。
- 首次部署被严格门禁拒绝：launcher传`utonia_checkpoint=auto`而断点保存resolved绝对路径；wrapper改为原`/home/zheng/.cache/utonia/ckpt/utonia.pth`后再启动，未放宽该科学配置。
- 四路随后从491/468/486/538成功恢复并越过断点；复核时已到Full525、No-CE502、No-SupCon519、No-Consistency593，NRestarts=0，GPU 97--99%，各服务峰值约2.90--4.09GiB且0 swap。
- 远端RTX6000原四路保持RUNNING；01:16 HKT为Full 2783/2759、No-Consistency 3855/3842。两路正式No-CE已启动并写出原子resume（约epoch6），总显存约12GiB/49GiB。
- 曾短暂启动两路No-SupCon试图填充GPU空洞；复核已有严格profile证明六路以上aggregate无增益后立即停止，未见完成epoch断点；保留2×Full+2×No-Consistency+2×No-CE，待No-Consistency很快结束后自然回到4路最优并发。

## 2026-08-07（workers2长期cache节流修复与远端No-SupCon补槽）

- 11:03 HKT复核发现本机四个user unit虽为active但GPU 0%；Full/No-CE/No-SupCon/No-Consistency仅到820/898/828/892，resume mtime分别停在09:59/10:58/10:44/06:23。
- 四cgroup均长期超过MemoryHigh=7G并顶满各自512MiB swap；memory.events high达数百万、局部memory PSI约84--90%，DataLoader进程阻塞于mem_cgroup_handle_over_high，单epoch退化到约1小时。
- 从最新原子resume重启，唯一operational变化为NUM_WORKERS=0；消除每个worker复制完整mesh/cache，四个独立训练主进程继续并发。重启后总内存37GiB降到18GiB、available升至44GiB、swap降至43MiB、PSI avg10归零。
- 52秒实测窗口本机Full/No-CE/No-SupCon各+11 epoch、No-Consistency +15；11:10 HKT为871/949/879/959，GPU约89--99%。前三关键路径约4.0--4.1小时，No-Consistency约2.9小时。
- 远端2×Full、2×No-Consistency、2×No-CE均已完成e4000并写outer completion；仅2×No-SupCon缺失。
- 两个No-SupCon标准run目录由此前手动短启留下，仅含config/canonical labels且无任何epoch checkpoint，slot-filler按设计标记blocked_incomplete并拒绝覆盖。
- 暂停controller后逐seed验证无PID、无锁、无completion/manifest/last/resume；把目录与日志带SHA原子归档为interrupted_semantic_initialization_archive_v1，前后checksum均通过，未删除证据。
- 两路正式No-SupCon随后由supervisor启动，约80秒均到epoch11；52秒窗口各+11到epoch32，约4.73秒/epoch，预计约5.2小时完成。双机全部12个loss checkpoints关键路径约5--5.5小时。

## 2026-08-07（RTX6000补满四路并提前启动stage42）

- legacy矩阵当前仅2×No-SupCon运行，显存约4.2GiB/49GiB；依据既有严格profile选择总并发4而非6，补两个空槽不会跨过已验证的CPU quota最优点。
- 新增任务不是重复legacy训练，而是论文缺失的independent held-out stage42：training seed20260806，split seed424242，val=.1，test=.2，约33/5/10实例且输出根与legacy完全隔离。
- 本机stage42 orchestration 12/12测试、bash -n和dry-run通过；8个文件先上传staging并逐文件SHA一致，再原子发布到远端。远端CUDA禁用回归同样12/12通过。
- supervisor新增四个stage42 one-shot；接力controller带include-stage42门禁，legacy两seed No-SupCon完成前仍处legacy phase，完成后自动补No-CE/No-SupCon stage42。
- 更新controller只重启slot-filler，legacy No-SupCon supervisor PID 225645/225656前后不变，训练未中断。
- 手动启动stage42 Full-fixed和No-Consistency后，总CUDA occupancy从2变4；首个snapshot显存8047MiB、SM50%瞬时，JSONL明确max_active=4/occupancy=4。
- 两个新run已越过初始化并写原子resume：Full到epoch2、No-Consistency到epoch3；命令日志确认split424242、test_ratio=.2、loss tuple与独立输出目录正确。

## 2026-08-07（用户决策：统一截断至e1750）

- 用户明确将速度置于第一优先级，认为固定4000 epochs无必要，并授权当前运行任务在epoch1750停止。
- 科学协议采用统一预算而非选择性截停：legacy与independent held-out stage42的所有variant/seed均以1750为目标；已经完成的e4000 run保留原始产物，并在比较时以可验证的统一checkpoint-selection规则处理，不能伪装为e1750。
- 执行要求：运行中任务必须从原子optimizer-continuous断点续接；已达到1750的任务停止后先验证checkpoint，再生成明确e1750 provenance/完成标记；未达到者续跑到1750；随后同步更新controller、launcher、测试、回传和评测门禁。

## 2026-08-07（e1750迁移已部署）

- release universal-field trainer使用固定AdamW且没有epoch相关scheduler；epochs只控制循环终点。因此从e4000 full-state断点把目标缩至1750不会改变已完成轮次或后续至1750的优化规则。
- strict resume现只允许epochs保持或缩短，拒绝增长；epochs/save_every/num_workers是受限operational字段，seed、split、loss、batch及所有模型参数继续逐项严格校验。训练状态与run-lock测试6/6、完整聚焦套件18/18通过。
- 旧产物路径继续使用e4000标签以原地恢复，实际config/completion明确target_epoch=1750；stage42协议用ARTIFACT_EPOCH_TAG=4000与EPOCHS=1750分离，避免路径名冒充训练预算。
- 本机四路暂停在原子断点1551/1634/1566/1779；前三路已从同一optimizer/scaler/RNG恢复并以1750为终点，No-Consistency在收到决策时已超出29轮，直接生成target1750完成标记并退出。
- 本机Stage42持久接力队列已启用：绑定实际hammer-local-loss@服务，严格max=4、workers0；当前已用空槽启动seed20260805 Full，4090快照99%、8539MiB、233W。
- 远端四份断点580/578/306/435通过真实缩短验证；staging py_compile/bash和6项恢复测试通过后原子发布，四路从原optimizer/RNG恢复，RTX6000快照100%、7749MiB、214W。
- 远端Stage42 controller改为legacy优先但允许空槽回填，新增回归后本机/远端各13/13通过；只重启controller，四个CUDA训练PID 240426/240447/240470/240490前后不变。
- 本机legacy预计约12:25 HKT全部结束；本机队列会随空槽继续补Stage42。远端当前四路目标均为1750，随后No-CE/No-SupCon自动接力。

## 2026-08-07（代码云端同步）

- 用户明确要求整理并上传当前改动，只上传代码等，不上传模型文件。上传边界暂定为源码、shell/config、单元测试、计划/论文实验文档；outputs、checkpoint、数据、日志、缓存和其他二进制产物全部排除，提交前还需做大文件与敏感信息审计。
- 已确认分支corl_revise、远端GitHub origin；当前HEAD与origin/main一致，尚无本地提交。
- outputs/paper_revision为38GB，明确整目录排除；corl_version0中的31MB稿件PDF也不上传，只选择Markdown文档。
- 将采用显式路径白名单暂存，之后对Git index做独立审计，不能依赖工作树忽略规则判断最终上传内容。
- 已给`.gitignore`增加`outputs/paper_revision/`与`corl_version0/*.pdf`，并采用显式白名单暂存；index二次审计确认无模型、权重、checkpoint、数据、输出目录和PDF，最大暂存文件约120KB。
- 待提交文件的私钥/token/API key/password特征扫描无命中；全部Python与Shell语法检查通过。
- 15个无可选依赖的聚焦测试文件共85项全部通过；全量本次测试文件97项中另有5项失败，其中3项为仓库既有EEF wrapper缺失/旧断言，2项因当前环境缺Utonia可选依赖，均已核对不是模型训练回归。
