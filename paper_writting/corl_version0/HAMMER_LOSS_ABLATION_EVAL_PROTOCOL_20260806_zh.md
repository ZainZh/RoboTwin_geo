# Hammer 三项 loss 消融统一评测协议

更新日期：2026-08-07

## 1. 目的与证据边界

本协议在 Full-fixed、No-CE、No-SupCon、No-Consistency 四个变体及三个独立训练 seed 上执行完全相同的 R1/R2 评测。

第一轮仍使用旧 validation split：`split_seed=42`、`val_ratio=0.1`、`test_ratio=0`。由于 checkpoint 也在这个 validation split 上选择，该轮只能标为固定协议诊断，不能写成独立 held-out test 结果。独立 test split 重训与评测仍属于 stage 42，必须单独报告，不能和本轮混表。

## 2. 训练矩阵门禁

评测前必须通过12/12严格审计：

- 三个 seed，每个 seed 恰好四个变体；
- 每个 run 都必须有 `config.json`、`best_sem.pt`、`last.pt`；
- `last.pt` 必须达到 epoch 4000；`best_sem.pt` 允许是1–4000之间由validation accuracy选择的epoch；
- checkpoint 必须能由 `torch.load(..., weights_only=True)` 加载；
- 格式必须为 `semantic_field_weights_only_v1`，不得含 optimizer/scaler；
- split、batch、support/query点数、augmentation和非消融训练配置必须匹配；
- 三组真实loss权重必须分别是 `(1,.2,.1)`、`(0,.2,.1)`、`(1,0,.1)`、`(1,.2,0)`；
- manifest记录每个最终 `best_sem.pt` 的SHA256、best epoch、last epoch和resolved协议。

`--wait` 只等待文件并执行CPU审计，不会改变默认的`audit`模式，更不会自动启动GPU评测。真正运行必须同时显式指定 `--mode run --confirm-run-evaluators`。

2026-08-07速度优先决策后，legacy矩阵不再追求补齐统一e4000：远端已完成的e4000产物原样保留，本机运行截断为目标e1750，其中No-Consistency因决策到达时已运行到1779。故本节e4000门禁只描述原始legacy诊断协议，不能再作为最终统一预算的12-run表；正式三seed消融改由下述Stage42 e1750独立test矩阵承担。

## 3. R1协议

- split：旧val；sampling seed：20260805；repeats：5；
- support/query：5000/2048；FP32；
- 输出aggregate accuracy/mIoU、逐seed结果、per-part IoU、完整confusion、per-model和per-repeat明细。

跨seed统计以独立field训练seed为单位，报告 mean ± sample SD。不能把5次query/support重采样当成5个训练seed。

## 4. R2协议

- 固定同一批raw mesh query，仅改变support；split与seed同R1；
- 每条件5 trials；FP32；normalization为support与reference两种；
- support count：128、256、512、1024、5000；
- dropout：0.25、0.5、0.75；coordinate noise：0.0025、0.005、0.01；
- crop keep：0.75、0.5、0.25；outlier replacement：0.05、0.1、0.2；
- normal-facing proxy keep：0.75、0.5、0.25。

每个condition输出mIoU、相对reference的mIoU变化、label agreement、feature cosine、L2 drift、query-coordinate drift、per-part结果与confusion。normal-facing依然只是proxy，不代表真实z-buffer single-view。

## 5. 使用方式

假设远端seed结果已经以只读/校验方式同步到本机。下面命令只等待并审计，不运行GPU：

```bash
python_bin=/home/zheng/miniforge3/envs/RoboTwin/bin/python
manifest=outputs/paper_revision/hammer_loss_ablation_eval/training_manifest.json

"${python_bin}" policy/DP3/scripts/prepare_hammer_loss_ablation_evaluation.py \
  --seed-root 20260805=/path/to/copied/remote_seed20260805_output_root \
  --seed-root 20260806=/path/to/copied/remote_seed20260806_output_root \
  --seed-root 20260807=/home/zheng/github/RoboTwin_geo/outputs/paper_revision/hammer_semantic_loss_ablation_local4090_seed20260807_e4000_20260806 \
  --wait \
  --manifest "${manifest}"
```

生成24条评测命令但不执行：

```bash
"${python_bin}" policy/DP3/scripts/prepare_hammer_loss_ablation_evaluation.py \
  --seed-root 20260805=/path/to/copied/remote_seed20260805_output_root \
  --seed-root 20260806=/path/to/copied/remote_seed20260806_output_root \
  --seed-root 20260807=/home/zheng/github/RoboTwin_geo/outputs/paper_revision/hammer_semantic_loss_ablation_local4090_seed20260807_e4000_20260806 \
  --mode plan --manifest "${manifest}" \
  --python-bin "${python_bin}" \
  --dataset-root /home/zheng/Datasets/PartNext_mesh \
  --alias-config include/3d_semantic_train/semantic_field_release/configs/hammer.json \
  --utonia-checkpoint /home/zheng/.cache/utonia/ckpt/utonia.pth \
  --evaluation-output-root outputs/paper_revision/hammer_loss_ablation_eval/old_val_r1_r2
```

确认GPU空闲后，只有把`--mode plan`替换为以下两个参数才会运行：

```text
--mode run --confirm-run-evaluators
```

完成24/24后汇总：

```bash
"${python_bin}" policy/DP3/scripts/summarize_hammer_loss_ablation_evaluation.py \
  --manifest "${manifest}" \
  --evaluation-root outputs/paper_revision/hammer_loss_ablation_eval/old_val_r1_r2 \
  --output-dir outputs/paper_revision/hammer_loss_ablation_eval/old_val_summary
```

汇总产物包括`summary.json`、R1逐seed表、R1 per-part表、R1 confusion长表和R2全部condition长表。汇总器再次检查checkpoint SHA、split对象、R1/R2随机协议、FP32及完整40组R2 condition-normalization组合，并把统计单位固定为三个训练seed。

## 6. Stage 42 独立 held-out test 审计

当前正在训练的12个checkpoint不能直接完成严格的stage 42：

- config及checkpoint args均固定为`val_ratio=0.1,test_ratio=0`，因此当前evaluator按`--split test`构造出的dataset为空；
- 现有`best_sem.pt`由旧val accuracy选择。若保持`split_seed=42`仅把比例改为`val=.1,test=.1`，新的test恰好仍是shuffle末尾的旧val 5个实例，即已经用于checkpoint选择和多轮R1/R2分析的对象，不能通过改名成为独立test；
- `last.pt`是预定epoch 4000权重，参数本身没有由validation loss更新。它可在旧val上补充“fixed-epoch, gradient-held-out validation”诊断，但旧val已经参与方法开发和监控，仍不能标为untouched independent test。

本机Hammer目录共有48个有效mesh。现有`.1/0`划分为43 train + 5 val，每epoch 8个batch（batch 6）。推荐冻结一个新的stage42协议后从头训练：

- `split_seed=424242,val_ratio=.1,test_ratio=.2`：33 train + 5 val + 10 test，每epoch 6个batch；10个test比5个更能抵抗单一困难实例造成的方差；
- checkpoint只由5个val实例选择；训练、选择、论文调参均不得读取10个test的R1/R2结果；
- 三项loss正式独立test表需要四变体×三训练seed，即12次新训练。由于每epoch batch数由8降为6，训练工作量约为当前12-run矩阵的75%；
- 最小主张包可只重训Full-fixed×3 seeds，工作量约为当前矩阵的18.75%，先形成论文可声明的Full R1/R2；但三项loss消融仍只能引用旧val诊断；
- 折中包为Full-fixed×3 seeds，加三个ablation各1 seed，共6 runs，约为当前矩阵的37.5%。它能展示独立test方向，但ablation没有训练seed方差，不宜作为最终强结论。

3-fold/5-fold交叉验证不是最低成本方案。3-fold×四变体×单seed已需12次训练，且fold不能替代独立training-seed方差；四变体×三seed×3-fold需36次训练。以当前双GPU目标，单个预注册holdout×三seed更直接。

新的prepare入口会显式拒绝把当前`test_ratio=0`矩阵当成独立test。Stage42重训完成后仅做CPU审计/计划的命令为：

```bash
"${python_bin}" policy/DP3/scripts/prepare_hammer_loss_ablation_evaluation.py \
  --seed-root 20260805=/path/to/stage42_seed20260805_root \
  --seed-root 20260806=/path/to/stage42_seed20260806_root \
  --seed-root 20260807=/path/to/stage42_seed20260807_root \
  --split-seed 424242 \
  --evaluation-protocol independent_test \
  --expected-val-ratio 0.1 \
  --expected-test-ratio 0.2 \
  --mode plan \
  --manifest outputs/paper_revision/hammer_loss_ablation_eval/stage42_training_manifest.json \
  --python-bin "${python_bin}" \
  --dataset-root /home/zheng/Datasets/PartNext_mesh \
  --alias-config include/3d_semantic_train/semantic_field_release/configs/hammer.json \
  --utonia-checkpoint /home/zheng/.cache/utonia/ckpt/utonia.pth \
  --evaluation-output-root outputs/paper_revision/hammer_loss_ablation_eval/stage42_test_r1_r2
```

GPU评测仍只有`--mode run --confirm-run-evaluators`能启动。24条结果完成后，用同一组参数把`--mode plan`改成`--mode validate`，即可只在CPU上验证24/24结果的checkpoint SHA、experiment、test split、split seed、evaluation seed、R1 repeats/R2 trials和FP32设置。已有完全一致的manifest/plan可重复验证；任一文件内容不一致时入口拒绝覆盖。

## 7. Stage42 独立训练矩阵的可执行调度（e1750，已部署）

独立矩阵现在冻结为`hammer_stage42_independent_test_v1`：

- 四个loss variants与legacy矩阵逐项相同，训练seeds仍为`20260805/06/07`；仅数据划分改为`split_seed=424242,val=.1,test=.2`，Hammer实际为33 train + 5 val + 10 untouched test；
- `batch=6,epochs=1750,support=5000,query=2048,SO(3),jitter=.005,AMP,optimizer/model/loss`全部锁定。运维差异只有本机为避免OOM登出采用`workers=0`、远端`workers=8`；
- seed到硬件固定：本机RTX 4090承担`20260805+20260807`共8 runs，远端RTX 6000 Ada承担`20260806`共4 runs。一个training seed的四变体绝不跨硬件；
- 每seed使用全新的唯一root，run name显式含`split424242`，与legacy `split42`没有目录或锁名碰撞；
- launcher默认只打印命令。只有显式`RUN_STAGE42=1`才会训练；每个run继续走`flock`、原子`resume.pt`、`--auto-resume`、completion SHA和`weights_only=True` identity协议。已有完整run必须通过全配置与checkpoint校验才能skip；已有不完整/身份漂移run会fail-closed；
- `hammer_stage42_protocol.py --mode audit`要求`config/last/best/best_sem/resume`全部存在、completion哈希集合精确、last/resume达到e1750、三份评测checkpoint为weights-only v1、resume为training-state v1，并逐字段核对split/loss/seed/model/optimizer/采样参数。迁移前已建立的目录/run name仍含e4000作为artifact tag，真实预算只能以config与completion的target_epoch=1750判断。

计算量可由batch更新数直接核对：原始legacy e4000 12-run为`12×4000×8=384,000` batches；Stage42 e1750为`12×1750×6=126,000` batches，仅为前者32.8%。按实测并发吞吐和当前断点，三seed×四variant训练预计在2026-08-07 16:00--17:30 HKT完成，波动主要来自CPU quota与各variant单轮成本。

当前只读状态检查（不会创建目录、不会启动GPU）：

```bash
python_bin=/home/zheng/miniforge3/envs/RoboTwin/bin/python
"${python_bin}" policy/DP3/scripts/hammer_stage42_protocol.py --mode plan
"${python_bin}" policy/DP3/scripts/hammer_stage42_local_queue.py --once
STAGE42_ROLE=local RUN_STAGE42=0 \
  bash policy/DP3/scripts/run_hammer_stage42_independent_matrix.sh
```

本机queue controller是variant-major顺序的8个one-shot任务，严格`max_active=4`。它把GNOME remote desktop的已知图形context排除，但所有训练或未知CUDA PID都计入槽位；RUNNING/STARTING且暂时没有CUDA child的managed unit也预留一个槽。四个legacy unit中，任何一个inactive且没有completion都会阻止stage42补位；成功完成释放的槽立即由低优先级stage42补上。controller已于12:15 HKT持久部署并启动首个seed20260805 Full任务。

```bash
RUN_STAGE42=1 systemd-run --user --collect \
  --unit=hammer-stage42-local-queue \
  --property=Restart=on-failure --property=RestartSec=20 \
  --property=MemoryHigh=1G --property=MemoryMax=2G \
  /usr/bin/env RUN_STAGE42=1 PYTHONUNBUFFERED=1 \
  /home/zheng/miniforge3/envs/RoboTwin/bin/python \
  /home/zheng/github/RoboTwin_geo/policy/DP3/scripts/hammer_stage42_local_queue.py \
  --confirm-start --max-active 4 --poll-seconds 15 \
  --event-log /home/zheng/github/RoboTwin_geo/outputs/paper_revision/hammer_stage42_local_queue/events.jsonl
```

若不使用动态queue、只在当前本机矩阵全部结束后顺序启动8 runs，则一键命令是：

```bash
STAGE42_ROLE=local RUN_STAGE42=1 \
  bash policy/DP3/scripts/run_hammer_stage42_independent_matrix.sh
```

远端successor、四个Stage42 programs和独立wrapper均已部署。调度规则为legacy pending优先；若所有未完成legacy都已active/reserved，Stage42可提前回填空槽，而无需等待严格完成屏障。所有CUDA PID仍统一计入`max_active=4`。本机与远端编排回归各13/13通过；发布controller时只重启filler，四个训练CUDA PID前后保持不变。
