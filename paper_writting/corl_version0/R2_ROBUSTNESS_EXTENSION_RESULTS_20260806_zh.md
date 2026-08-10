# R2 Robustness Extension：Outlier 与 Normal-facing Proxy

日期：2026-08-06
状态：三类别完整运行与产物审计通过；旧 validation split 上的诊断结果，不是 independent-test 论文主结果

## 1. 结论摘要

1. Hammer、Mug、Spoon 三类输出完整：分别有 300、960、300 条 trial，均严格等于 `对象数 × 6 conditions × 5 trials × 2 normalization modes`；JSON、逐 trial CSV、condition 聚合 CSV 和逐对象 reference CSV 均存在。
2. AABB-shell outlier replacement 对类别的影响差异很大。Mug、Spoon 在 5%--20% 污染下相对稳定；Hammer 对污染明显敏感，尤其在 support-normalization 下。
3. 跨类别等权 macro 下，reference-normalization 的 outlier mIoU 为 0.8636 / 0.8652 / 0.8883（5% / 10% / 20%），support-normalization 为 0.8027 / 0.7998 / 0.8196。该非单调趋势不应解释成“更多 outlier 有益”。
4. normal-facing support proxy 随保留比例下降而稳定恶化。跨类别 macro reference-normalization mIoU 从 keep75 的 0.8687 降至 keep50 的 0.7277、keep25 的 0.5470；support-normalization 分别为 0.8732 / 0.7391 / 0.5565。
5. keep25 时平均 embedding cosine 仍约 0.963--0.964，但 label agreement 只有约 0.710--0.719、mIoU 相对 clean 下降约 0.38。平均 cosine 不能替代 pointwise part quality。
6. normal-facing 是 mesh support 上的几何代理：只按随机相机方向与表面法向的 facing score 选点，没有相机内外参、深度图、z-buffer、自遮挡或分割误差。它不能在论文中写成真实 single-view camera experiment。

## 2. 完整性与实验协议审计

| 类别 | Validation 对象数 | Reference points | Fixed queries | Trials / condition | Aggregate conditions | Trial rows | Condition forwards | FP32 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Hammer | 5 | 5000 | 2048 | 5 | 12 | 300 / 300 | 300 | 是 |
| Mug | 16 | 5000 | 2048 | 5 | 12 | 960 / 960 | 960 | 是 |
| Spoon | 5 | 5000 | 2048 | 5 | 12 | 300 / 300 | 300 | 是 |

共同协议：

- seed：20260806；
- split：checkpoint 内旧 validation split，split seed=42；
- 每个对象只采一次带 GT 的 2048 个 raw mesh-surface queries；所有 condition 与 normalization 使用相同 query 坐标和标签；
- 每个 condition 独立运行 5 个随机 trial；
- `reference`：support 和 query 均使用 clean 5000-point reference 的 normalization；因此 query-coordinate drift 恒为 0；
- `support`：按当前 perturbed support 重算 center / scale / z-shift；
- 两种模式下 tri-plane projector bounds 仍依赖当前 support AABB，因此 `reference` 只隔离输入 normalization 漂移，不等于完全固定 field bounds；
- 三个 checkpoint SHA、配置 SHA、Utonia SHA、GPU/runtime 和逐对象 split records 均写入 JSON；运行时 worktree 为 dirty，代码来源以结果中的 commit + status manifest 为准。

产物中未发现缺失 trial、空指标或 NaN/Inf。Mug validation split 依旧没有 Closure GT 点，所以 Mug 的 mIoU 只平均 Container 与 Handle。

## 3. 扰动定义

### 3.1 AABB-shell outlier replacement

保持 support 总数为 5000，将其中 5% / 10% / 20% 行替换成扩展 1.5 倍 AABB shell 内、且至少一个坐标落在原 AABB 外的合成点；其 normal 为随机单位向量，color 在原颜色范围内随机采样。

这模拟 segmentation / point-cloud contamination，但不是来自真实传感器或 SAM2 的经验错误分布。因为 outlier 会改变 support AABB，support-normalization 和 support-dependent projector bounds 都可能同时变化。

### 3.2 Normal-facing support proxy

每个 trial 采一个随机单位“camera direction”，按 `surface_normal · camera_direction` 排序，保留 facing score 最大的 75% / 50% / 25% support。

这个 proxy 只引入 normal-facing selection：

- 没有透视投影或相机视锥；
- 没有 depth consistency、z-buffer 和 mesh ray casting；
- 没有物体自遮挡、背景遮挡、深度缺失和 segmentation contamination；
- 不对应任何真实相机位姿分布。

因此下文只能称为 normal-facing proxy / normal-facing support selection，不能称为真实 single-view、RGB-D partial observation 或 camera-view benchmark。

## 4. Clean reference

| 类别 | Accuracy | mIoU | 有效 parts |
|---|---:|---:|---|
| Hammer | 0.9296 | 0.8684 | Handle、Head |
| Mug | 0.9843 | 0.9676 | Container、Handle；Closure N/A |
| Spoon | 0.9826 | 0.9658 | Handle、Head |

所有 delta 均相对同一结果文件中的类别 clean aggregate reference 计算。

## 5. Outlier 结果

表格单元为 `pooled mIoU (delta vs clean)`。

| 类别 | Normalization | Replace 5% | Replace 10% | Replace 20% |
|---|---|---:|---:|---:|
| Hammer | reference | 0.7110 (-0.1574) | 0.7307 (-0.1377) | 0.8178 (-0.0506) |
| Hammer | support | 0.5603 (-0.3081) | 0.5502 (-0.3182) | 0.6064 (-0.2620) |
| Mug | reference | 0.9422 (-0.0254) | 0.9274 (-0.0402) | 0.9091 (-0.0586) |
| Mug | support | 0.9273 (-0.0403) | 0.9241 (-0.0435) | 0.9219 (-0.0457) |
| Spoon | reference | 0.9376 (-0.0282) | 0.9374 (-0.0284) | 0.9381 (-0.0277) |
| Spoon | support | 0.9204 (-0.0454) | 0.9251 (-0.0407) | 0.9304 (-0.0354) |

以下为三个类别等权 macro average，不按 Mug 较多的对象数加权：

| Normalization | Replace | mIoU | Delta | Label agreement | Cosine | Embedding L2 | Query drift |
|---|---:|---:|---:|---:|---:|---:|---:|
| reference | 5% | 0.8636 | -0.0703 | 0.9287 | 0.9908 | 0.0777 | 0.0000 |
| reference | 10% | 0.8652 | -0.0687 | 0.9313 | 0.9910 | 0.0790 | 0.0000 |
| reference | 20% | 0.8883 | -0.0456 | 0.9419 | 0.9918 | 0.0778 | 0.0000 |
| support | 5% | 0.8027 | -0.1313 | 0.8871 | 0.9863 | 0.0978 | 0.1983 |
| support | 10% | 0.7998 | -0.1341 | 0.8800 | 0.9850 | 0.1017 | 0.1996 |
| support | 20% | 0.8196 | -0.1144 | 0.8954 | 0.9866 | 0.0975 | 0.2001 |

解释边界：

- Hammer 的 reference/support 差距很大，说明 normalization drift 和 support-dependent bounds 会放大 contamination；但 reference 模式也明显退化，不能把问题全归因于 normalization。
- Mug 的退化随污染率整体增加；Spoon 基本持平。类别、形状和 part 布局对 outlier 敏感性影响明显。
- Hammer 与 Spoon 的 20% 结果反而高于 5% / 10%。这是有限对象、随机替换位置及 support-dependent bounds 下的非单调结果，不是 outlier regularization 的证据。
- 本轮只有一个 evaluator seed，且 Hammer/Spoon 各只有 5 个对象；没有做对象级 bootstrap 或多 seed 复现，不能声称差异有统计显著性。

## 6. Normal-facing proxy 结果

表格单元为 `pooled mIoU (delta vs clean)`。

| 类别 | Normalization | Keep 75% | Keep 50% | Keep 25% |
|---|---|---:|---:|---:|
| Hammer | reference | 0.7692 (-0.0992) | 0.6345 (-0.2339) | 0.5648 (-0.3036) |
| Hammer | support | 0.7638 (-0.1046) | 0.6402 (-0.2282) | 0.5653 (-0.3031) |
| Mug | reference | 0.8929 (-0.0747) | 0.7801 (-0.1875) | 0.5177 (-0.4500) |
| Mug | support | 0.8971 (-0.0705) | 0.7837 (-0.1839) | 0.5271 (-0.4406) |
| Spoon | reference | 0.9438 (-0.0220) | 0.7686 (-0.1972) | 0.5585 (-0.4073) |
| Spoon | support | 0.9586 (-0.0072) | 0.7935 (-0.1723) | 0.5772 (-0.3886) |

三个类别等权 macro average：

| Normalization | Keep | mIoU | Delta | Label agreement | Cosine | Embedding L2 | Query drift |
|---|---:|---:|---:|---:|---:|---:|---:|
| reference | 75% | 0.8687 | -0.0653 | 0.9368 | 0.9912 | 0.0809 | 0.0000 |
| reference | 50% | 0.7277 | -0.2062 | 0.8394 | 0.9765 | 0.1591 | 0.0000 |
| reference | 25% | 0.5470 | -0.3870 | 0.7098 | 0.9626 | 0.2187 | 0.0000 |
| support | 75% | 0.8732 | -0.0608 | 0.9362 | 0.9915 | 0.0788 | 0.0198 |
| support | 50% | 0.7391 | -0.1948 | 0.8441 | 0.9772 | 0.1564 | 0.0237 |
| support | 25% | 0.5565 | -0.3774 | 0.7185 | 0.9640 | 0.2139 | 0.0718 |

这组 proxy 的方向性跨三个类别一致：support 只保留 normal-facing 25% 时，mIoU 均降至约 0.52--0.58。reference 与 support normalization 的结果接近，说明该条件下主要问题是几何/part coverage 缺失，而不是 normalization drift 单独造成的。

但它仍不能替代真实相机实验。正式 single-view 评测至少需要：指定相机位姿和内参、mesh/RGB-D z-buffer、self-occlusion、深度有效性、视锥裁剪，并固定或记录每个对象的 camera poses；真实数据还应加入 segmentation noise 和 depth artifacts。

## 7. 可用于论文和不可用于论文的表述

当前可以写成诊断结论：

- fixed-query readout 对普通 clean support 很准，但对 normal-facing 的低覆盖 support 明显不稳；
- AABB-shell contamination 的敏感度强烈依赖类别，Hammer 明显弱于 Mug/Spoon；
- support-dependent normalization/bounds 会放大 Hammer outlier contamination；
- 高平均 embedding cosine 不保证 part label 和 mIoU 稳定。

当前不能写：

- “模型已在真实 single-view camera observation 上验证”；
- “20% outlier 比 5% 更好”或“outlier 有正则化作用”；
- “结果具有统计显著性”；
- “Mug 完整三 part mIoU 为 0.9676”；
- “reference-normalization 完全消除了 support-dependent coordinate/bounds 变化”。

正式投稿前应在 independent test split 上复现，并将 normal-facing proxy 与 camera-rendered / real RGB-D partial observation 分开列项。若篇幅受限，proxy 可进入 robustness appendix，真实 camera single-view 进入主表。

## 8. 产物索引

- `outputs/paper_revision/r2_robustness_extension/hammer_seed20260806_t5_fp32/`
- `outputs/paper_revision/r2_robustness_extension/mug_seed20260806_t5_fp32/`
- `outputs/paper_revision/r2_robustness_extension/spoon_seed20260806_t5_fp32/`

每个目录包含：

- `results.json`：完整协议、split、checkpoint/config/Utonia hash、runtime、reference 与 condition 聚合；
- `trials.csv`：逐对象、逐 condition、逐 trial、逐 normalization 指标；
- `aggregate_conditions.csv`：pooled confusion 聚合；
- `reference_per_model.csv`：逐对象 clean reference。
