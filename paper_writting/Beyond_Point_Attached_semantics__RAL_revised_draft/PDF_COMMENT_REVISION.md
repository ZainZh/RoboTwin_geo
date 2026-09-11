# PDF 批注修改记录

批注来源：`0_base_and_abstract_comment.pdf`。本轮仅修改 revised draft，保留批注 PDF 和原稿。

| 批注位置 | 本轮处理 |
| --- | --- |
| 第 1 页：PartNext 是否放到方法中 | Introduction 不再介绍具体训练数据集；方法中保留数据来源。Related Work 中仍可作为相关数据集引用。 |
| 第 3 页：没有显式建模噪声参数 | 删除观测生成函数及噪声参数公式，改为文字说明观测变化，并明确不估计这些因素。 |
| 第 3 页：adapter 的作用 | 说明其用于通道投影和冻结编码器后的部件语义适配；保留实际模块，不把设计目的写成已验证的独立贡献。若要证明必要性，仍需额外消融。 |
| 第 3 页：f、h 的关系 | f_theta 表示完整场，h_psi 表示其查询解码器；theta 包括 adapter、局部融合和解码器的可训练参数。 |
| 第 4 页：分类损失后的无依据表述 | 删除防止塌缩及“分类不足以组织嵌入”的判断，改为分类监督与嵌入对齐的互补目的。 |
| 第 5 页：没有 cross-view 实验 | 实验问题改为三个损失和查询采样；一致性模块改称 paired-augmentation consistency。保留实际的双增强训练，不声称进行了跨视角评测。 |
| 第 5 页：RoboTwin 数据划分 | 按批注明确使用标准划分，移除该 TBD，不再暗示重新构造跨实例模拟测试划分。 |
| 第 5 页：图 3、4 互换 | 已调整源文件中的浮动体顺序，先仿真任务，后真实平台。 |
| 第 5 页：Training Set / Test Set | 建议图内改为 Demonstration Objects / Held-out Evaluation Objects，突出对象实例而非整个数据集。当前位图暂保留，图注已解释对应关系。 |
| 第 5 页：G3Flow 实现说明 | 根据批注明确遵循官方实现，部署和跟踪细节移至 Implementation；说明它不是 DINOv2 Lifting 的同义项。 |
| 第 6 页：补 G3Flow 结果段 | 已添加比较段落，保留平均成功率和结果分析的 TBD；不能在结果未知时填写优劣结论。 |
| 第 3、8 页：段末孤词 | 重写对应句子，并通过重新编译检查断行；不改字号或页边距。 |

## 仍需确认

- 图内标签尚未修改，建议在绘图源文件中采用上述名称后重新导出。
- G3Flow 表格、均值和结果趋势仍待实测结果。批注中的 foundation model tracking 按官方 G3Flow 的 FoundationPose 物体位姿跟踪理解，不写成只发生在策略训练过程。
- 原有策略训练 seed 数、优化器、学习率、扩散步数和 rollout 上限仍待填写。
- 有些高亮没有评论文字（第 1 页训练扰动句、第 2 页 related work 句等），未视为删除指令；空的 FreeText 也没有可读取的文字。

核对来源：[G3Flow 官方实现](https://github.com/TianxingChen/G3Flow)、[G3Flow 论文](https://tianxingchen.github.io/G3Flow/files/G3Flow.pdf)。
