# GeometryFlow: Functional Geometry as a Verified Skill Interface for Learned Robot Manipulation

Working draft, 2026-08-01.  The quantitative statements below use only the
completed frozen fold-0 and fold-3 evaluations.  Results from the running GPU-2
replication are intentionally excluded until its protocol is complete.

## Abstract

Object-centric 3D representations can improve the spatial awareness of learned
robot policies, yet they are commonly injected as latent conditioning and leave
the policy to discover how geometry should affect continuous motion and discrete
skill transitions.  We argue that this interface, rather than representation
capacity alone, is a central source of instability in geometry-enhanced
manipulation.  We present **GeometryFlow**, a functional-geometry interface that
couples an existing learned visuomotor policy with explicit, verifiable object
motion.  From camera point clouds, GeometryFlow selects a confidence-scored
object-in-gripper SE(3) relation and converts the desired object displacement into
an exact rigid action flow.  The learned policy produces the nominal skill
trajectory, while the geometric interface corrects terminal motion and owns the
release transition only after a temporally persistent translation-and-rotation
test is satisfied.  This division preserves the multimodal behavior learned from
demonstrations while assigning geometrically identifiable decisions to an
interpretable closed-loop module.  On two object-disjoint folds of the post-grasp
placement phase of a bimanual shoe task, the frozen method succeeds in 96/100 paired trials, compared with
82/100 for the matched ablation without the verified terminal transition, giving
a 14 percentage-point gain
(bootstrap 95% confidence interval: 7--22 points; exact paired
\(p=5.19\times10^{-4}\)).  It also reduces mean completion time from 81.51 to
46.46 control steps.  This comparison shares the nominal policy, relation bank,
verifier, and analytic recovery between conditions and isolates the bilateral
geometry-owned transition; it therefore provides causal evidence for that
component rather than attributing all 14 points to the full stack.  Failure
analysis shows that the dominant remaining error is
post-grasp violation of the assumed rigid object-in-gripper relation, motivating
live pointwise visual verification rather than further global feature pooling.
These results indicate that geometric representations become most effective when
they serve as an executable and uncertainty-aware skill interface, rather than as
an unconstrained auxiliary token.

## 中文摘要（讨论版）

物体中心的三维表征可以增强学习式机器人策略的空间感知能力，但现有方法通常将
几何信息作为隐式条件输入，仍要求策略自行学习几何如何影响连续动作以及何时推进
技能阶段。我们认为，造成几何增强不稳定的关键不仅是表征能力，还包括表征到动作
之间的接口。为此，我们提出 **GeometryFlow**：一种连接学习式视觉运动策略与
显式、可验证物体运动的功能几何接口。GeometryFlow 从相机点云中选择带置信度的
object-in-gripper SE(3) 关系，将期望物体位移转换为精确刚体 action flow。学习
策略生成名义技能轨迹，几何接口负责终端运动修正，并且只有在平移和旋转误差持续
满足条件后才允许释放。在双臂鞋子放置任务的两个物体不相交测试折上，带双边几何
抓取后 placement 阶段的 100 个严格配对试验中成功 96 次，关闭该终态转换、但保留相同
名义策略、关系库、verifier 和解析恢复的对照成功 82 次，绝对提升 14 个百分点
（bootstrap 95% 置信区间为 7--22 个百分点，精确配对检验
\(p=5.19\times10^{-4}\)），平均完成步数从 81.51 降至 46.46。该对照严格证明
的是双边几何终态转换的因果作用，而完整 GeometryFlow 相对 nominal policy 的
效果仍由正在进行的三阶段消融验证。失败分析显示，当前主要残余误差来自抓取后
刚性 object-in-gripper 假设被滑移或初始偏差破坏，因此下一步应使用逐点视觉
特征在线验证和校正该关系，而不是继续进行全局特征池化。

## Note to Practitioners

Learned manipulation policies often fail near the end of an otherwise successful
operation: the robot reaches the target but releases too early, keeps holding the
object, or repeatedly corrects an already acceptable pose.  GeometryFlow is
intended as a supervisory interface for such cases.  It leaves the learned policy
responsible for nominal motion, but converts an object-to-target pose error into a
bounded rigid correction and prevents release until the error is persistently
within specified translation and rotation tolerances.  This separation lets an
engineer inspect the selected object relation, correction magnitude, confidence,
and exact release condition instead of treating the full policy as a black box.
In the post-grasp placement phase of our simulated shoe task, this verified
transition improves success and reduces execution steps.  Deployment currently
requires segmented RGB-D point clouds, calibrated end-effector poses, and a
demonstration relation library.  The present implementation assumes that the
object remains rigidly fixed in the gripper after initialization; practical use
with deformable grasps or contact-induced slip will require the live visual
relation correction described in our next-stage design.  The same interface may
be useful in kitting, fixture loading, insertion, and other structured automation
tasks where terminal pose and release reliability dominate throughput.

## 1. Introduction

Many manipulation tasks are decided by small geometric errors.  A policy may
reach the correct object, execute a plausible transport motion, and make contact
with the target, yet fail because the grasped object is a few centimeters or
degrees away from its functional pose, or because the gripper opens at the wrong
time.  Such errors are particularly common under object-level generalization:
appearance and scale vary, the same semantic part can occupy a different metric
location, and the relationship between an end effector, an operated object, and a
target must remain consistent throughout the interaction.  Successful control
therefore requires more than recognizing geometry.  It requires a policy to
translate object geometry into motion, assess whether the intended relation has
actually been achieved, and use that assessment to advance the skill state.

Diffusion-based visuomotor policies provide a powerful model of multimodal action
distributions and smooth action chunks \cite{chi2024diffusionpolicy}.  Point-cloud
policies such as DP3 further improve spatial reasoning by conditioning action
generation on compact 3D observations \cite{ze2024dp3}.  These systems establish
that 3D input is useful, but they generally place the entire burden of converting
perception into continuous control and phase transitions on a learned action
decoder.  In limited-data manipulation, this burden is substantial: the model
must simultaneously learn category correspondence, rigid transformation algebra,
temporal completion criteria, and the consequences of opening or closing the
gripper.

A growing body of work develops stronger object-centric representations.  Neural
Descriptor Fields (NDFs) encode category-level, SE(3)-equivariant relations and
recover manipulation poses through descriptor matching
\cite{simeonov2021ndf}.  G3Flow maintains an object-centric 3D semantic flow by
combining generative object models, foundation-model features, and pose tracking
\cite{chen2024g3flow}.  Any3D-VLA fuses diverse point clouds with 2D observations
to improve robustness across point-cloud sources and domains
\cite{fan2026any3dvla}.  HeRO constructs global and local 3D semantic fields for
pose-aware diffusion policies \cite{xu2026hero}, while Action--Geometry
Prediction jointly predicts action chunks and future 3D latent geometry
\cite{xu2026gap}.  The PA3FF/PADP framework learns a dense, part-aware 3D field
and conditions a diffusion policy on functional-part structure
\cite{chen2026pa3ff}.  Object-centric neural fields have also been connected to a
temporal mixture of movement primitives for compositional motion generation
\cite{tekden2026compositional}.  These approaches demonstrate the value of richer
geometry and semantic correspondence.  However, a representation supplied as a
feature condition does not by itself specify which geometric quantities should
be preserved by an action, when a geometric prediction is trustworthy, or which
skill transitions it should be allowed to trigger.  GeometryFlow is complementary:
it can consume such learned fields, but exposes their output through a rigid,
confidence-scored control and transition interface.

Our controlled development experiments expose this distinction.  Replacing raw
object geometry with globally pooled frozen descriptors produced unstable gains,
and pointwise NDF or UTONIA features did not consistently outperform raw XYZ
under generic token fusion.  Moreover, shuffling descriptors across points often
had little causal effect, indicating that the policy was not reliably using the
intended local correspondence.  These observations do not imply that learned
geometric representations are intrinsically ineffective.  Instead, they suggest
an **interface problem**: a generic action network can ignore, shortcut, or
misinterpret a representation whose operational role is unspecified.

We address this problem with **GeometryFlow**, which treats functional geometry
as a verified interface between perception and a learned skill.  Let
\(\mathcal{P}_A\) denote the point cloud of the operated object and
\(\mathcal{P}_B\) that of the target.  The interface first selects a
confidence-scored object-in-gripper relation and represents the desired object
motion by \(T=(R,t)\in\mathrm{SE}(3)\).  Rather than compressing this motion into
an unconstrained global token, it induces the rigid pointwise action flow

\[
    f_i = t + (R-I)(x_i-p_A), \qquad x_i\in\mathcal{P}_A,
\]

where \(p_A\) is the current object origin and \(t\) is its desired world-frame
displacement.  This field retains the translational and rotational structure of the desired
object motion and can be mapped analytically to an end-effector correction under
the grasp relation.  A learned policy continues to generate the nominal action
trajectory; GeometryFlow intervenes only through a confidence-gated correction
and a small skill-state machine.  Before release, it preserves Cartesian control
while keeping the active gripper closed.  Release is latched only when an
exponentially smoothed geometric error remains below fixed translation and
rotation thresholds for multiple consecutive observations.  After the latch, the
Cartesian command is held while the gripper opens.  Geometry is consequently not
an alternative to policy learning or a stand-alone planner.  It is an executable
contract that assigns known rigid-body structure and safety-critical phase changes
to an auditable module, while retaining learned behavior for the rest of the
skill.

We evaluate the bilateral transition using strict, seed-matched closed-loop
comparisons on held-out shoe instances, starting after an expert-admitted grasp
and evaluating the placement phase rather than full grasp-to-place success.  The policy checkpoint, relation bank,
verifier, analytic recovery, simulator seeds, observations, and task configuration
are shared between candidate and baseline; the only intervention is whether the
frozen geometric module owns both sides of terminal gripper release.  On fold 0,
this transition improves success from 39/50 to 48/50.  A separately frozen fold-3
replication improves success from 43/50 to 48/50.  Pooling the pre-specified
paired trials gives 96/100 versus 82/100, with 15 transition-only successes and
one ablation-only success.  Qualitative and continuous diagnostics show that many
ablation failures reach the target region but remain closed or stall, whereas
persistent geometric verification converts these near-successes into completed
placements.  A separate, smaller 10-seed pilot compares the camera/analytic route
against nominal interaction diffusion (6/10 versus 0/10); we treat it only as an
expansion gate and require the paper-scale nominal-only ablation now being run
before making a full-stack effect claim.

The same analysis also reveals a precise limitation.  The current system is
camera-initialized, but after grasp it propagates the selected object-in-gripper
relation using end-effector proprioception and therefore assumes a rigid grasp.
Object slip or an initially biased relation can make internal alignment disagree
with the actual point cloud.  We do not hide this failure behind a broader
end-to-end perception claim.  Instead, it defines the next representation
problem: learned pointwise descriptors such as NDF, UTONIA, or semantic foundation
features should verify and correct the live object-in-gripper transform, with
calibrated uncertainty, rather than merely append another global conditioning
vector.

The main contributions of this work are:

1. **A functional-geometry formulation.** We formulate geometry as an explicit
   interface between a learned visuomotor policy and object-centric rigid motion,
   separating nominal skill generation from geometrically identifiable control
   responsibilities.
2. **A verifiable action-flow architecture.** We convert relative SE(3) hypotheses
   into exact rigid pointwise action flow and use persistent geometric consistency
   to govern terminal correction and gripper release.
3. **Causal, paired evidence for the terminal interface.** On two held-out object
   folds, the bilateral transition yields a 14-point absolute improvement over an
   otherwise identical analytic-route ablation and substantially reduces
   completion steps; phase-level failure analysis identifies geometry-owned
   release as the observed mechanism.  We keep this claim separate from the
   ongoing full-stack versus nominal-policy ablation.
4. **A representation diagnosis and concrete boundary.** We show why generic
   global or pointwise feature fusion can fail to exploit correspondence and
   identify live visual correction of the object-in-gripper transform as the
   remaining bottleneck, yielding a testable route for NDF/UTONIA-style features.

## Plain-language route summary (Chinese)

这条路线不是“几何网络直接生成整段机器人动作”，也不是“把 NDF token
拼给 DP3 后期待它自己学会”。它把任务拆成两类责任：

- 学习策略负责从演示中学到抓取、搬运、接近目标等复杂而多模态的名义行为；
- 几何接口负责可以明确计算和验证的部分，包括物体需要的 SE(3) 变化、该变化
  对夹爪动作的约束，以及什么时候真的满足条件可以松爪。

因此它仍然是学习方法，但加入了一个可解释、可验证的几何技能接口。当前正式
100 对结果严格证明的是其中“双边终态转换”相对相同 analytic route 的增益；
完整 GeometryFlow 相对 nominal policy 的大规模消融仍需补齐。下一步还要用实时
点云修正抓取后的物体—夹爪关系，并在多任务和真实机器人上验证其普适性。

## Citation placeholders

- `chi2024diffusionpolicy`: Chi et al., *Diffusion Policy: Visuomotor Policy
  Learning via Action Diffusion*, IJRR 2024, arXiv:2303.04137.
- `ze2024dp3`: Ze et al., *3D Diffusion Policy: Generalizable Visuomotor Policy
  Learning via Simple 3D Representations*, RSS 2024, arXiv:2403.03954.
- `simeonov2021ndf`: Simeonov et al., *Neural Descriptor Fields:
  SE(3)-Equivariant Object Representations for Manipulation*, arXiv:2112.05124.
- `chen2024g3flow`: Chen et al., *G3Flow: Generative 3D Semantic Flow for
  Pose-aware and Generalizable Object Manipulation*, arXiv:2411.18369.
- `fan2026any3dvla`: Fan et al., *Any3D-VLA: Enhancing VLA Robustness via
  Diverse Point Clouds*, arXiv:2602.00807.
- `xu2026hero`: Xu et al., *HeRO: Hierarchical 3D Semantic Representation for
  Pose-aware Object Manipulation*, arXiv:2602.18817.
- `xu2026gap`: Xu et al., *Action--Geometry Prediction with 3D Geometric Prior
  for Bimanual Manipulation*, arXiv:2602.23814.
- `chen2026pa3ff`: Chen et al., *Learning Part-Aware Dense 3D Feature Field for
  Generalizable Articulated Object Manipulation*, ICLR 2026,
  arXiv:2602.14193 (the repository's `reference/1.pdf`).
- `tekden2026compositional`: Tekden and Bekiroglu, *Compositional Motion
  Generation from Demonstration with Object-Centric Neural Fields*, IEEE RA-L
  2026, arXiv:2607.07129.
