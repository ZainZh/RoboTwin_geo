# Dense task-aligned NDF functional frame: fold-0 result

Date: 2026-08-04

## Paper claim under test

Can camera-derived object geometry improve a learned short-horizon action-chunk
policy when the geometry is represented as a task-aligned, uncertainty-aware
relative functional frame rather than a globally pooled descriptor?

The evaluated architecture is:

```text
current A camera cloud
        -> task-aligned NDF two-vector head
        -> Gram--Schmidt SO(3) functional frame

cached early B camera cloud
        -> marker functional frame

A/B frames -> relative SE(3) [delta-p, R6D] + confidence
        -> direct Transformer scene/memory conditioning
        + A/B point tokens + robot proprioception
        -> six-step 7D active-arm action chunk
```

The interaction-flow prediction is an auxiliary training loss. There is no
analytic planner, simulator-state input, stage classifier, or recovery policy
in this experiment.

## Protocol

- Task: shoe manipulation trajectories before release.
- Data: 50 independent demonstrations, 3,245 camera-policy states.
- Fold 0 train objects: 2, 3, 4, 7, 8, 9.
- Validation objects: 1, 6.
- Test objects: 0, 5.
- Test set: 648 frames from ten independent episodes.
- NDF initialization: pretrained `shoe.pth`.
- Policy: 1.123 M-parameter interaction-functional direct-flow Transformer.
- Three independently trained NDF models and three independently trained policy
  seeds per condition.
- Policy controls are trained from scratch with identical architecture and
  parameter count: correct frame, exact zero, and progress-matched shuffled
  frame.

The object pose is an NDF supervision label and an evaluation diagnostic only.
It is not an NDF input or policy inference input. The deployable relative token
uses the live A cloud and cached B cloud.

## Why the earlier single-checkpoint attempt failed

The earlier full-frame dataset contained only 214 sparse correction states.
Three variants failed on held-out objects:

| variant | mean deg | median deg | p90 deg | >90 deg |
|---|---:|---:|---:|---:|
| joint full fine-tuning | 31.88 | 12.31 | 118.16 | 11.4% |
| freeze Z, learn beta/Y only | 44.63 | 22.16 | 116.66 | 22.7% |
| learn Y first, freeze it, learn Z | 42.29 | 22.87 | 92.78 | 13.6% |

The one-axis backbone became specialized to that axis, so a second 4k-parameter
linear coefficient head could not recover the missing yaw information. This is
an optimization/data-coverage failure, not evidence that a complete NDF frame
is unlearnable.

## Dense single-model frame result

Training the same joint two-vector head on all 3,245 object-disjoint trajectory
states changed the result qualitatively:

| NDF seed | mean deg | median deg | p90 deg | <=15 deg | >90 deg |
|---:|---:|---:|---:|---:|---:|
| 0 | 8.15 | 2.89 | 8.08 | 95.4% | 2.8% |
| 1 | 11.60 | 3.02 | 12.37 | 92.1% | 4.3% |
| 2 | 9.71 | 3.36 | 9.35 | 94.3% | 3.4% |
| shuffled-label control | 88.84 | 79.58 | 166.92 | 3.5% | 44.6% |

The previous two-checkpoint prototype was 14.89-degree mean, 9.87-degree
median, and 35.39-degree p90. The dense joint model therefore improves the
typical and long-tail non-flip prediction while requiring one NDF forward pass
per ensemble member rather than separate Z/Y networks.

The dense shuffled-label control uses the same model, data volume, optimizer,
and object split. Its near-random held-out frame proves that the correct result
comes from task/geometry alignment rather than a dataset-wide constant frame or
added network capacity.

## What NDF pretraining contributes

A matched three-seed ablation used the same 50 episodes and all 8,880 dense
camera observations. The encoder architecture, functional-frame supervision,
optimizer, and object-disjoint fold were held fixed.

| initialization / trainable scope | mean deg | median deg | p90 deg | >90 deg |
|---|---:|---:|---:|---:|
| frozen pretrained NDF, train heads | 74.01 | 58.08 | 167.25 | 32.27% |
| random initialization, full fine-tune | 4.40 | 2.44 | 6.10 | 0.44% |
| **pretrained NDF, full fine-tune** | **3.85** | **2.32** | **5.47** | **0.16%** |

The frozen representation fails in all three seeds, while both fully trained
models learn the task frame. Pretraining improves the three-seed mean, p90,
flip rate, and seed-to-seed stability over the identical randomly initialized
architecture, but it is not sufficient without task-level end-to-end
fine-tuning.

The completed K=1 comparison retained exactly one complete demonstration for
each of the six training objects while leaving validation and test objects
unchanged:

| K=1 initialization | mean deg | median deg | p90 deg | >90 deg |
|---|---:|---:|---:|---:|
| legacy NDF checkpoint, full fine-tune | 21.73 | 4.44 | 70.40 | 8.97% |
| NDF backbone, reinitialized axis heads | 20.98 | 4.63 | 86.42 | 8.71% |
| **random initialization, full fine-tune** | **15.05** | **3.88** | **35.56** | **5.22%** |

The legacy initialization is worse in all three K=1 seeds. Therefore the old
single-axis `shoe.pth` is not evidence of low-shot transfer to a complete
functional frame; its axis-specific output bias amplifies rare 180-degree
errors when supervision is scarce. Retaining the pretrained equivariant
backbone and vector basis while reinitializing both task-specific axis heads
helps one seed but hurts two; its three-seed mean remains 5.93 degrees worse
than random and its p90 is worse than both alternatives. The negative transfer
therefore is not explained by the old output head alone. This variant is not
promoted. The defensible role for `shoe.pth` is a full-data initialization that
must be end-to-end adapted, not a frozen or intrinsically low-shot descriptor.

## Epistemic uncertainty detects the remaining flips

The three SO(3) estimates are projected to a rotation mean. Confidence is the
maximum pairwise geodesic agreement. Its threshold, 5.79 degrees, is the 95th
percentile on training objects only.

- test acceptance: 537/648 frames, or 82.9%;
- accepted-frame mean: 3.16 degrees;
- accepted-frame median: 2.45 degrees;
- accepted-frame p90: 5.88 degrees;
- accepted-frame maximum: 20.13 degrees;
- accepted-frame >90-degree flips: 0;
- rejected-frame mean: 38.41 degrees;
- rejected-frame >90-degree rate: 19.8%;
- disagreement/error Pearson correlation: 0.598.

All 22 ensemble flips on test objects are in the rejected set. This supplies a
label-free deployment diagnostic rather than using simulator error. The policy
intervention below shows that detecting an ambiguous frame and hard-masking it
are separate questions: disagreement is useful, but hard p95 masking is not
currently supported as the best action-policy input.

## Fully camera-derived relative frame

The cached B marker estimator is no longer the rotation bottleneck:

- test goal translation error: 0.400 cm mean;
- test goal rotation error: 0.919 degrees mean, 1.553 degrees p90.

Before confidence masking, the complete camera-relative token has:

- translation error: 2.828 cm mean, 2.398 cm median, 4.854 cm p90;
- rotation error: 9.304 degrees mean, 2.842 degrees median, 8.992 degrees p90.

The high rotation mean is caused by the rejected discrete flips. The policy
receives an exact-zero projected frame feature for rejected observations.

## Initial fold-0 policy result: three seeds

All test frames, mean over three policy seeds:

| trained condition | action dt mm | action dr deg | endpoint t cm | endpoint r deg |
|---|---:|---:|---:|---:|
| **correct confidence-gated frame** | **9.130** | **1.913** | **2.781** | **9.268** |
| exact zero | 9.530 | 2.502 | 3.068 | 13.076 |
| matched shuffled frame | 9.621 | 2.613 | 3.004 | 13.651 |

The correct frame wins both endpoint metrics in 3/3 seeds against both
controls. The action-translation metric wins in 2/3 seeds; no universal claim
is made for that local metric.

## Initial fold-0 independent-episode statistics

Frame errors are first averaged within each independent episode. Bootstrap
resamples both the three policy seeds and the ten test episodes.

| comparison, lower is better | mean delta | 95% hierarchical bootstrap | seed wins | episode wins |
|---|---:|---:|---:|---:|
| endpoint rotation: correct - zero | **-3.755 deg** | **[-6.970, -1.239]** | 3/3 | 10/10 |
| endpoint rotation: correct - shuffled | **-4.288 deg** | **[-7.789, -1.384]** | 3/3 | 9/10 |
| action rotation: correct - zero | **-0.580 deg** | **[-1.095, -0.178]** | 3/3 | 9/10 |
| action rotation: correct - shuffled | **-0.687 deg** | **[-1.250, -0.212]** | 3/3 | 9/10 |
| endpoint translation: correct - zero | -0.287 cm | [-0.557, +0.019] | 3/3 | 9/10 |
| endpoint translation: correct - shuffled | -0.220 cm | [-0.588, +0.080] | 3/3 | 7/10 |

On this initial fold, the supported result was therefore a camera-only
rotation/action benefit; the translation interval still crossed zero. The
complete five-fold result below supersedes that limitation: with ten held-out
objects, all translation and rotation object-level intervals exclude zero.

## Confidence intervention: geometry matters, hard p95 masking does not

The same three clean policy checkpoints were re-evaluated with confidence
computed only from train-object disagreement percentiles. This is a paired
test-time intervention; it does not retrain the policy.

| frame feature at test | coverage | action dt mm | action dr deg | endpoint t cm | endpoint r deg |
|---|---:|---:|---:|---:|---:|
| **always on** | 100.0% | **9.063** | **1.826** | **2.740** | **8.690** |
| train p99 hard gate | 90.7% | 9.078 | 1.842 | 2.764 | 8.791 |
| train p97.5 hard gate | 87.5% | 9.117 | 1.876 | 2.770 | 9.011 |
| train p95 hard gate | 82.9% | 9.130 | 1.913 | 2.781 | 9.268 |
| train p90 hard gate | 72.1% | 9.207 | 2.014 | 2.829 | 9.938 |
| always off | 0.0% | 9.730 | 2.696 | 3.141 | 14.640 |

Episode/seed hierarchical bootstrap strongly favors always-on geometry over
always-off geometry in all four metrics: action translation +0.671 mm
`[+0.321,+1.097]`, action rotation +0.861 degrees `[+0.512,+1.232]`, endpoint
translation +0.404 cm `[+0.159,+0.687]`, and endpoint rotation +5.895 degrees
`[+3.580,+8.278]` for off minus on. Conversely, p95 minus always-on endpoint
rotation is +0.590 degrees with interval `[-0.184,+1.564]`; there is no evidence
that p95 hard masking helps the policy. P90 masking is significantly worse in
rotation, while p99 is statistically indistinguishable from always on.

The correct conclusion is therefore: keep the relative functional frame as a
policy condition; retain ensemble disagreement for monitoring and closed-loop
safety evaluation; do not make binary p95 gating a central algorithmic claim.

The subsequent from-scratch comparison (three policy seeds per variant) found
the same result. Episode-balanced means were:

| trained confidence variant | action dt mm | action dr deg | endpoint t cm | endpoint r deg |
|---|---:|---:|---:|---:|
| always on | 9.135 | 1.857 | 2.825 | 8.887 |
| train-p99 hard | **8.936** | **1.819** | **2.802** | 8.722 |
| continuous confidence | 9.423 | 1.853 | 2.952 | **8.680** |

None of the paired p99-minus-always-on intervals excluded zero: action
translation -0.198 mm `[-0.763,+0.249]`, action rotation -0.039 degrees
`[-0.138,+0.053]`, endpoint translation -0.023 cm `[-0.241,+0.157]`, and
endpoint rotation -0.165 degrees `[-0.765,+0.414]`. Continuous confidence was
also statistically indistinguishable and worsened translation numerically.
The fixed train-p99 rule is therefore retained as a conservative engineering
choice because it rejects only the most ambiguous frames; it is not claimed as
an accuracy contribution.

## Five-fold functional-frame generalization

One of three frame seeds failed catastrophically on fold 4. Averaging all three
models consequently made confidence unreliable. A deployable selection rule
now ranks the three candidates by validation-object p90, keeps the best two,
and calibrates disagreement on training objects only. Test metrics and test
poses are never read by selection or calibration.

| fold | validation-selected seeds | test median deg | test p90 deg | >90 deg | train-p99 test coverage |
|---:|:---:|---:|---:|---:|---:|
| 0 | 0, 1 | 2.77 | 9.76 | 2.93% | 91.82% |
| 1 | 2, 1 | 3.87 | 6.85 | 0.24% | 98.20% |
| 2 | 1, 0 | 2.64 | 5.24 | 0.84% | 95.36% |
| 3 | 0, 1 | 2.22 | 4.07 | 0.19% | 94.97% |
| 4 | 0, 2 | 3.17 | 8.24 | 2.62% | 87.64% |

This establishes good typical object-disjoint frame accuracy on every fold,
but also exposes the remaining tail-risk limitation: fold 4 still contains
rare discrete flips, and train-p99 disagreement does not reject every one.
Five-fold policy training therefore uses the same selected-top-two p99 input
and retains fold-specific tail metrics rather than reporting only a pooled
mean.

## Five-fold policy result

The complete policy comparison now covers all five folds, ten non-overlapping
test objects, fifty independent test episodes, and three optimization seeds.
Every model uses the same architecture, parameter count, training schedule,
point clouds, and proprioception. The only intervention is the relative
functional-frame token: correct camera-estimated geometry, all zeros, or a
wrong-object shuffled relation.

Object-balanced test means are:

| policy condition | action dt mm | action dr deg | endpoint t cm | endpoint r deg |
|---|---:|---:|---:|---:|
| **correct geometry** | **8.526** | **1.794** | **2.504** | **8.420** |
| zero relation | 9.093 | 2.397 | 2.784 | 12.509 |
| shuffled relation | 9.296 | 2.546 | 2.825 | 13.247 |

The primary uncertainty unit is the held-out object: optimization seeds and
episodes are averaged within each object, then the ten paired object effects
are bootstrapped. Correct-minus-control effects are:

| metric | correct - zero, object 95% CI | correct - shuffled, object 95% CI |
|---|---:|---:|
| action translation | -0.567 mm `[-0.785,-0.333]` | -0.769 mm `[-1.086,-0.442]` |
| action rotation | -0.603 deg `[-0.713,-0.475]` | -0.752 deg `[-0.945,-0.561]` |
| endpoint translation | -0.280 cm `[-0.358,-0.195]` | -0.321 cm `[-0.448,-0.199]` |
| endpoint rotation | -4.089 deg `[-4.929,-3.193]` | -4.827 deg `[-6.107,-3.709]` |

All eight primary intervals exclude zero. Correct geometry beats shuffled
geometry on all ten objects for all four metrics. Against zero, it wins on
9/10 objects for action translation and 10/10 for the other metrics; it also
wins all 15 fold-by-policy-seed rotation comparisons. This is the current
strongest evidence for the paper's central claim: the gain is caused by the
correct geometric relation, not merely an additional token or parameter
capacity.

## Coordinate, task-scope, and action-chunk contract audits

The policy coordinate-system audit rejects an initially plausible alternative:
expressing robot state and actions in the estimated goal frame.  With all task
phases retained, the original world-frame policy was better than full
goal-frame canonicalization.  Transforming only the geometry tokens was closer
but still did not improve over the original world-frame interface.  A strict
dual world/task-frame adapter initialized to exactly reproduce the world-frame
model also converged to an effectively tied result.  Therefore the retained
interface is deliberately asymmetric:

- robot proprioception and Cartesian actions remain in the stable robot/world
  coordinates;
- camera geometry supplies an explicit current-to-goal SE(3) relation token;
- the Transformer learns how that relation modulates the action chunk.

This is not a failure of geometric conditioning.  It shows that the task frame
is useful as *information* but introduces estimator noise when it is imposed as
the coordinate system for every robot variable.

The original policy was trained on both grasp/pre-relation and placement states,
even though online evaluation starts after expert grasp at the placement phase.
A matched task-scope audit now filters the demonstrations to relation/placement
rows only.  This filter is a training/deployment contract, not a stage label:
the deployed model has no stage input, stage head, or stage classifier.  The
result contains 2,796 training samples and 796 held-out frames from eight
independent test episodes.

Episode-balanced three-policy-seed results are:

| placement-only condition | action dt mm | action dr deg | endpoint t cm | endpoint r deg |
|---|---:|---:|---:|---:|
| **correct camera geometry** | **2.540** | **0.670** | **1.410** | **3.804** |
| exact zero relation | 3.250 | 1.123 | 1.798 | 6.611 |
| matched shuffled relation | 3.391 | 1.161 | 1.891 | 6.837 |

Correct-minus-zero effects are -0.711 mm `[-1.100,-0.322]`, -0.453 degrees
`[-0.692,-0.218]`, -0.388 cm `[-0.621,-0.146]`, and -2.807 degrees
`[-4.219,-1.406]`.  Correct-minus-shuffled effects are -0.851 mm
`[-1.210,-0.514]`, -0.491 degrees `[-0.760,-0.221]`, -0.480 cm
`[-0.696,-0.272]`, and -3.033 degrees `[-4.662,-1.400]`.  All eight
hierarchical intervals exclude zero, and correct geometry wins all four metrics
in 3/3 policy seeds against both controls.

This all-relation aggregate includes states at and after the demonstrated
release, which are easier than the states that determine online placement.
Restricting evaluation of the same checkpoints to pre-release states gives
3.735 mm / 0.976 degrees action error and 2.067 cm / 5.524 degrees endpoint
error for correct geometry.  The correct-minus-zero and
correct-minus-shuffled hierarchical intervals still exclude zero in all four
metrics, so the geometric effect survives the harder audit, but the original
absolute error was optimistic.

A follow-up trained three new correct-geometry policy seeds only on the 1,781
fold-0 pre-release training rows, while leaving the six-step chunks free to
contain the release transition.  On the identical pre-release test subset this
was uniformly worse: 3.906 mm / 1.014 degrees and 2.172 cm / 5.788 degrees.
Thus simply deleting post-release rows loses useful trajectory data and is not
promoted.  The remaining online issue should be addressed by better coverage
or weighting of terminal correction states, not by calling this filtering
heuristic a contribution.

Finally, online inference must honor the learned six-step action-chunk
contract.  On the same policy and simulator seed, replanning after every first
action of a predicted chunk failed after 240 actions, while executing the full
six-step chunk succeeded in 52 actions.  The model is still a learned
short-horizon action policy--there is no analytic trajectory or inverse-
kinematics planner--but truncating every training target to its first action
creates a severe train/deploy mismatch.

## Real pose-correction adapter: object-disjoint three-seed result

The nominal placement data still contains few large-error states.  A new
single-stage dataset therefore records only grasp-preserving return motions
from an SE(3)-perturbed, gripper-closed shoe to its final pre-release pose.  The
scripted controller generates demonstrations only; it is not part of policy
inference.  The object split and data roles are strict:

- 24 training episodes: six train shoes x four perturbation levels;
- 8 validation episodes: shoes 1 and 6, used for checkpoint selection;
- 8 untouched test episodes: shoes 0 and 5, used once after selection;
- 131/42/40 recovery action states in train/validation/test respectively.

The nominal policy is frozen and a zero-initialized second action decoder is
trained as a geometry-conditioned correction adapter.  Its output is
continuously gated by current-to-goal translation/rotation magnitude and by
camera geometry confidence.  Nominal rows distill the adapter residual to
exactly zero, and checkpoint selection enforces at most 2% nominal-validation
degradation.  This is a learned action policy adapter, not an analytic residual
controller or stage predictor.

Episode-balanced means over three independently initialized nominal policies
and adapters on the eight untouched recovery episodes are:

| held-out recovery | action dt mm | action dr deg | endpoint t cm | endpoint r deg |
|---|---:|---:|---:|---:|
| frozen nominal policy | 5.490 | 2.931 | 2.748 | 15.991 |
| **geometry correction adapter** | **4.598** | **2.762** | **2.041** | **13.699** |
| relative improvement | **16.2%** | **5.8%** | **25.7%** | **14.3%** |

All three policy seeds improve all four recovery metrics.  Within each seed,
the paired eight-episode bootstrap interval excludes zero for endpoint
translation and endpoint rotation.  It also excludes zero for action
translation in all three seeds and action rotation in two of three (the third
lower bound is -0.0002 degrees).

The same checkpoints remain effectively unchanged on the 796 nominal held-out
states.  Three-seed means change from 2.538 to 2.540 mm, 0.669 to 0.673 degrees,
1.410 to 1.411 cm, and 3.798 to 3.816 degrees.  These are all below 1%; paired
nominal intervals generally include zero.  Seed 1 has a tiny statistically
detectable action-rotation degradation of 0.012 degrees, which is too small to
be practically material but is reported rather than hidden.

The intervention controls support geometric causality.  Exact-zero geometry
keeps the zero-start adapter identical to the nominal policy.  For a fixed
correct-trained policy, zeroing the token worsens all four recovery metrics;
shuffling it particularly worsens endpoint rotation.  High-confidence test
episodes show the largest gains, while low-confidence NDF failures are masked
instead of driving a correction.

This is currently strong evidence for offline action accuracy, not yet a
closed-loop recovery-success claim.  The paired camera-only simulator protocol
below is being extended from nominal placement starts to identical
grasp-preserving perturbations so that EEF-label improvements and executed
object-pose improvements are measured separately.

## First camera-only closed-loop contract result

The full online path was run with the same simulator seed 300001 for clean and
zero geometry. The clean camera/NDF policy succeeded in 30 control steps; the
otherwise identical zero-geometry intervention failed after the 240-step
policy limit. No simulator object pose or privileged oracle was enabled.

This single pair proves that the camera-to-functional-frame-to-action runtime
is executable and that the learned policy can use its geometry online. It is
not yet a success-rate estimate.

The legacy three-model p95 policy was then evaluated on ten fixed seeds. Clean
geometry succeeded on 3/10 and the test-time zero intervention on 0/10; the
paired gain was +30 percentage points with bootstrap interval `[0,+60]` and an
exact paired sign p-value of 0.25. This is a useful diagnostic but not a
statistically supported success-rate claim. Two failures were caused by the
overly strict p95 mask; other confident failures exposed policy covariate shift.

The final validation-selected-top-two p99 runtime was tested against a
separately trained, capacity-matched zero policy on seed 300001. Correct geometry
succeeded in 42 steps at 2.04 cm / 6.70 degrees; the zero-trained policy failed
at the 240-step limit at 18.47 cm / 38.65 degrees. A 10-seed-by-3-policy-seed
paired evaluation is running to measure both scene and optimization variance.
The first complete policy seed yields 2/10 successes with correct geometry and
0/10 for its independently trained zero control: +20 percentage points,
bootstrap interval `[0,+50]`, exact paired sign p=0.5. This is directionally
positive but not significant, and it confirms that closed-loop covariate shift
remains after the strong five-fold offline result. The other two policy seeds,
temporal action ensembling, and an oracle-geometry upper bound remain queued.

## Corrected recovery labels and closed-loop contact diagnosis

Every pose-correction recording begins with one static duplicate EEF frame.
The first recovery dataset therefore assigned a zero first action to the exact
state used at deployment.  `build_task_flow_dataset.py` now has an explicit
`--collapse-static-eef-frames` path that removes only duplicate EEF frames; it
does not interpolate or synthesize labels.  All 40 corrected recovery episodes
now have non-zero initial actions.

A seed-2 adapter retrained on the corrected 24/8/8 split improves the eight
untouched initial correction states over its frozen nominal policy by 2.964 mm
action translation, 0.582 degrees action rotation, 1.969 cm endpoint
translation, and 6.756 degrees endpoint rotation.  Episode-bootstrap 95%
intervals exclude zero for all four metrics.  Over all 33 test states, three of
four intervals exclude zero; action rotation is borderline
`[-0.003,+0.486]` degrees.  Nominal differences remain near zero with all
intervals including zero.

Closed-loop execution is more demanding.  On two exactly matched moderate
perturbations, the corrected adapter improves final translation by 0.72 cm and
rotation by 1.46 degrees on average, but neither method reaches the success
threshold.  On a 22.60 cm / 65.85 degree hard perturbation, both policies
diverge and the adapter is worse when six actions are executed open loop.
Re-observing after every action initially helps the adapter: the first two
actions reduce the hard state to 21.26 cm / 60.45 degrees.  The third action
then causes a discontinuous object jump.

Per-action instrumentation localizes that jump.  The NDF estimate before the
third action is 23.17 cm / 60.72 degrees with full confidence, matching the
simulator metric, so this is not a pre-action encoder failure.  At the same
transition, object pose relative to the active gripper changes by 1.85 cm and
12.13 degrees.  Across 140 expert correction transitions, the corresponding
95th percentiles are only 0.033 cm and 0.258 degrees.  The failed learned
transition is therefore roughly 56x/47x those expert contact-drift levels.

This exposes a missing relation in the policy interface: object-to-goal
geometry states where the object should go, but not how the currently closed
hand is attached to it.  A camera-deployable active-hand-to-NDF-functional-frame
SE(3) token has been added to the learned recovery decoder.  With only 24
training grasps, neither joint fine-tuning nor projection-only training beats
the corrected adapter's validation checkpoint; the best model remains the
zero-start initialization.  This is a useful negative result, not evidence
against the interaction relation itself: the training split has only four
grasp instances per shoe, while the failure specifically depends on grasp-axis
compatibility.  The next experiment expands distinct training grasps before
retesting the same frozen architecture and paired closed-loop protocol.

## Current paper story

The main contribution should be stated as:

> A task-aligned equivariant functional frame converts object geometry into an
> explicit relative SE(3) token for action-chunk policies. This structured
> interface improves object-disjoint manipulation action and endpoint accuracy over
> capacity-matched zero and wrong-geometry controls, while ensemble
> disagreement exposes discrete geometric ambiguity as a deployment
> diagnostic.

This is different from simply concatenating NDF/UTONIA/DINO features:

1. geometry is supervised into a complete manipulation-relevant SO(3) frame;
2. the policy receives the current-to-goal relation, not an arbitrary pooled
   latent;
3. ensemble disagreement has a measurable calibration target and exposes
   harmful flips without simulator state;
4. geometry conditions a learned action-chunk policy directly; it does not
   replace policy learning with planning.

## What is supported and what remains

Supported on fold 0:

1. dense task-aligned NDF learns a complete functional frame on unseen objects;
2. training-object-calibrated disagreement detects all held-out flip failures;
3. correct geometry causally improves rotation over both zero and shuffled
   controls across three policy seeds and ten independent episodes.
4. removing geometry at test time degrades all four action/endpoint metrics;
   increasingly aggressive hard confidence masks monotonically lose benefit.
5. one strictly paired camera-only rollout succeeds with geometry and fails
   under an exact-zero test-time intervention.

Supported across all five frame folds:

1. validation-only model selection prevents a single failed seed from
   contaminating the ensemble;
2. typical held-out functional-frame errors remain below 4 degrees median and
   10 degrees p90 on every fold;
3. correct geometry improves all four action/endpoint errors over both zero and
   shuffled controls with object-level intervals excluding zero across ten
   non-overlapping held-out objects;
4. all 15 fold-by-policy-seed rotation comparisons favor correct geometry.

Not yet supported:

1. statistically powered closed-loop task-success improvement;
2. generality beyond this task family;
3. a single-model uncertainty estimate with the same flip recall as the
   validation-selected two-member ensemble.

## Next T-ASE gates

1. Repeat frame training and the three policy controls on all five object folds.
2. Run paired closed-loop rollouts with identical initial perturbations for raw,
   zero, shuffled, single-model ungated, ensemble-gated, and oracle frames.
3. Report task success, final object translation/SO(3), collision/drop rate,
   confidence coverage, accepted risk, and calibration curves.
4. Add one additional short-horizon object-relation task and one alternate
   encoder as a representation ablation without changing the SE(3)-token
   interface.
5. If three-model runtime is excessive, distill ensemble confidence into a
   single uncertainty head only after the five-fold result is reproduced.

## Reproducibility artifacts

- `train_ndf_functional_frame_head.py`: joint two-vector NDF head.
- `ensemble_ndf_functional_frames.py`: SO(3) ensemble and train-only confidence.
- `select_ndf_functional_frame_ensemble.py`: validation-only top-two model
  selection and train-only percentile calibration.
- `augment_task_flow_with_ndf_full_frame.py`: NDF frame policy bridge.
- `augment_task_flow_with_camera_ndf_frame.py`: camera-relative SE(3) and
  confidence composition.
- `run_dense_functional_frame_ablation.sh`: exact three-seed control protocol.
- `evaluate_dense_functional_frame_ablation.py`: episode/seed hierarchical
  evaluation.
- `online_ndf_functional_frame.py` and `interaction_flow_runtime.py`:
  camera-only online functional-frame inference.
- `run_dense_ndf_closedloop_pilot.sh`: strictly paired clean/zero rollout
  runner for one seed or a fixed-seed shard.

Remote result roots:

- `outputs/geometry_flow/remote4090/ndf_functional_frame_dense_fold0*`
- `outputs/geometry_flow/remote4090/dense_single_ndf_policy_seed*`
- `outputs/geometry_flow/remote4090/dense_single_ndf_policy_ablation_episode_stats.json`
