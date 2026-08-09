# Handled-mug two-level NDF policy result (2026-08-09)

## Claim under test

A short-horizon learned manipulation policy can use category-specific geometry
more reliably when geometry is represented at two mutually consistent levels:

1. per-point coordinates in the current and goal functional frames; and
2. an explicit current-to-goal SE(3) relation token.

This is a Transformer action-chunk policy.  It contains no analytic planner,
IK action prior, residual controller, task-stage head, or grasp token.

## Frozen benchmark

- Task: `place_handled_mug_geometry_marker`
- Assets: `039_mug` object IDs 0--9
- Split: train 0--5, validation 6--7, blind test 8--9
- Accepted demonstrations: 60 (6 per object), 64 attempts, 93.75% admission
- Perturbation levels: `{0:20, 1:20, 2:10, 3:10}`
- Policy rows: 219; relation-phase blind-test rows: 40
- Inputs at deployment: A/B camera point clouds and robot proprioception
- Seeds: 0, 1, 2

The camera T-marker goal estimator has blind-test absolute pose error
`0.120 cm / 0.523 deg`, so target perception is not the current bottleneck.

## Leakage-free NDF protocol

The task-specific two-axis NDF head was trained from random initialization on
the handled-mug category.  The shoe checkpoint was used only to instantiate
the architecture; shoe weights were not transferred.

For policy-training rows, each object's frame is predicted by a three-seed NDF
ensemble that excluded that object during training (leave-one-object-out
cross-fitting).  Validation and blind-test rows use the final ensemble trained
on all six training objects.  This removes the previous in-sample token shift:

- accepted cross-fitted policy-train frame error: mean 7.210 deg, p90 12.481 deg
- blind-test frame error: mean 5.328 deg, p90 9.321 deg
- blind-test relative relation error: 0.772 cm / 5.505 deg
- two cross-fit training rows were rejected by ensemble disagreement

SO(3) medoid aggregation is used because a single ensemble member flipped on
rare cross-fit frames.  The rejection threshold is calibrated only on
cross-fitted training-object disagreement.  Simulator poses are label-only for
training and diagnostics; the generated deployment relation is camera-only.

## Oracle architecture ceiling

Mean blind-test action metrics are `[step dt mm, step dr deg, endpoint cm,
endpoint deg]`:

| Input | Mean metrics |
|---|---|
| raw point policy | `[2.467, 2.133, 0.870, 7.401]` |
| correct oracle relation | `[2.500, 1.833, 0.713, 5.404]` |
| zero relation | `[2.669, 2.184, 1.086, 7.844]` |
| shuffled relation | `[2.292, 1.995, 0.753, 7.780]` |

The correct oracle relation wins both endpoint metrics in all three seeds
against raw, zero, and shuffled controls.  Thus the action-chunk policy can use
an accurate task relation; the architecture ceiling is positive.

## Falsified fusion variants

### Direct global relation fusion

The camera/NDF relation directly added to the policy produced
`[3.181, 2.257, 1.239, 8.405]`, losing the raw baseline on all four metrics in
all three seeds.  A useful NDF frame estimator does not make unregularized token
concatenation useful.

### Scalar-gated global relation fusion

Gating prevented the catastrophic regression but did not establish causality:

| Input | Mean metrics |
|---|---|
| correct | `[2.276, 1.887, 0.816, 7.198]` |
| zero | `[2.603, 2.107, 0.959, 7.449]` |
| shuffled | `[2.703, 2.335, 0.965, 7.732]` |

Correct endpoint translation won only 1/3 seeds against zero.  The apparent
mean gain was not stable enough to claim.

Cross-fitted NDF training tokens did not repair global-only fusion.  Correct
gave `[2.554, 2.074, 0.953, 7.743]`, shuffled gave
`[2.295, 1.876, 0.840, 7.254]`, and zero gave
`[2.603, 2.107, 0.959, 7.449]`.  This falsifies the explanation that the only
problem was in-sample NDF leakage.

## Two-level task-aligned representation

For A, points are expressed in the NDF-estimated current functional frame.  For
B, points are expressed in the camera-estimated goal functional frame.  These
three local coordinates are appended per point, while the explicit relative
SE(3) token remains a gated global condition.

| Input | Step dt mm | Step dr deg | Endpoint cm | Endpoint deg |
|---|---:|---:|---:|---:|
| correct two-level | 2.524 | 2.073 | 0.886 | 6.790 |
| zero both levels | 2.668 | 2.240 | 0.909 | 7.726 |
| shuffled both levels | 2.634 | 2.064 | 0.907 | 7.948 |

Correct versus zero:

- endpoint translation: 2.6% lower, wins 2/3 seeds
- endpoint rotation: 12.1% lower, wins 3/3 seeds

Correct versus shuffled:

- endpoint translation: 2.3% lower, wins 1/3 seeds
- endpoint rotation: 14.6% lower, wins 2/3 seeds

This is a positive pilot for directional control, but translation evidence is
not yet seed-stable.

## Why both levels matter

Per-point functional coordinates alone are not a valid method.  With no global
relation token, correct coordinates produced `1.016 cm / 7.275 deg`, while
shuffled coordinates unexpectedly produced `0.708 cm / 6.525 deg` and won all
metrics in all three seeds.  Wrong local frames can behave like a strong random
coordinate augmentation and create a misleading apparent gain.

A paired ablation retained the exact two-level network and correct local point
coordinates but zeroed only the global token.  Its result was exactly
`1.016 cm / 7.275 deg`, matching the module-removal experiment per seed.
Restoring the consistent global token reduced mean endpoint error by 12.9% in
translation and 6.7% in rotation (2/3 seed wins for each).  The defensible
mechanism is therefore consistency between local functional coordinates and
the global remaining relation, not either representation in isolation.

## Current claim boundary and next gate

Supported now:

- camera-only target-frame estimation is accurate;
- a category-specific NDF two-axis frame generalizes to blind mug assets;
- an oracle task relation is causally useful to a learned action-chunk policy;
- direct/global-only NDF token fusion is unreliable;
- the two-level NDF representation gives a repeatable rotation benefit and
  avoids the false shuffled-local-coordinate solution.

Not supported yet:

- simulator closed-loop success improvement;
- stable translation improvement against shuffled controls;
- transfer to a second object category/task;
- a T-ASE-level general claim.

## Online parity and closed-loop intervention result

The deployment path now reconstructs the training-time functional point
coordinates exactly.  It normalizes the base XYZ/RGB channels before appending
the already-scaled local coordinates, uses the same SO(3) medoid ensemble as
cross-fitting, and reads the marker-local target-frame calibration from the
dataset metadata.  With the three real NDF checkpoints, the online current
frame reproduces the stored offline NDF frame to `5.38e-6 deg`.  The remaining
`0.071 cm / 1.138 deg` target-frame difference observed in a replay diagnostic
comes from using the stored 128-point B cloud; deployment latches three full
camera clouds and does not use this downsampled diagnostic input.

A paired fresh-scene evaluation then restored the same post-perturbation
physics snapshot for every policy, ran six receding-horizon calls, held the
gripper closed, and used simulator object poses only for metrics.  The success
threshold was `2.5 cm / 15 deg` on blind mug IDs 8--9.

The original direct-concatenation two-level model failed its causal test:

| Branch | Success | Final translation | Final rotation |
|---|---:|---:|---:|
| correct geometry | 2/3 | 1.670 cm | 8.165 deg |
| same model, geometry zeroed online | 3/3 | 1.063 cm | 6.589 deg |
| zero-trained policy | 3/3 | 1.272 cm | 6.805 deg |

This falsifies direct per-point coordinate concatenation, despite its modest
offline benefit.  NDF confidence was normal and grasp slip was shared across
variants, so neither explains the regression.

## Reliability-preserving geometry adapter

The baseline XYZ/RGB point encoder is now preserved exactly.  Only the final
three functional coordinates enter a shared zero-initialized scalar-gated
adapter; the existing global SE(3) token also remains zero-start gated.  At
initialization, or when geometry is absent, the architecture is an exact no-op
rather than a corrupted baseline.

Its seed-0 offline factorial result was:

| Training relation | Endpoint translation | Endpoint rotation |
|---|---:|---:|
| correct | 0.775 cm | 6.451 deg |
| zero | 0.825 cm | 6.907 deg |
| shuffled | 0.905 cm | 7.990 deg |

The adapter removed the direct-concatenation failure.  In the first three
paired closed-loop episodes, correct geometry reached `1.399 cm / 5.809 deg`,
the same checkpoint with geometry zeroed reached `1.471 cm / 5.691 deg`, and
the zero-trained policy reached `1.507 cm / 5.606 deg`.  Correct geometry thus
improved translation by 4.9% versus its online ablation and by 7.2% versus the
zero-trained policy, but did not improve rotation.  This is a partial result,
not a closed method claim.

Increasing the deployed endpoint SO(3) action loss to 0.5 was rejected because
shuffled geometry (`0.832 cm / 6.759 deg`) beat correct geometry
(`0.899 cm / 6.958 deg`) offline.

## Rotation supervision and identifiability diagnostic

A separate lightweight SO(3) auxiliary head was added through the existing
rotation supervision path, with weight 0.1.  It predicts the remaining rigid
rotation during training but is not an inference input.  Offline seed 0 gave:

| Training relation | Endpoint translation | Endpoint rotation |
|---|---:|---:|
| correct | 0.745 cm | 6.060 deg |
| zero | 0.859 cm | 6.763 deg |
| shuffled | 0.745 cm | 6.567 deg |

However, strict online intervention again separated regularization from
geometry use:

| Branch | Success | Final translation | Final rotation |
|---|---:|---:|---:|
| correct geometry | 3/3 | 1.170 cm | 6.214 deg |
| same model, geometry zeroed online | 3/3 | 1.123 cm | 5.978 deg |
| zero-trained policy | 3/3 | 1.785 cm | 7.685 deg |

The auxiliary task improved the shared policy, but the action decoder still
did not causally use the geometry token.  This distinction is essential: the
correct model may beat an independently trained zero baseline without beating
its own inference-time geometry ablation.

Finally, the zero-trained policy was frozen and only its 35,074 geometry
adapter parameters were optimized.  Correct geometry produced
`0.775 cm / 6.465 deg`, whereas shuffled geometry produced
`0.743 cm / 6.411 deg`.  This rejects the hypothesis that freezing the baseline
alone makes the task relation identifiable.

## Revised claim boundary and next gate

Supported now:

- category-trained NDF and camera target frames are deployment-consistent;
- the failure is in fusion/identifiability, not online NDF reproduction;
- direct concatenation is unsafe;
- a zero-start adapter prevents catastrophic degradation;
- auxiliary SO(3) supervision improves the shared representation but does not
  prove that actions use the task relation.

Not supported yet:

- a causal closed-loop advantage from the learned NDF tokens;
- seed-scale expansion or a T-ASE-level method claim.

The fixed marker goal makes the desired relation strongly redundant with the
raw A/B clouds.  The next experiment must create relation-identifying training
pairs: for one observed object state, sample multiple reachable target frames,
transform the demonstrated remaining action consistently, and require the same
policy to follow the counterfactual relation.  Correct/zero/shuffled online
interventions remain the decision gate.  More encoder swaps, gate tuning, or
seed expansion should not precede that test.

## Relation-identifying multi-goal dataset

The identifiability gate was implemented without changing the deployed policy
or introducing a planner.  Each of the 219 camera observations is paired with
four real camera target observations from the same object-disjoint split.  The
current A cloud, proprioception, and cross-fitted NDF current frame remain
fixed, while B, the camera goal frame, the NDF current-to-goal relation, the
six-step action chunk, and the 16-step rigid object flow are changed together.

Simulator object and EEF poses are used only to generate behavior-cloning
labels and diagnostics.  At test time the policy still receives only camera
A/B clouds, proprioception, and the camera/NDF two-level relation.  The output
is still produced by the learned Transformer; the rigid interpolation used to
augment training labels is not an inference controller.

The resulting object-disjoint dataset contains 876 rows:

| Split | Object IDs | Rows |
|---|---|---:|
| train | 0--5 | 520 |
| validation | 6--7 | 188 |
| blind test | 8--9 | 168 |

Every source observation has exactly four goals and no donor crosses an object
split.  The predicted current frame reconstructed from each original
camera/NDF relation is numerically identical to the stored frame (maximum
rotation discrepancy `7.2e-6 deg`).  Across all new pairs, NDF relation error
is `1.141 cm / 7.181 deg` mean and `1.676 cm / 12.498 deg` p90.  The blind-test
desired relations cover a median `6.953 cm / 39.810 deg`, with p90
`14.037 cm / 67.306 deg`; the policy can no longer solve all rows by ignoring
the target relation and replaying one fixed goal action.

## Multi-goal offline result: three independent seeds

The frozen architecture is the reliability-preserving two-level adapter with
the training-only SO(3) auxiliary loss at weight 0.1.  It has 1.22 M trainable
parameters, consumes 128 point tokens, and predicts six 14-D bimanual action
deltas.  All variants use the same object split and training budget.

Blind-test endpoint error is:

| Seed | Correct NDF relation | Zero-trained | Shuffled-trained |
|---:|---:|---:|---:|
| 0 | 1.725 cm / 10.018 deg | 2.312 cm / 17.683 deg | 2.079 cm / 14.546 deg |
| 1 | 1.739 cm / 9.635 deg | 2.143 cm / 20.083 deg | 1.956 cm / 12.700 deg |
| 2 | 1.424 cm / 7.399 deg | 2.644 cm / 20.572 deg | 2.024 cm / 10.284 deg |
| mean | **1.629 cm / 9.017 deg** | 2.366 cm / 19.446 deg | 2.020 cm / 12.510 deg |

Correct geometry wins both endpoint metrics in every seed.  Relative to the
independently trained zero policy, it reduces translation error by 31.1% and
rotation error by 53.6%; relative to shuffled training, the reductions are
19.3% and 27.9%.

## Strict same-checkpoint causal intervention

Independent retraining can confound geometry use with regularization.  The
stronger test therefore freezes each correct checkpoint and changes only its
geometry input at blind-test inference:

| Input to the same checkpoint | Endpoint translation | Endpoint rotation |
|---|---:|---:|
| correct | **1.629 cm** | **9.017 deg** |
| shuffled | 2.338 cm | 18.717 deg |
| zero | 2.608 cm | 23.124 deg |

Correct geometry wins both metrics in all three seeds.  Removing the token
from the same network increases translation error by 60.1% and rotation error
by 156.4%, equivalently a 37.5% / 61.0% reduction when correct geometry is
restored.  Shuffling produces the same conclusion.  This is the first result
in this route that demonstrates causal action dependence rather than a shared
representation regularization effect.

## Strict paired closed-loop result

For every compared branch, the evaluator restores the exact same
post-perturbation physics snapshot, latches the same three camera target
frames, executes six fixed receding-horizon policy calls, and keeps the gripper
closed.  Simulator poses are metrics only.  Final success requires
`<= 2.5 cm` and `<= 15 deg` on blind mug IDs 8--9.

| Policy seed | Correct | Same checkpoint, geometry zeroed | Zero-trained |
|---:|---:|---:|---:|
| 0 | 3/3, 1.476 cm / 8.797 deg | 2/3, 3.561 cm / 18.794 deg | 2/3, 2.634 cm / 22.164 deg |
| 1 | 3/3, 1.590 cm / 5.925 deg | 2/3, 1.681 cm / 8.259 deg | 2/3, 1.873 cm / 7.966 deg |
| 2 | 3/3, 1.572 cm / 8.156 deg | 0/3, 2.033 cm / 18.605 deg | 2/3, 2.269 cm / 15.242 deg |
| aggregate | **9/9, 1.546 cm / 7.626 deg** | 4/9, 2.425 cm / 15.219 deg | 6/9, 2.259 cm / 15.124 deg |

At the same fixed execution horizon, correct geometry reduces final error by
36.3% in translation and 49.9% in rotation versus its own online ablation.  It
also improves both metrics versus the independently trained zero policy.  The
correct policy succeeds on every paired scene in every seed; the geometry
ablation does not.

## Updated conclusion and claim boundary

The earlier negative results did not show that NDF or geometry-enhanced policy
learning was intrinsically ineffective.  They showed that a fixed-goal
dataset made the desired relation redundant: a policy could minimize behavior
cloning loss without using the relation token.  Safe fusion prevented damage
but could not by itself create statistical identifiability.

The supported one-task claim is now:

> A task-aligned two-level geometric relation can causally improve a learned
> short-horizon action-chunk policy when training contains physically
> consistent multi-goal pairs that make the relation action-identifying.

This closes the method mechanism on one handled-mug task: category-trained NDF
provides the current functional frame, camera geometry provides the goal
frame, the consistent per-point and SE(3) tokens condition a learned action
decoder, and both offline intervention and paired closed-loop tests show that
the decoder uses them.

It is not yet a T-ASE-level general claim.  The remaining gates are more blind
goals and perturbation levels, then one second category/task with its own
category-specific NDF.  Those experiments should reuse the frozen method and
factorial controls; they should not add stage prediction, grasp tokens,
analytic action priors, or residual planning.
