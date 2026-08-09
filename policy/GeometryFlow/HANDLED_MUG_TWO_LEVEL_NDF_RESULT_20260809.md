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

The next required experiment is matched closed-loop deployment of correct,
zero, and shuffled two-level policies on blind mug IDs 8--9.  Offline
architecture search should stop until that test is complete.
