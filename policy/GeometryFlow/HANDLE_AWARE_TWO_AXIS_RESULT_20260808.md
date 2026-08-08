# Handle-aware two-axis result — 2026-08-08

## Frozen question

For a learned short-horizon manipulation policy, is one task axis sufficient,
or should camera geometry estimate a minimal two-axis functional frame?

The operated object is a handled mug.  Its directed local `+Z` axis points to
the handle and its directed local `+Y` axis follows the mug body.  These two
non-collinear axes determine a complete right-handed functional frame.  The
policy route remains the locked route:

```text
camera A/B point clouds
  -> task-trained NDF-style equivariant two-axis estimator
  -> local point-relation tokens + relative functional-frame token
  -> target-frame action canonicalization
  -> Transformer Cartesian action chunks
```

No simulator pose, planner trajectory, inverse-kinematics prior, learned
residual, grasp token, or stage prediction is a policy input.

## Why the original `021_cup` handle pilot was rejected

Only `021_cup` IDs 6--12 contain handles.  The frozen admission audit had
already found IDs 6/7 and 8--12 unreachable with the current robot/contact
interface.  A new marker-task smoke test reproduced that failure: the first
version required 22 attempts to admit one trajectory.  This is not evidence
against handle geometry, and seed filtering would be invalid.

The benchmark was therefore changed to `039_mug`, whose models have explicit
handle functional points and six grasp contacts.  The new task uses one direct
right-arm grasp and removes the old hanging-mug handover confound.

## Object-disjoint two-axis ablation

The existing 24-episode hanging-mug development dataset was used only for the
representation ablation.  The split is unchanged:

- train objects: mug IDs 2, 3, 4, 7;
- validation objects: mug IDs 1, 6 (used during the frozen frame training);
- held-out test objects: mug IDs 0, 5;
- test frames: 124 from six episodes;
- frame-model seeds: 0, 1, 2.

The evaluated NDF-style head already predicts exactly the proposed axes:
alpha predicts object `+Z` (handle), beta predicts object `+Y` (body), and
Gram--Schmidt produces `R=[X,Y,Z]` with `X=Y cross Z`.

Single-axis controls retain one predicted axis and reconstruct the missing one
from a mean direction computed on training identities only.  The shuffled
control retains the predicted vertical axis but substitutes a handle axis from
a different test episode at matched normalized time.

| representation | held-out SO(3), mean ± seed SD | handle-axis error | vertical-axis error |
|---|---:|---:|---:|
| **predicted two-axis frame** | **8.745 ± 0.712 deg** | **3.992 ± 0.399 deg** | **7.289 ± 0.581 deg** |
| predicted handle only + train vertical prior | 12.858 ± 0.118 deg | 9.750 ± 0.172 deg | 12.210 deg |
| predicted vertical only + train handle prior | 15.511 ± 0.254 deg | 11.163 ± 0.037 deg | 7.289 ± 0.581 deg |
| predicted vertical + shuffled handle | 18.866 ± 0.328 deg | 14.796 ± 0.228 deg | 7.289 ± 0.581 deg |

Thus two axes reduce mean SO(3) error by 32.0% relative to the stronger
single-axis control and by 43.6% relative to vertical-only.  Replacing only the
handle evidence while preserving vertical prediction more than doubles the
two-axis error.  The handle axis is therefore a task-relevant causal variable,
not an unused extra output.

The raw two-axis estimator still has a 2.42% over-90-degree flip rate.  This is
the known discrete sign ambiguity and motivates retaining label-free temporal
sign stabilization and estimator disagreement confidence.  It must not be
hidden by reporting only the median error (4.231 degrees).

Machine-readable result:
`outputs/geometry_flow/hanging_mug_cross_task_v1/handle_aware_two_axis_ablation.json`.

## Downstream learned-policy evidence already available

On the same held-out objects and three policy seeds, the correctly paired
camera-estimated functional-frame token achieved 11.090-degree endpoint
rotation error, versus 11.450 degrees for the capacity-matched exact-zero token
and 11.898 degrees for shuffled tokens.  It won all three seeds against both
controls.  The oracle frame reaches 10.800 degrees, so the camera estimator
retains 55.4% of the oracle-versus-zero gain.

When the same camera geometry also canonicalizes actions in the target frame,
endpoint rotation improves from 12.055 to 10.936 degrees and wins all three
policy seeds.  This supports using geometry structurally, rather than merely
concatenating a global feature.

The old hanging-mug closed-loop result is not a finished claim: strict success
was 3/12 versus 2/12, while mean endpoint metrics were worse because of
handover-induced hand-object slip.  The new direct-grasp marker task is intended
to remove that physical confound before the paper-scale policy comparison.

## New short-horizon task admission

Task: `place_handled_mug_geometry_marker`.

- object: `039_mug/base0` for the first smoke test;
- target: a visible T marker with continuously varying position and yaw;
- active arm: fixed right arm to isolate representation from arm reachability;
- initial mug grasp: fixed and shared by all representation conditions;
- recorded phase: gripper-closed pose correction only;
- success: translation at most 4 cm and complete SO(3) error at most 10 degrees;
- admission additionally requires the perturbed initial state to fail.

The first three consecutive scene seeds were all admitted without retries:

| scene seed | before translation / rotation | after one expert correction |
|---:|---:|---:|
| 0 | 5.186 cm / 11.665 deg | 0.102 cm / 0.190 deg |
| 1 | 6.312 cm / 13.203 deg | 0.145 cm / 0.311 deg |
| 2 | 8.149 cm / 17.598 deg | 0.289 cm / 2.162 deg |
| mean | 6.549 cm / 14.155 deg | 0.179 cm / 0.888 deg |

This 3/3 result proves task/expert feasibility, not learned-policy improvement.
The next gate must collect multiple mug identities and compare policies on
paired held-out initial states.

Manifest:
`outputs/geometry_flow/handled_mug_marker_smoke/pose_correction_manifest.json`.

## Refined paper framework

The defensible method is **task-aligned minimal functional frames**, not “NDF
feature concatenation.”

1. A category/task-adapted equivariant encoder estimates two directed
   functional axes and uncertainty from segmented camera points.
2. Per-point tokens encode local object-target relations; a compact global
   token encodes the remaining relative functional frame.
3. Robot state and demonstrated Cartesian actions are expressed in the target
   functional frame before a Transformer predicts action chunks.
4. Confidence gates unreliable geometry, while temporal sign stabilization
   resolves rare equivariant-frame flips.

The novelty is the task-derived representation contract and how it structures
action learning.  NDF is one suitable equivariant implementation; a
category-mismatched pretrained checkpoint is not itself the contribution.

## Remaining gates before a paper claim

1. Collect an object-balanced direct-grasp marker dataset over mug IDs 0--9,
   with train/validation/test identities fixed before training.
2. Train matched raw-point, one-axis, two-axis, shuffled-handle, exact-zero, and
   oracle-ceiling policies with identical capacity and three seeds.
3. Run paired closed-loop evaluation from stored initial simulator snapshots;
   report success, translation, rotation, and hand-object drift.
4. Retrain the category-specific NDF-style estimator using only training mug
   identities, then test blind identities.  Do not initialize from `shoe.pth`
   because the existing controlled experiment showed negative transfer.
5. Promote the result only if two-axis beats the strongest one-axis control and
   shuffled-handle in both held-out action error and paired closed-loop success.
