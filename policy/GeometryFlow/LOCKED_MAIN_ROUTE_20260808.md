# Locked geometry-conditioned policy route (2026-08-08)

## Frozen research scope

The paper route is fixed to a learned, short-horizon manipulation policy:

```text
current camera point clouds of operated object A and target object B
  -> task-aligned NDF / UTONIA / raw geometric features
  -> local point-relation / rigid-flow tokens
  -> explicit global current-to-goal SE(3) token with uncertainty
  -> robot proprioception and active-arm identity
  -> Transformer action-chunk policy
  -> Cartesian end-effector actions
```

The following are explicitly outside the method:

- analytic planning or inverse-kinematics trajectories at policy inference;
- nominal controller plus learned residual;
- task-stage prediction;
- a hand-object or grasp token.

Scripted motion is permitted only for demonstration generation and matched
pre-policy setup. Simulator object poses are permitted only for supervision and
evaluation, never as deployment inputs.

## Why the previous natural-task rollouts were misleading

The policy comparison was confounded by non-rigid object motion after grasping.
This is a task/data admission failure, not evidence against geometric
representation.

The fixed no-policy gate executes grasp, hold, nominal task motion, an exact
1 cm / 10 degree perturbation, and expert return while measuring the object pose
relative to the active end effector.

| task / assets | gate result | decision |
|---|---:|---|
| hanging mug, IDs 0--7 | 0/8 within 5 mm / 3 deg; mean max drift 1.520 cm / 10.480 deg | reject for primary policy comparison |
| phone stand, five phone IDs | 1/5 within 5 mm / 3 deg; several objects drop | reject |
| kettle to mug, kettle 0 smoke | marked grasp has no reachable IK solution | stop before policy training |
| bowl to plate, four original IDs | 1/4 task-aligned stable, one setup-unstable | reject bowl category |
| cup to plate, original IDs 1--7 | IDs 1/2/3/5 stable; ID 4 slips; IDs 6/7 unreachable | admit stable subset only |
| cup to plate, extra IDs 0/8--12 | ID 0 stable; IDs 8--12 unreachable | add ID 0 only |

The admitted natural-task object set is therefore cup IDs `0,1,2,3,5`.
Rejected assets remain useful as a separately reported physical stress test;
they are not silently counted as policy failures or mixed into training.

## Repaired geometry contract

The original `place_container_plate` expert used a 3 cm placement clearance in
the current grasp direction. Consequently, identical cup/plate geometry could
produce different expert targets under different grasps. This violated the
object-object representation claim and implicitly required grasp information.

The repaired task defines the same clearance along the plate functional z axis.
The following now share exactly one target definition:

1. the scripted demonstration endpoint;
2. `goal_T_A_from_B_oracle` supervision;
3. the continuous translation and symmetry-axis error metric.

On two replayed smoke scenes, the old terminal translation errors were
2.947 cm and 3.071 cm. After the repair they are 0.085 cm and 0.192 cm, with
0.235 and 0.316 degree symmetry-axis errors. Both camera recordings contain
separate 1024-by-6 A/B point clouds, A/B pose labels, the oracle functional
relation, relation-phase metadata, EEF poses, and actions on all 11 frames.

## Current experiment

Collect 20 object-balanced correction demonstrations:

- cup IDs `0,1,2,3,5`;
- four admitted perturbation bands per object;
- exact object-separated camera point clouds;
- one learned-policy scope only: gripper-closed short-horizon correction;
- no stage label as policy input or output.

After validation, build six-step Cartesian action chunks and run the same model
capacity, split, and optimization for:

1. raw A/B point tokens;
2. zero global relation token;
3. correct oracle relation token as an architecture ceiling;
4. shuffled relation token as a causal control;
5. camera-derived task-aligned NDF;
6. camera-derived UTONIA or an alternate point encoder.

The oracle/zero/shuffled gate is run before category-NDF training. If correct
geometry cannot improve held-out-object action prediction, the fusion/action
interface is repaired without blaming the encoder. Only after this gate passes
is a cup/mug-category NDF trained and inserted through the unchanged token
interface.

## First repaired-task evidence

The admitted dataset contains 20 demonstrations (five cup identities by four
perturbation bands), producing 233 six-step action-chunk samples after repeated
EEF frames are collapsed.  All results below are object-disjoint.

For held-out cup 1, the endpoint translation errors across three training seeds
were:

| input | seed 0 | seed 1 | seed 2 | mean |
|---|---:|---:|---:|---:|
| raw A/B point tokens | 0.839 cm | 0.886 cm | 0.966 cm | 0.897 cm |
| local task-relation tokens | 0.797 cm | 0.746 cm | 0.792 cm | 0.778 cm |
| local + symmetry-aware global token | 0.779 cm | 0.761 cm | 0.772 cm | 0.771 cm |
| local + shuffled global token | 0.841 cm | 0.823 cm | 0.911 cm | 0.858 cm |

Local task-relation tokens therefore improve the raw-token mean by 13.2% on
this held-out identity.  A correctly paired quotient token improves over its
shuffled intervention in all three seeds (0.771 versus 0.858 cm mean).

The local-token result also has the same direction on all three legal
object-disjoint folds at seed 0:

| held-out fold | raw | local relation |
|---|---:|---:|
| fold 0 (cup 0/5) | 0.860 cm | 0.807 cm |
| fold 1 (cup 1) | 0.839 cm | 0.797 cm |
| fold 2 (cup 2) | 0.829 cm | 0.802 cm |

The direct global-token fusion is not yet stable across folds: it obtains
1.002, 0.779, and 0.686 cm on folds 0/1/2.  Zero-start gated injection and an
explicit per-anchor remaining-flow injection both learn gates close to zero and
fall back to local-token performance.  The global quotient token is therefore
retained as a confidence-gated secondary condition and causal ablation, not
claimed as the current source of the main gain.  This is an architecture/data
finding rather than evidence against NDF or UTONIA, neither of which has yet
been inserted into this repaired cup benchmark.

## First paired closed-loop pilot

The deployment/evaluation path was repaired to support raw/local checkpoints
without requiring NDF metadata, to serialize the active-arm identity, and to
enforce the manifest's cup category and object IDs.  Four perturbation levels of
held-out cup 1 were evaluated from identical portable simulator snapshots.  No
simulator pose, oracle transform, planner, or residual controller was supplied
to either policy.

| policy | success at 1 cm / 3 deg | mean translation reduction | mean rotation reduction |
|---|---:|---:|---:|
| raw point-token policy | 0/4 | 0.102 cm | -0.752 deg |
| local relation-token policy | 1/4 | 0.367 cm | 0.004 deg |

The one recovery entered the threshold after two receding-horizon policy calls;
the matched raw policy failed after all six.  Final grasp drift was small, so
the comparison was not invalidated by the earlier non-rigid-grasp confound.

This pilot is positive but not a paper-level success result.  Twenty
demonstrations are sufficient to detect a representation effect, but not to
train a reliable closed-loop controller over the largest perturbations.  The
next admission gate is a larger object-balanced cup dataset with denser state
coverage, followed by the same raw/local pairing, then matched NDF and UTONIA
descriptors through the unchanged local-token interface.

## 100-demonstration scale check and first frozen-encoder gate

The larger admitted dataset contains 100 demonstrations, exactly balanced over
five object identities and four perturbation bands.  It produces 1,139
six-step action samples.  At seed 0, raw and from-scratch local relations are
effectively tied across folds 0/1/2: raw endpoint translation averages 0.914 cm
and local relations average 0.920 cm.  More trajectories from the same small
set of identities therefore do not replace a transferable point
representation.

The old descriptor adapter was then audited and found to inject features only
on object A.  A shared, zero-start adapter now projects frozen point descriptors
for both A and B before relation attention.  The first controlled fold-0 test
uses public UTONIA stage-0 descriptors (54 dimensions) and identical-capacity
zero and within-cloud-shuffled controls:

| descriptor input | best epoch | endpoint translation | endpoint rotation |
|---|---:|---:|---:|
| correct UTONIA s0 | 81 | 0.541 cm | 3.590 deg |
| all-zero descriptor | 42 | 0.479 cm | 3.396 deg |
| pointwise-shuffled UTONIA s0 | 42 | 0.475 cm | 3.288 deg |

The correct descriptor stream learns a nonzero gate (0.121), but it does not
beat either causal control.  Thus the favorable absolute error relative to the
earlier RGB runs must not be attributed to UTONIA; the main change is the safe
XYZ-only fallback.  This result rejects UTONIA-s0 concatenation as current
evidence for the paper claim, but does not reject the locked token architecture
or a category-matched NDF.  The next encoder gate is a jointly trained
cup/container-and-support NDF inserted through exactly this shared interface,
again requiring correct features to beat both zero and shuffled controls before
any closed-loop scaling.
