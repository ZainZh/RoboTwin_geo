# Frozen Task-Aligned Geometry Token Protocol — 2026-08-09

## Status

This document freezes method version `TAGRT-v1` for the remaining paper
experiments.  The method was selected before the final handled-mug stress-seed
replication and must not be changed in response to individual test episodes.

The paper question is fixed:

> Can category-level geometric representations improve a learned
> short-horizon manipulation policy when converted into task-aligned local and
> global relation tokens?

The method is learned behavior cloning with structured geometric conditioning.
It is not a planner, geometric controller, inverse-kinematics prior, learned
residual, task-stage predictor, or grasp-pose policy.

## Frozen inference graph

```text
segmented current/target camera point clouds A/B
  -> category-specific equivariant functional-frame estimator
  -> confidence and causal temporal branch stabilization
  -> local per-point functional coordinates / task flow
  -> global remaining relative SE(3) token
  -> robot proprioception + zero-start gated Transformer
  -> six-step, 14-D bimanual Cartesian action chunk
```

The estimator may be implemented with NDF or another category representation,
but every replacement must produce the same functional-frame contract and must
be evaluated without changing the policy architecture.

## Frozen representation and policy

- 128 camera point tokens;
- 32 relation anchors and 16 supervised rigid-flow samples;
- mutually consistent local functional coordinates and global relative SE(3);
- zero-initialized local geometry adapter and zero-start global relation gate;
- target-functional-frame action canonicalization;
- approximately 1.22 M trainable policy parameters;
- six 14-D bimanual Cartesian action deltas;
- training-only SO(3) auxiliary supervision with weight 0.1;
- action, geometry-flow, and rigidity objectives already stored in checkpoint
  metadata;
- no additional inference head for stage, success, grasp, planning, or
  recovery.

No encoder, fusion module, action space, horizon, auxiliary head, or loss
weight may be tuned on the frozen blind test objects.  Implementation fixes are
allowed only when they restore training/deployment parity and are covered by a
unit or exact-replay test.

## Frozen data principle

Every task must use an object-disjoint train/validation/test split fixed before
policy training.  Training-object geometry is leave-one-object-out cross-fitted
so the policy never receives in-sample estimator outputs.  Validation/test use
an estimator trained on training identities only.  Confidence thresholds are
calibrated only on cross-fitted training objects.

The behavior-cloning dataset must be relation-identifying: one current camera
state is paired with multiple physically consistent camera goals, and the goal
cloud, relation token, local flow, and action label change together.  Simulator
poses may generate labels and metrics but are not policy inputs.

Frozen category instances:

| Task | Train / validation / blind objects | Goals per source |
|---|---|---:|
| Shoe placement | `2,3,4,7,8,9` / `1,6` / `0,5` | 2 |
| Handled-mug alignment | `0--5` / `6,7` / `8,9` | 4 |

## Frozen controls

Every principal table must include matched capacity, data, split, and seed:

1. correct two-level geometry;
2. exact-zero geometry;
3. shuffled geometry from a different row;
4. same-checkpoint zero and shuffled interventions;
5. raw/capacity-matched point policy and oracle ceiling where available;
6. local-only/global-only ablations as mechanistic secondary evidence.

The same-checkpoint intervention is the primary causal test because it changes
only geometry content while holding weights and all other observations fixed.

## Frozen evaluation

- policy seeds: 0, 1, and 2;
- object-blind offline endpoint translation and complete SO(3) error;
- exact in-process paired simulator snapshots for closed loop;
- three latched target camera frames;
- one executed step per receding-horizon call;
- six fixed policy calls with `continue_after_success` for matched budgets;
- final-pose success is primary; first-threshold crossing is reported as a
  diagnostic when forced continued execution leaves the success set;
- simulator object poses are metrics only;
- replay/setup failures are paired protocol errors and never counted as policy
  failures or silently discarded per method.

Task thresholds remain fixed:

| Task | Translation | Rotation |
|---|---:|---:|
| Shoe placement | 4.0 cm | 15 deg |
| Handled-mug alignment | 2.5 cm | 15 deg |

## Evidence at freeze

Shoe placement, common ten closed-loop scenes:

- correct: `80%, 60%, 60%` over policy seeds;
- same-checkpoint zero: `50%, 30%, 30%`;
- mean causal gain: `+30.0` percentage points in every seed;
- scene wins/losses/ties: `6/0/4`, exact sign `p=0.03125`.

Handled-mug alignment, complete twelve closed-loop scenes:

- correct: `6/12, 6/12, 7/12`;
- same-checkpoint zero: `3/12, 4/12, 1/12`;
- descriptive pooled count: `19/36` versus `8/36`;
- scene wins/losses/ties: `7/0/5`, exact sign `p=0.015625`.

The two categories jointly support a representation-and-learning claim, not a
shoe-specific NDF trick.  NDF supplies the functional frame; the paper method
is the task-aligned two-level representation contract, leakage-free training,
and reliable policy fusion.

## Remaining paper experiments

The architecture is frozen.  Remaining work is evidence expansion:

1. collect larger fully replayable blind snapshot sets for both categories;
2. complete raw, zero, shuffled, local-only, global-only, oracle, and full
   factorial tables under identical snapshot sets;
3. report per-object, per-perturbation, and three-seed uncertainty;
4. measure estimator latency, confidence coverage, and failure calibration;
5. run action-budget curves only as diagnostics; retain the predeclared
   six-call result as the primary number;
6. prepare the two-category method/ablation/statistical tables and qualitative
   failure analysis for the paper.

Translation range and forced-budget over-correction are current limitations.
They may motivate future work, but they must not trigger an unfrozen method
change during the confirmatory experiment phase.
