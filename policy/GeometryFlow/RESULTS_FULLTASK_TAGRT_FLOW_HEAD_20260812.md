# Full-task TAGRT motion-head gate — 2026-08-12

## Decision

This gate asked whether the current full-task failure was primarily caused by
the deterministic regression action head or by an insufficient number of
training epochs.  It kept the fixed learned-policy route: camera point clouds,
NDF/TAGRT local and global relation tokens, robot state, and a temporal
Transformer.  No planner, stage gate, residual controller, or simulator pose
was introduced at deployment.

The decision is **not to promote the current flow head and not to spend the
full 3000-epoch budget on it**.  Geometry remains useful offline, but replacing
the dense-joint regression head with conditional flow matching does not close
the online pre-grasp distribution shift.

## Why the older success video and current failures are not contradictory

The full-task success video in
`/shared2/sz/TAGRT-v1/paper_media/fulltask_pilot_20260811/` was a genuine
closed-loop policy inference, not an expert replay.  It used one memorized
training trajectory from the earlier nine-trajectory pilot plus a frozen
gripper-head calibration.  On that single scene, TAGRT succeeded while Raw and
zero geometry failed.  It admitted the full learned-policy interface and
showed causal geometry consumption; it was never a multi-scene stability
result.

The later fixed-left 60-trajectory gate evaluated six distinct initializations
from the actual training split and yielded 0/6 for Raw and TAGRT.  The newer
gate therefore asks a harder and more relevant question: whether the policy is
stable across multiple training-distribution scenes rather than whether one
scene can be memorized.

## Training-budget audit

RoboTwin's DP3 configuration uses 3000 complete dataloader epochs, batch size
256, cosine scheduling, a 500-update warmup, and EMA.  The earlier full-task
regression policies used roughly 8k optimizer updates, while the first flow
pilot used only about 5--7k updates and lacked cosine scheduling and EMA.

To obtain a fast decision without mechanically copying a multi-hour budget,
the corrected paired gate used:

- 300 complete epochs and 7500 optimizer updates;
- batch size 256;
- 500-update warmup and cosine decay;
- EMA checkpoint selection;
- identical Raw/TAGRT data, split, capacity, and optimization;
- the four admitted on-policy correction trajectories in the training split.

The validation curves were near their plateau by epochs 225--300, so a longer
run was not justified before an online gate.

## Corrected head isolation

The first regression-to-flow conversion jointly updated the complete network.
Its seed-173 rollout failed immediately by closing the wrong right gripper at
physical control 10 in a fixed-left task.  This exposed an implementation
coupling: the nominally separate gripper output still consumed the noisy flow
action query.

The corrected experiment therefore:

1. froze the point/object/NDF-TAGRT temporal memory trunk;
2. optimized only the 24D continuous flow motion decoder;
3. excluded gripper loss from flow training;
4. deployed the already calibrated regression gripper policy on the same
   normalized memory, independent of flow noise.

This makes the intervention an actual continuous motion-head replacement.

## Offline result

| condition | best validation motion MSE | test motion MSE |
|---|---:|---:|
| Raw flow motion head | 0.071603 | 0.241527 |
| TAGRT flow motion head | **0.036392** | **0.202102** |
| TAGRT with zero geometry | n/a | 0.460549 |
| TAGRT with shuffled geometry | n/a | 0.386826 |

TAGRT lowered validation error by 49.2% relative to Raw.  Zeroing or mismatching
geometry strongly damaged the same model.  Thus the fixed trunk still exposes
useful task-aligned geometry to a flow decoder.

## Online seed-173 gate

The corrected TAGRT flow motion head plus frozen regression gripper failed 0/1:

- 200 closed-loop chunks / 3000 physical controls;
- neither gripper closed, because the left EEF never reached the learned
  pre-grasp neighborhood;
- the shoe did not move;
- best combined task error: 0.2400 m / 32.56 deg;
- NDF confidence: 1.0;
- NDF source ensemble disagreement: 1.38 deg;
- no invalid NDF ensemble member.

This rejects a perception-collapse explanation and does not support spending
ten times more compute on the same head.  Raw and the remaining five online
seeds were not run because correct TAGRT itself failed the predeclared minimum
gate; those runs could not establish a geometry advantage.

## Narrow next action

The next fast gate should change only the action contract.  The current labels
are replay-equivalent low-level joint drive targets, but learning a 15-step
high-frequency controller trace is brittle under small observation drift and
is not the usual high-level action space of mainstream manipulation policies.

Instrument the expert collector to record the actual commanded absolute EEF
targets passed to the motion controller, rather than using observed next-frame
EEF states.  Admit 5--10 trajectories by replaying those commands through the
same online controller, then train a capacity-matched Raw/TAGRT pair and run
the single-scene online gate.  Scale only if correct TAGRT first reaches grasp
and task progress.  This remains a geometry-enhanced learned policy and does
not add analytic planning, residual execution, or stage rules.
