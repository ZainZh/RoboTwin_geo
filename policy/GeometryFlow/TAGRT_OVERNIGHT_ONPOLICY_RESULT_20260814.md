# TAGRT overnight on-policy result — 2026-08-14

## Decision

The fixed learned-policy mainline remains scientifically viable, but the
complete-task result is **not yet robust enough to scale**.  The strongest
paired result remains one complete seed-173 success for TAGRT against a Raw
failure.  Seed 175 remained unsuccessful after two admitted on-policy
correction iterations.  Do not report a multi-seed complete-task claim yet.

The overnight experiments rule out three narrow explanations:

1. invalid NDF frames are not the main seed-175 blocker;
2. relabeling actuator feedback as causal gripper commands alone is insufficient;
3. separately adapting motion and an observation-only gripper head does not
   maintain a consistent closed-loop state distribution.

The next minimum method change is an **action-coupled discrete gripper head**:
the open/close prediction must read the final denoised action-query tokens from
the same flow-matching trajectory it accompanies.  It must remain learned and
continuous-conditioned; no stage label, simulator pose, analytic recovery, or
hand-written gate is introduced.

## Fixed deployment contract

```text
camera current/goal point clouds
  -> frozen shoe NDF functional-frame ensemble
  -> local anchor relation tokens + global remaining-SE(3) token
  -> three-frame relation/temporal Transformer
  -> pi-style flow-matching six-step Cartesian action block
  -> learned binary gripper command
  -> execute one step and reobserve
```

The deployed policy never receives simulator object pose or an expert action.
Simulator state is used only to admit successful correction trajectories and
to compute evaluation success.

## Starting positive result

The exact-trunk geometry adapter `v8 alpha=0.8` produced the first strict paired
complete-task result on seed 173:

| Condition | Complete success | First close | Minimum translation | Minimum rotation |
|---|---:|---:|---:|---:|
| TAGRT | **1/1** | 37 | 2.36 cm | 1.01 deg |
| Raw | 0/1 | never | 24.00 cm | 32.56 deg |

TAGRT first entered pose alignment at policy step 220 and completed the required
success hold after the evaluation horizon was extended to 260.  This is a real
closed-loop inference rollout, not a demonstration replay.

Seed 175 failed under the same adapter, so this result establishes feasibility,
not robustness.

## Admitted seed-175 corrections

The original policy's first predicted-close state at chunk 41 was rejected:
the expert planner itself could not recover and the object was displaced at
high velocity.  That episode was never added to training.

Two recoverable states were admitted:

| Policy inducing the state | Handoff | Frames | Dense controls | Expert result |
|---|---:|---:|---:|---:|
| original deterministic-gripper policy | fixed chunk 20 | 198 | 2,412 | success, 26 hold steps |
| jointly repaired `v14` policy | predicted close, chunk 48 | 227 | 2,475 | success, 26 hold steps |

Only rows after each handoff were retained as expert supervision.  Policy-prefix
actions were excluded.  Causal binary commands were inferred from the physical
gripper motion onset; both admitted continuations contain exactly one close and
one open transition.

Frozen perception was accurate on both continuations:

| Correction | NDF frame median | Relative rotation median | Relative translation median |
|---|---:|---:|---:|
| chunk 20 | 1.05 deg | 1.16 deg | 2.08 cm |
| chunk 48 | 1.15 deg | 1.20 deg | 2.14 cm |

## Online results

All rows below are seed-175 complete-task rollouts with a 260-step limit.

| Model | Learned change | Closed interval(s) | Minimum EEF-object distance | NDF invalid onset | Ramp contact | Success |
|---|---|---|---:|---:|---:|---:|
| `v12` | causal command labels, contextual gripper head | 44--49 | 17.4 cm | 55, after object was flung | no | 0/1 |
| `v14` | chunk-20 flow-motion repair + contextual head | 48--64 | 11.7 cm | never | no | 0/1 |
| `v16` | two-state motion repair + retrained head | 6, 9 | 11.0 cm | never | no | 0/1 |
| isolated `v15 motion + v14 head` | second motion repair, stable prior head | 6, 8--107 | 12.7 cm | never | yes | 0/1 |

The progression is informative even though it is negative:

- `v12` showed that a six-step close followed by reopening can fling the shoe;
- `v14` eliminated perception degeneration and moved closer to grasp, but
  reopened after 17 steps without establishing contact;
- retraining the independent gripper head on both correction trajectories made
  it close spuriously at the initial state;
- restoring the older head while keeping the newer motion decoder held the
  gripper closed for about 100 steps and made contact, but at the wrong
  configuration and never lifted or placed the shoe.

Thus, adding more copies of the same correction protocol is no longer the
right next action.

## What the evidence supports

Supported:

- camera NDF/TAGRT tokens are accurate on the admitted online states;
- the policy consumes geometry: on the held-out nominal test split, TAGRT
  first-action normalized motion MSE is about `0.180`, versus Raw `0.256`, and
  zero/shuffled TAGRT rises to `0.566/0.668`;
- task-aligned geometry can cause a complete learned-policy success where the
  capacity-matched Raw policy fails (seed 173);
- the collect/admit/preprocess/train/evaluate DAgger loop is reproducible and
  rejects expert-unrecoverable states.

Not supported:

- robust complete-task success across seeds;
- a paper-scale success-rate advantage;
- the claim that an observation-only gripper classifier can be trained
  independently from a stochastic action generator.

## Narrow technical diagnosis

The current contextual gripper head reads the newest encoded state token but
does not read the action block produced by flow matching.  Motion adaptation
changes the online states seen by that head; gripper adaptation changes object
contact and therefore the states seen by the motion decoder.  Offline command
accuracy can consequently exceed 99% while the joint online sequence fails.

This explains why extra epochs or another single-seed correction are unlikely
to solve the blocker.  It also preserves the core TAGRT hypothesis: the failure
is at the joint action-consumption interface, not at NDF frame estimation or at
local/global geometry token construction.

## Next decisive experiment

Use the existing data and frozen geometry stack.  Add one action-coupled head
that predicts gripper commands from the **final denoised action-query tokens**
and temporal memory.  Train it with clean expert motion queries at flow time
one, then deploy it on the sampled motion queries.  Raw and TAGRT must have the
same head and parameter count.

Run only this gate first:

1. TAGRT on seeds 173 and 175;
2. capacity-matched Raw on the same seeds if TAGRT obtains at least one success;
3. scale to the six fixed scenes only if TAGRT reaches `2/2` and beats Raw;
4. run zero/wrong-geometry controls only after the positive complete-task gate.

This change stays inside the geometry-enhanced learned-policy story.  It does
not add planning, a stage classifier, simulator pose input, or an analytic
release rule.

## Canonical artifacts

- Seed-175 correction 1:
  `/shared/sz/TAGRT-v1/simulator_data/place_shoe_geometry_marker/fulltask_tagrt_pi_deterministic_gripper_dagger_seed175_chunk20_v1`
- Seed-175 correction 2:
  `/shared/sz/TAGRT-v1/simulator_data/place_shoe_geometry_marker/fulltask_tagrt_pi_joint_v14_dagger_seed175_preclose_v1`
- Processed correction 1:
  `/shared/sz/TAGRT-v1/processed_datasets/fulltask_tagrt_v2/denseeef_left60_pi_deterministic_dagger_seed175_chunk20_fixedconf_v1`
- Processed correction 2:
  `/shared/sz/TAGRT-v1/processed_datasets/fulltask_tagrt_v2/denseeef_left60_pi_joint_v14_dagger_seed175_preclose_fixedconf_v1`
- Best seed-173 TAGRT checkpoint:
  `/shared/sz/TAGRT-v1/models/fulltask_tagrt_v2/denseeef_left60_pi_flow_dagger_geometry_adapter_v8_alpha080_exacttrunk_seed0/tagrt_pi_flow_seed0_final.pt`
- Latest isolated diagnostic checkpoint:
  `/shared/sz/TAGRT-v1/models/fulltask_tagrt_v2/denseeef_left60_pi_flow_dagger_joint_v16_motion175_iter2_command_seed0/tagrt_pi_flow_seed0_v15motion_v14head.pt`

All newly produced datasets, checkpoints, traces, and videos are under the
fallback canonical root `/shared/sz/TAGRT-v1`; no new artifact was written to
the Git workspace.
