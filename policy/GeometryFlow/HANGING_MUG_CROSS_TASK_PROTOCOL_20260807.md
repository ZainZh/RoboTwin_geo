# Hanging-mug cross-task geometry-token protocol

Date frozen: 2026-08-07, before collecting pose-correction demonstrations or
training any hanging-mug policy.

## Question

Does the two-level task-aligned geometry-token architecture transfer from shoe
placement to a new object category and a new functional relation, or was the
shoe result task-specific?

## Task boundary

- Task: the final short-horizon phase of `hanging_mug`.
- Initial state: the right gripper already holds the mug after the scripted
  handoff, near the rack.
- Learned output: six-step bimanual EEF action chunk with the gripper kept
  closed during pose correction.
- Excluded: initial grasp, bimanual handoff, release, stage prediction, inverse
  kinematics supervision, and residual planning.
- Geometry input: current camera point clouds for mug A and rack B, plus robot
  proprioception.

This boundary isolates the claim under test: whether task-aligned object
geometry improves a learned manipulation action policy.

## Frozen identity split

- Development/train mug IDs: 0--7 from `039_mug`.
- Blind mug IDs: 8 and 9.
- No blind state, success label, or representation error may select an epoch,
  width, token construction, perturbation range, or loss weight.

## Development collection

- 24 admitted correction episodes: eight development mug IDs x three levels.
- At most three episodes per mug identity.
- Perturbation levels (commanded): 2--4 cm / 10--25 degrees,
  3--5 cm / 20--35 degrees, and 4--6 cm / 30--45 degrees.
- Every admitted initial state must fail the 2 cm / 10 degree geometric
  threshold; no already-solved episode is retained.
- The demonstration expert may perform at most two recorded re-alignment
  moves.  The second move recomputes the EEF target from the newly observed
  mug-in-gripper relation; the success threshold is not relaxed.
- The expert endpoint, camera A/B clouds, robot EEF state, and task-state
  geometry labels must all be finite and present before dataset conversion.

The initial one-pass feasibility collection was stopped after 16 accepted
episodes when repeated near-threshold failures exposed in-gripper pose drift.
That partial set is retained as diagnostics but excluded from training.  The
24-episode development set is recollected from scratch under the uniform
two-correction contract above.

## Staged gate

1. Expert and data-contract smoke test on development IDs.
2. Oracle relative-functional-frame ceiling on development validation states.
3. Camera-only task-aligned frame estimator trained only on development IDs.
4. Minimal policy comparison: raw point tokens, exact zero-global intervention,
   and full two-level tokens.
5. If the development gate is positive, freeze checkpoints and evaluate exact
   paired snapshots on mug IDs 8 and 9 with three policy seeds.

The development gate requires the full method to improve both endpoint
rotation and closed-loop success over raw points and exact zero-global.  Flow
Chamfer is diagnostic only and cannot pass the gate by itself.

## Reporting

- Primary: paired closed-loop success.
- Secondary: final translation, final SO(3) error, monotonic error reduction,
  grasp slip, and representation frame error.
- Report every admitted state and every runtime failure.
- Use mug identity/state hierarchical resampling; policy seeds on one stored
  state are not independent scene trials.

The completed three-seed oracle gate and its limitations are recorded in
`HANGING_MUG_ORACLE_GATE_RESULT_20260807.md`.

## Status on 2026-08-07

- The oracle ceiling passed: the full relative SE(3) token beat the
  capacity-matched zero-global control on endpoint rotation in all three
  seeds.
- The current-mug camera-frame gate also passed offline: correctly aligned
  predicted tokens beat both zero and shuffled controls in all three seeds and
  retained 55.4% of the oracle orientation gain.
- This is not yet a fully camera-only result because the rack/goal functional
  frame is still supplied from task-state labels.
- The next frozen intervention is to estimate camera-B rack/goal geometry and
  rerun the same controls before any closed-loop or blind-identity claim.
