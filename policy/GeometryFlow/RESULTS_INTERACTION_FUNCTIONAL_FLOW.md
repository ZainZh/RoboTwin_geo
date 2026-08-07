# Interaction-functional flow: current evidence and locked route

Date: 2026-08-01

## Decision

The direction is retained, but the first action architecture is not. The
camera geometry is useful and causally active; a local behavior-cloning
diffusion adapter does not yet turn that signal into recovery-capable
closed-loop behavior. The route is therefore:

```text
A/B camera point clouds + gripper proprioception
  -> functional correspondence / object-in-gripper hypothesis bank
  -> relative SE(3), sparse rigid action-flow tokens, uncertainty
  -> phase-factorized action generation
       motion/recovery prior: analytic grasp-constrained rigid flow
       contact/release head: learned residual
  -> component-wise safe gates and base-policy fallback
```

NDF, UTONIA, DINO, PCA, and PointTransformerV3 are candidate front ends for
the same hypothesis/flow interface. They are not concatenated as one global
vector and called the method. A pi-style policy is also not the immediate
answer: changing the backbone cannot create the missing off-trajectory
recovery labels or an explicit rotation objective.

## Rotation-encoding audit

The dataset stores a pose rotation as `R[:, :2].reshape(6)`, which is a
row-major interleaving of the first two columns. The old relative-frame path
decoded it as two contiguous columns. This left translation unchanged but
corrupted all earlier relative-rotation tokens.

The decoder is fixed in `functional_action_frame.py`, new checkpoints carry
`functional_frame_encoding=relative_se3_pose6d_v2`, deployment rejects legacy
checkpoints, and a unit test guards the two layouts. Rotation claims from
legacy adapters are invalid and are not used below.

## What the camera estimator can recover

The camera-only estimator retrieves a source relation once, expands gripper
symmetries, selects the relation after grasp, and then propagates it using EEF
proprioception. Simulator object pose is used only for offline error labels;
the deployment path reads A/B point clouds and robot state.

Across five object-disjoint folds, ten held-out shoes, and 8,880 frames:

| Camera relative-pose error | Translation | Rotation |
|---|---:|---:|
| All frames | 1.509 cm | 18.093 deg |
| Before a grasp relation is available | 2.088 cm | 29.173 deg |
| Object held / relation available | **1.033 cm** | **8.975 deg** |
| Target B marker frame alone | 0.398 cm | 2.010 deg |

Thus the target marker is not the main perception bottleneck. Pre-grasp
orientation and cross-object relation selection remain weak; the held-object
relation is already useful enough to test the action interface.

This agrees with the larger relation-planner experiment in
`RESULTS_GRASP_RELATION_ROUTE.md`: geometry-selected relations improve paired
success from 39/100 to 53/100 over PCA relations (`p=0.0288`) and strongly
reduce orientation error. Its repeat audit shows that contact/release variance
still prevents a stable task-success effect size. The representation therefore
has physical closed-loop value, but does not by itself solve manipulation.

## Correct-v2 frozen diffusion adapter

The action model is a 1.26M-parameter sample-prediction diffusion transformer.
The base is frozen and a zero-gated 31,298-parameter adapter injects relative
SE(3) as explicit anchor-flow tokens at every denoising layer. Translation,
rotation, and gripper outputs can be accepted independently.

With correct v2 oracle relative geometry, the five-fold seed-0 factorization
is:

| Output group accepted from adapter | Translation | Rotation |
|---|---:|---:|
| Frozen diffusion | 1.9913 cm | 5.8200 deg |
| Translation only | **1.9546 cm** | 5.8200 deg |
| Rotation only | 1.9913 cm | 5.6205 deg |
| Shoe bootstrap 95% | `[-0.0732,-0.0041]` cm | `[-0.5780,+0.1575]` deg |

The translation improvement is supported over the ten held-out shoes. The
rotation interval crosses zero and two folds regress, so rotation remains
disabled in the promoted runtime.

## Camera-conditioned causal test

Replacing oracle relative pose with the camera estimator retains the seed-0
five-fold translation result:

| Condition | Shoe-balanced translation | Change from base |
|---|---:|---:|
| Frozen diffusion | 1.9913 cm | -- |
| Correct camera relative geometry | **1.9541 cm** | **-0.0372 cm** |
| Zero geometry | 2.0075 cm | +0.0162 cm |
| Phase-matched wrong episode | 2.0037 cm | +0.0124 cm |
| Relation phase only | 1.9600 cm | -0.0313 cm |
| Pre-relation only | 1.9854 cm | -0.0059 cm |

The correct camera result wins five shoes, ties two, and loses three; its shoe
bootstrap interval is `[-0.0749,-0.0030] cm`. It is about 0.0535 cm better
than zero input and 0.0496 cm better than shuffled input. This establishes that
the adapter uses the geometry content rather than merely benefiting from extra
parameters.

The result is not fully seed-stable. Over three fold-0 policy seeds, correct
camera geometry changes 2.0309 to 2.0008 cm, but wins only two seeds and the
hierarchical bootstrap interval is `[-0.0728,+0.0297] cm`. This is enough to
retain the interface, not enough to claim a finished algorithm.

## Why offline gains do not close the loop

On fixed unseen-shoe seed 100002, with identical diffusion noise and action
schedule:

| Deployment | Success | Minimum translation | Minimum rotation |
|---|---:|---:|---:|
| Frozen base, all geometry gates zero | 0/1 | **1.819 cm** | **75.996 deg** |
| Camera adapter, translation gate only | 0/1 | 3.222 cm | 77.541 deg |
| Camera adapter, translation + rotation | 0/1 | 4.236 cm | 105.634 deg |

The translation adapter is worse on this scene and cannot affect rotation by
construction. Enabling the unvalidated rotation branch makes the result much
worse and lowers mean commanded rotation from about 0.217 to 0.169 degrees per
action. This three-way paired failure rejects the current local-adapter
checkpoint as a deployable policy, despite its positive teacher-forced
endpoint metric.

The reason is visible in the supervision. During the held-object phase, 13.4%
of samples have more than 75 degrees of remaining geometric goal error, but
the median first rotation label is only 0.025 degrees. The model is trained to
imitate the demonstrator's next six local deltas, not to recover from an
arbitrary 75-degree closed-loop deviation. Twelve previously collected
on-policy recoveries improved teacher-forced metrics but did not close this
coverage gap; simple synthetic perturbations also degraded nominal behavior.

Therefore the current bottleneck is primarily the action objective and data
distribution, not the target-frame encoder. A bigger DP3, PTv3, or pi backbone
trained on the same targets is unlikely to fix it.

A broad recovery augmentation confirms that the remedy cannot be a naive data
mixture. Adding 3,806 grasp-preserving samples (30% of the mixed dataset) with
15--120 degree rotations raises the held-object first-action median from 0.025
to 3.19 degrees. On augmented validation the relative-flow model beats its
paired no-geometry model (0.247 versus 0.270 normalized loss), but both lose
nominal behavior badly. On the original fold-0 held-out trajectories they
score 2.78 cm/16.19 degrees and 3.04 cm/18.41 degrees, respectively, versus
about 1.97 cm/6.49 degrees for the original nominal policy. Recovery must be a
separate routed expert/residual, not mixed into one unconditional decoder.

The isolated-expert pilot supports that factorization. On 628 held-out
synthetic recovery states, the untouched nominal policy scores 15.90 cm/68.98
degrees, while a recovery-only relative-flow expert scores 8.54 cm/58.45
degrees. Correct geometry is 0.12 cm/1.68 degrees better than zero input and
0.19 cm/0.14 degrees better than a shuffled relation. Most of the recovery
gain therefore comes from the routed recovery objective; the present geometry
token adds a smaller causal effect. These are synthetic-state results, not yet
closed-loop task success.

A stricter action-equivalent bottleneck is substantially better. A 39,460
parameter MLP sees only `[relative translation, relative SO(3) rotvec,
active-arm bit]` and predicts the six-step active-arm recovery twist. On the
same 628 object-disjoint recovery states:

| Relative-twist recovery input | Translation | Rotation |
|---|---:|---:|
| Correct relative SE(3) | **4.868 cm** | **17.100 deg** |
| Zero relative SE(3) | 15.551 cm | 69.279 deg |
| Wrong held-out-shoe relation | 20.167 cm | 95.873 deg |

This is the strongest current evidence that geometry should be converted into
an action-equivalent error representation rather than appended to a generic
point-cloud policy. The head remains routed: it controls pose only while the
error is large, holds the grasp closed, and returns to the untouched nominal
policy for contact and release.

The first recovery labels rejoined the next nominal demonstration window while
the input described the final object goal. Their translation directions can
conflict. Rebuilding labels by applying the final object SE(3) to the currently
grasped EEF improves the same compact head to 3.416 cm/3.752 degrees; zero and
wrong-relation controls are 16.604/77.819 and 23.266/105.830. In closed loop,
the older mismatched-label head reaches 7.113 degrees minimum rotation but
drifts to 16.754 cm minimum translation, confirming the label mismatch rather
than an orientation-perception failure.

Adding the raw grasp-relation vector to the MLP does not generalize across
held-out shoes (4.325 cm/7.831 degrees). The rigid composition itself should
not be relearned: object relative transform plus the world-frame EEF lever arm
analytically determines the EEF transform. On all 628 goal-aligned recovery
samples this analytic action-flow reproduces the recovery labels to numerical
precision: endpoint error is `2.4e-6 cm / 3.6e-6 degrees`. It is therefore the
retained motion prior; learning is reserved for relation scoring, bounded
residuals, and contact/release.

For relative object motion `(R, t)` and the current world-frame lever arm
`l = p_eef - p_object`, the action bottleneck is
`delta_p_eef = t + (R - I)l`, `delta_R_eef = log(R)`. The implementation is
in `rigid_action_flow.py`, its object-disjoint evaluator is
`evaluate_analytic_recovery.py`, and the mapping is guarded by exact rigid-body
unit tests.

## First camera-only closed-loop validation

On fixed unseen-shoe seed 100002, all variants use the same camera point
clouds and initial placement. The original local policies fail, while the
analytic action bottleneck succeeds:

| Action route | Success | Minimum translation | Minimum rotation |
|---|---:|---:|---:|
| Frozen local diffusion | 0/1 | 1.819 cm | 75.996 deg |
| Camera translation adapter | 0/1 | 3.222 cm | 77.541 deg |
| Camera full-SE(3) adapter | 0/1 | 4.236 cm | 105.634 deg |
| Routed learned diffusion recovery | 0/1 | 15.340 cm | 101.490 deg |
| Routed learned relative-twist recovery | 0/1 | 16.754 cm | **7.113 deg** |
| **Routed analytic rigid action-flow** | **1/1** | **1.832 cm** | **3.677 deg** |

The analytic route succeeds at policy step 98 after 93 recovery and five
nominal calls. It uses no simulator object pose: the target frame comes from B
XYZRGB, the object-in-gripper relation is selected from A XYZ and EEF
proprioception, and the nominal diffusion handles the final release.

A representation-content intervention keeps the analytic controller fixed but
forces the worst-fitting training relation. It fails 0/1 after 240 steps with
136.124-degree minimum and 179.021-degree final rotation error. Thus the
controller alone is not sufficient; the selected object representation is
causally necessary for the success.

The follow-up locks one configuration before replaying five diagnostic seeds:
relation-cost acceptance at `2.5e-4 m^2`, recovery exit at `1.5 cm / 5 deg`,
and PCA fallback for rejected relations. The 1.5-cm threshold is not fit to
task success: it is the independently measured fold-0 held-object camera-error
p90 (1.533 cm). The locked result is:

| Seed | Relation accepted | Result | Policy steps |
|---:|:---:|:---:|---:|
| 100002 | yes | success | 38 |
| 100003 | yes | success | 28 |
| 300001 | no, PCA fallback | failure | 240 |
| 300004 | yes | success | 42 |
| 400002 | yes | success | 33 |

This challenge-set pilot is 4/5 overall, with 80% relation coverage and 4/4
conditional success on accepted relations. It is not a random-seed success-rate
estimate. The rejected scene also shows that deterministic PCA is not a safe
substitute: the fallback remains in recovery for all 240 calls and fails.

Seed 300001 cleanly localizes the remaining bottleneck. The online Chamfer
selector chooses source 39, whose simulator-score-only endpoint is
10.251 cm/178.291 degrees. The same camera-visible candidate bank contains
source 14 at 0.610 cm/1.791 degrees. Forcing this oracle-scored candidate into
the otherwise identical camera/analytic runtime succeeds in 39 policy steps.
Thus candidate generation and action execution both cover the scene; online
hypothesis ranking is the measured failure. This is the component where a
task-refined PTv3/NDF/UTONIA scorer is justified.

## Fixed random-seed paired pilot

A subsequent pilot froze the same fold-0 camera/analytic configuration before
evaluating ten replayable seeds on held-out shoes 0 and 5. All episodes begin
after the expert grasp (`policy_start_phase: placement`); this is therefore a
placement-and-release result, not yet a full grasp-to-place success rate. The
comparison uses identical simulator seeds for the geometry route and the
untouched nominal interaction-diffusion policy:

| Paired condition | Success |
|---|---:|
| Camera relation + analytic rigid action flow + nominal contact head | **6/10** |
| Nominal interaction diffusion without geometry recovery | **0/10** |

The paired success-rate gain is `+0.60`, with a seed bootstrap 95% interval
of `[+0.30,+0.90]`. There are six geometry-only successes and no
baseline-only successes; the exact two-sided paired sign test is `p=0.03125`.
The sample is deliberately small, so this is a positive expansion gate rather
than the final paper estimate.

The accepted-relation coverage is 8/10 and its conditional success is 6/8.
The two rejected scenes (seeds 63 and 84) both fail under the propagated PCA
fallback, identifying candidate-library coverage/ranking as one remaining
bottleneck. The two accepted-relation failures are structurally different:

- seed 0 finishes only 0.753 mm outside the task's x tolerance, a terminal
  residual/contact-refinement failure;
- seed 48 reaches task-valid pose and ramp contact, but never opens the
  gripper, a recovery-to-release failure.

Threshold probes on seed 48 reject a tuning-only explanation. Raising the
translation exit threshold to 2.0 or 2.5 cm still fails and produces 112/128
or 103/137 recovery/nominal calls, respectively: the controllers oscillate.
A one-way handoff removes oscillation (44 recovery then 196 nominal calls) but
also fails; the unconstrained nominal policy drives the already aligned object
back to 6.6 cm/41.1 degrees of final task error and still does not release.
Thus the retained interface needs a geometric motion safety envelope plus an
independent contact/release decision. It must not hand full motion authority
back to the local behavior-cloning policy.

Camera occlusion is no longer treated as a fresh pose-estimation problem. The
runtime caches the last visible A/B clouds, initializes the object-to-EEF
relation once, and propagates it with EEF proprioception when the grasped
object is occluded. This prevents rejected-relation episodes from crashing,
although it does not make PCA fallback geometrically accurate.

## 100-demonstration verifier and release follow-up

An additional 100 demonstrations were collected and all 100 passed structural
validation. Blindly enlarging the raw relation library did not improve the
locked ten-seed result: the camera/analytic route with the larger library and
no new release logic remains 6/10, accepts 8/10 relations, and succeeds on
6/8 accepted relations. More data also exposes a symmetry long tail. During
the held-object relation phase, mean/p90 relative rotation error is
23.85/178.54 degrees, compared with 5.18/19.17 degrees for the smaller pilot
library. Library size is therefore not a substitute for explicit multimodal
hypotheses and symmetry-aware ranking.

A raw-XYZ point-token verifier was trained with object-disjoint shoe splits
and three model seeds. The conservatively calibrated ensemble changes only
high-confidence failures. On validation it raises the geometric proposal's
strict 2-cm/15-degree coverage from 80/81 to 81/81; on held-out test shoes it
raises 23/24 to 24/24. However, it vetoes none of the ten closed-loop pilot
relations. It is safe on this pilot but has no measured closed-loop
contribution yet; the remaining seeds 63/84 are candidate-generation or
abstention failures, not verifier corrections.

The accepted failures reveal a separate phase-transition problem. A naive
release gate that opens after only three aligned calls is a negative control:
it fixes seeds 43/48 but regresses seeds 0/7, leaving the total at 6/10. The
retained state machine releases only when all of the following hold:

1. the grasp relation is accepted;
2. the camera estimate is inside 2 cm and 5 degrees;
3. the analytic router is still outside its stricter 1.5-cm/5-degree handoff,
   so the controller is stalled rather than already in the nominal phase;
4. the condition persists for ten consecutive policy calls.

After the gate latches, Cartesian motion is frozen and only the active
gripper is opened. The nominal diffusion branch still executes on every call,
so the intervention changes the phase transition rather than silently changing
the policy execution path.

On the same ten development seeds, one consistent final-code replay gives:

| Development condition | Success | Accepted-relation success | Release latches |
|---|---:|---:|---:|
| Larger library + verifier, no release state machine | 6/10 | 6/8 | 0 |
| Three-call naive release gate | 6/10 | 6/8 | 8 |
| **Ten-call stalled-near-goal release** | **8/10** | **8/8** | 3 |
| Paired nominal interaction diffusion | 0/10 | -- | -- |

Relative to the no-release geometry route, the development gain is +0.20
with bootstrap 95% `[0.00,+0.50]` and exact paired `p=0.5`: it fixes exactly
seeds 43/48 without losing an earlier success, but the sample is far too small
for a statistical claim. Relative to the nominal policy it is +0.80 with
bootstrap 95% `[+0.50,+1.00]` and exact paired `p=0.0078125`.

This 8/10 result is explicitly **not** an unbiased final test: the state
machine was designed after inspecting failure modes on these ten seeds. Its
value is architectural and causal. It shows that propose/verify/transport is
not enough; geometry must also govern a persistent, uncertainty-aware phase
transition. The next paper result must calibrate the dwell rule on validation
shoes, freeze it, and evaluate new unseen seeds without further tuning.

## Validation calibration of the temporal transition

The terminal transition was then calibrated only on the object-disjoint
validation shoes 1 and 6.  The protocol fixes 20 expert-admitted seeds before
comparing conditions.  The no-release geometry route succeeds on 15/20.  The
earlier recovery-only ten-call dwell succeeds on 17/20: it repairs two contact
failures but still misses two scenes whose camera estimate alternates across
the nominal/recovery boundary.

The retained rule filters translation and rotation errors with an EMA
(`alpha=0.5`) and counts ten aligned calls independently of the upstream
nominal/recovery phase.  This is an object-level terminal-state estimate, not
another motion policy.  A three-value EMA screen (`alpha=0.2/0.5` with five or
ten calls) selects `alpha=0.5`, ten calls because it repairs both jitter cases
without regressing four guard scenes.  The relation-cost accept threshold is
calibrated from `2.5e-4` to `3.0e-4 m^2`: the one validation boundary case is
at `2.5928e-4`, while the two original held-out rejected cases remain above
`4.02e-4`.

On the full 20-seed validation set the frozen candidate reaches 20/20 versus
15/20 for the no-release route.  The paired gain is `+0.25`, with seed
bootstrap 95% `[+0.05,+0.45]`: five scenes are candidate-only successes and
none regress.  The exact paired sign-test value is `p=0.0625`.  This is a
calibration/validation result, not a new-test claim, because the parameters
were selected on these validation shoes.  Its purpose is to freeze
`deploy_routed_analytic_verifier_temporal_release_more100.yml` before any
evaluation on shoes 2 and 7.

## Strict replay test and bilateral transition

A protocol audit found that automatic seed discovery and fixed replay were not
action-equivalent. Discovery performs an expert setup/rollout and then a
second policy setup, whereas `skip_expert_check_on_replay` performs one policy
setup. Therefore an integer seed from discovery cannot be paired directly
with a fixed replay log. `eval_policy.py` now records `fixed_seed_replay` in
every manifest, and `summarize_paired_closedloop.py` rejects discovery logs or
mismatched replay protocols. All results below rerun both conditions from the
same seed file with expert replay skipped.

The frozen one-sided EMA transition was first tested on two independent
ten-seed cohorts on shoes 2/7. It succeeds on 16/20, exactly matching the
no-release route at 16/20: all sixteen successes and all four failures are
paired. The paired gain and bootstrap interval are both exactly zero, and the
exact sign-test value is `p=1.0`. Seven release latches change the controller
trajectory but not the task outcome. Thus validation improvement alone did
not generalize; an extra positive release trigger is insufficient.

One accepted failure localizes the interface bug. The nominal policy can open
the gripper before the persistent geometry condition latches. A terminal
module that only says "open now" cannot prevent this earlier transition. The
bilateral version therefore:

1. preserves every nominal/analytic Cartesian command but forces the active
   gripper closed while an accepted relation has not latched;
2. waits for ten EMA-filtered camera estimates inside 2 cm and 5 degrees;
3. after latching, freezes Cartesian motion and opens only the active gripper.

This changes phase authority, not the motion proposal, relation scorer, or
nominal network. On the inspected failure seed 400100022 it changes failure
to success in 88 calls, while two guard seeds remain 2/2. A six-seed screen
on the earlier validation shoes remains 6/6. These are development and
regression checks, not independent rates.

The candidate was then frozen and replayed on ten newly discovered seeds from
the same shoes 2/7, which are fold-0 training identities. Both candidate and
baseline use identical fixed seed files and the one-setup replay protocol:

| Fresh-seed condition | Success | Mean policy calls | Release latches |
|---|---:|---:|---:|
| **Bilateral geometry-owned transition** | **10/10** | **44.4** | **10** |
| No geometry transition | 8/10 | 82.8 | 0 |

The paired gain is `+0.20`, with bootstrap 95% `[0.00,+0.50]`. There are two
candidate-only successes, no baseline-only successes, eight paired successes,
and exact sign-test `p=0.5`. At latch time the candidate's mean filtered error
is 1.489 cm and 2.101 degrees (translation range 1.359--1.691 cm; rotation
range 0.825--4.814 degrees), confirming that the logged transition obeys the
predeclared geometry bounds.

The two repaired seeds are mechanistically informative. Seed 700100029 ends
the baseline 0.053 cm outside the 2-cm translation tolerance with a closed
gripper; the candidate latches at 1.518 cm/0.825 degrees and succeeds in 32
calls. Seed 700100012 leaves the baseline closed at 240 calls with 10.872
degrees best rotation error; holding contact until a valid transition lets the
candidate latch at 1.489 cm/0.854 degrees and succeed in 52 calls. This is a
fresh-seed replication of the bilateral-transition mechanism, but not a
fresh-object claim because shoes 2/7 were used to diagnose the design and are
not held out from the fold-0 relation models.

The same frozen candidate was then replayed on ten new seeds from fold-0 test
shoes 0/5, which are held out from the relation models. It reaches 10/10
(40.1 mean calls) versus 9/10 (70.6 mean calls) without the transition. The
paired gain is `+0.10`, bootstrap 95% `[0.00,+0.30]`, with one candidate-only
success, no baseline-only success, and exact sign-test `p=1.0`. Mean latch
error is 1.565 cm/2.268 degrees at action chunk 32.8. On the repaired seed
800100005, the candidate latches at 1.458 cm/1.874 degrees and succeeds in 44
calls; the baseline remains closed and fails after 240 calls. This confirms
the mechanism on model-held-out object identities, but shoes 0/5 were used in
earlier architecture diagnostics, so it is still not a pristine final test.

The two fresh-seed pilots are therefore reported separately: 10/10 versus
8/10 on fold-0 training identities and 10/10 versus 9/10 on fold-0 test
identities. Their direction is consistent, but their small discordant counts
do not support a significance claim. They justify freezing the interface and
expanding across folds; they do not justify another threshold search.

## Architecture decision

Keep from the current implementation:

1. object-in-gripper SE(3) hypotheses rather than an overconfident single pose;
2. explicit rigid `[current anchor, goal anchor, displacement]` tokens;
3. iterative action generation with zero-initialized residual paths;
4. separate translation, rotation, and gripper confidence gates;
5. zero and wrong-episode causal controls.

Replace in the next action model:

1. The large-error motion branch is the exact grasp-constrained SE(3) flow.
   Any learned branch predicts a bounded correction to this proposal, not an
   unconstrained replacement or the next demonstration delta.
2. A learned rotation residual gets its own SO(3) endpoint/progress loss and
   validation gate.
3. Release is a bilateral persistent contact-state decision. Geometry vetoes
   premature opening as well as issuing the final release. A learned residual
   may replace it only after beating the same calibrated controls.
4. The promoted motion prior is the analytic grasp-constrained flow. The
   compact relative-twist head is retained as a learned ablation, not the main
   route. Diffusion, DP3, or a larger policy may learn only a bounded residual
   around the rigid proposal.

PointTransformerV3 is promoted only for the correspondence/hypothesis scorer,
where the measured pre-grasp 29-degree error leaves room for improvement. It
does not replace the action decoder wholesale. NDF/UTONIA/DINO features enter
the same scorer and must beat raw/PCA under clean-versus-shuffled controls.

## Next paper-scale gates

1. Freeze the bilateral transition exactly as tested. Run 30--50 paired seeds
   on untouched object folds; shoes 2/7 may be used only for replication, not
   further hyperparameter selection.
2. Train a multi-hypothesis relation scorer using raw local geometry,
   task-refined NDF/UTONIA, and PTv3 under identical folds and
   clean/wrong-relation controls. It must reduce abstentions like seeds 63/84,
   not merely improve mean pose error.
3. Train only bounded contact/compliance residuals around the analytic motion;
   compare the current diffusion contact head and a parameter-matched small
   transformer before considering a pi-scale backbone.
4. Run five folds x three seeds offline, then paired closed-loop seeds with
   base, correct geometry, shuffled geometry, oracle relation, and PCA relation.
5. Report hypothesis calibration and reject-option coverage, not just average
   pose error. Low-confidence relations must abstain or fall back explicitly;
   they may not silently receive confidence one.
6. Expand to multiple pose-aware tasks only after correct geometry produces a
   repeat-stable closed-loop advantage on the shoe task.

The defensible paper claim is not "geometry tokens help DP3." The candidate
claim is: **grasp-constrained, uncertainty-aware rigid task flow is an
action-equivalent bottleneck for goal-directed and recovery-capable object
manipulation.** The present evidence validates this bottleneck on a five-scene
diagnostic set, a paired ten-seed fold-0 pilot, and a fresh-seed
bilateral-transition pilot; establishes relation-content causality; and rejects
both the old learned recovery heads and a one-sided release trigger. The latest
10/10 versus 8/10 result is positive seed-level evidence, but it is not
object-independent and is not yet a T-ASE-ready multi-fold, multi-task result.

## Reproducible artifacts

- `camera_relative_pose.py` and `estimate_camera_relative_frames.py`
- `camera_factorized_policy.py` and `factorized_adapter_runtime.py`
- `camera_relative_pose_allfold_summary.json`
- `camera_relative_adapter_allfold_seed0_summary.json`
- `camera_relative_adapter_fold0_3seed_summary.json`
- `interaction_diffusion_factorized_pose6dv2_allfold_seed0_summary.json`
- `action_recovery_coverage_fold0.json`
- `analytic_goal_recovery_fold0.json`
- `camera_routed_analytic_locked_fold0_five_challenge_seeds.json`
- `camera_routed_analytic_locked_fold0_random10_summary.json`
- `camera_routed_analytic_vs_nominal_locked_fold0_random10_paired.json`
- `geometry_marker_more100_seed141_validation.json`
- `candidate_pose_verifier_ensemble_more100_fold0.json`
- `camera_routed_analytic_verifier_more100_locked10_summary.json`
- `camera_routed_analytic_verifier_release_stable3_more100_locked10_summary.json`
- `camera_routed_analytic_stalled_release_more100_locked10_development_summary.json`
- `camera_routed_analytic_stalled_release_vs_no_release_more100_locked10_development_paired.json`
- `camera_routed_analytic_stalled_release_more100_vs_nominal_locked10_development_paired.json`
- `camera_routed_analytic_no_release_more100_validation16_n20_summary.json`
- `camera_routed_analytic_recovery_stall_release10_more100_validation16_n20_summary.json`
- `camera_routed_analytic_temporal_release_fit3e4_a05_s10_more100_validation16_n20_summary.json`
- `camera_routed_analytic_temporal_release_fit3e4_a05_s10_more100_vs_no_release_validation16_n20_paired.json`
- `camera_routed_temporal_release_vs_no_release_test27_n20_paired.json`
- `camera_routed_gated_release_validation16_screen6_summary.json`
- `camera_routed_gated_release_test27_fresh_n10_summary.json`
- `camera_routed_no_release_test27_fresh_n10_summary.json`
- `camera_routed_gated_release_vs_no_release_test27_fresh_n10_paired.json`
- `camera_routed_gated_release_test05_fresh_n10_summary.json`
- `camera_routed_no_release_test05_fresh_n10_summary.json`
- `camera_routed_gated_release_vs_no_release_test05_fresh_n10_paired.json`
- `online_flow_eval_seed300001_oracle_hypotheses.json`
- `rigid_action_flow.py`, `routed_recovery_runtime.py`, and
  `camera_routed_recovery_policy.py`
