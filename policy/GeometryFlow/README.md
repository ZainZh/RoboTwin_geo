# GeometryFlow

Current interaction-functional flow evidence and the locked next route are in
[`RESULTS_INTERACTION_FUNCTIONAL_FLOW.md`](RESULTS_INTERACTION_FUNCTIONAL_FLOW.md).
The pre-result hypotheses, endpoints, hashes, and no-retuning rule for the
50-seed fold-0 plus fold-3 expansion are frozen in
[`PAPER_SCALE_PROTOCOL_20260801.md`](PAPER_SCALE_PROTOCOL_20260801.md).
The completed fold-0 result is recorded without post-hoc exclusions in
[`FOLD0_PAPER_SCALE_RESULT_20260801.md`](FOLD0_PAPER_SCALE_RESULT_20260801.md):
48/50 versus 39/50, paired gain `+0.18`, bootstrap 95% `[+0.06,+0.30]`,
and exact paired `p=0.01171875`.
The frozen fold-3 replication reaches 48/50 versus 43/50, and the two-fold
paired summary reaches 96/100 versus 82/100 (`+0.14`, 95% `[+0.07,+0.22]`,
exact `p=0.0005188`).  See
[`FOLD3_PAPER_SCALE_RESULT_20260801.md`](FOLD3_PAPER_SCALE_RESULT_20260801.md)
and [`TWOFOLD_PAPER_SCALE_RESULT_20260801.md`](TWOFOLD_PAPER_SCALE_RESULT_20260801.md).
The exact evidence and partial-run state at the user-requested experiment pause
are preserved in
[`PAUSE_SNAPSHOT_20260801.md`](PAUSE_SNAPSHOT_20260801.md).
The evidence-bounded paper abstract, Introduction, contribution statement, and
plain-language route summary are drafted in
[`PAPER_ABSTRACT_INTRO_DRAFT_20260801.md`](PAPER_ABSTRACT_INTRO_DRAFT_20260801.md).
A concrete Chinese walk-through of one episode, including the exact routing and
release thresholds, is in
[`METHOD_EXPLAINER_CN_20260801.md`](METHOD_EXPLAINER_CN_20260801.md).
The reviewer-facing boundary between supported, preliminary, and missing claims
is tracked in
[`PAPER_CLAIM_EVIDENCE_MATRIX_20260801.md`](PAPER_CLAIM_EVIDENCE_MATRIX_20260801.md).
The frozen next-version design for representation-backed live SE(3) correction,
uncertainty, and slip interventions is in
[`LIVE_VISUAL_CORRECTION_DESIGN_20260801.md`](LIVE_VISUAL_CORRECTION_DESIGN_20260801.md).
The resumed single-GPU experiment order, endpoints, and launch-time hashes are
frozen in [`GPU2_RESUME_PROTOCOL_20260801.md`](GPU2_RESUME_PROTOCOL_20260801.md).
The user-requested nominal-only stop is documented in
[`NOMINAL_FUTILITY_STOP_20260802.md`](NOMINAL_FUTILITY_STOP_20260802.md): the
frozen fold-0 nominal policy completed 0/30 before the remaining queue was
terminated, and the result is retained as a diagnostic rather than a completed
three- or five-fold confirmatory evaluation.

The current research interface is camera point clouds -> confidence-gated
object-in-gripper SE(3) hypothesis -> analytic grasp-constrained rigid action
flow -> confidence-gated contact/release policy. The camera estimator and
routed runtime are implemented in `camera_relative_pose.py`,
`camera_routed_recovery_policy.py`, `rigid_action_flow.py`, and
`routed_recovery_runtime.py`. The old local-BC
diffusion adapters fail the closed-loop recovery gate; the analytic bottleneck
is now the retained motion route.

The current post-grasp pose path propagates a camera-selected object-in-gripper
relation with EEF proprioception.  It is therefore camera-initialized but still
assumes a rigid grasp; a live operated-point-cloud consistency gate is the
explicit next reliability test for detecting slip before release.

On the original frozen fold-0 ten-seed
placement pilot it succeeds 6/10 versus 0/10 for the paired nominal DP3 policy
(paired gain
`+0.60`, bootstrap 95% `[+0.30,+0.90]`, exact paired `p=0.03125`). Accepted
relations succeed 6/8. A subsequent test-informed development pass adds a
stalled-near-goal release state machine and reaches 8/10, with 8/8 accepted
relations succeeding and both rejected PCA fallbacks failing. Because that
release rule was designed after inspecting the original ten seeds, 8/10 is a
development result rather than an unbiased paper statistic. This remains a
placement-only positive expansion gate, not yet a full grasp-to-place,
multi-fold, multi-task result.

The terminal transition was calibrated on object-disjoint validation shoes
1/6. A phase-independent ten-call alignment dwell with camera-error EMA
(`alpha=0.5`) and a `3.0e-4 m^2` relation gate reaches 20/20 on 20 fixed
validation seeds, versus 15/20 without the release transition (paired `+0.25`,
bootstrap 95% `[+0.05,+0.45]`, exact sign `p=0.0625`). After freezing, that
one-sided release rule gives 16/20 versus 16/20 on shoes 2/7: it changes seven
release decisions but produces zero paired success gain. This falsifies the
claim that an additional release trigger alone is a general improvement.

Failure inspection exposed the missing interface: a geometry terminal module
must also veto nominal gripper opening before alignment, rather than only issue
an open command after alignment. The resulting bilateral transition owns the
active gripper until the same persistent camera test latches, then freezes
Cartesian motion and opens it. On a fresh-seed shoes-2/7 pilot it reaches
10/10 versus 8/10 without the transition (paired `+0.20`, bootstrap 95%
`[0.00,+0.50]`, exact sign `p=0.5`; two candidate-only successes and no
regressions). This is new-seed evidence on fold-0 training identities, not
new-object evidence: shoes 2/7 were also used to diagnose the interface. The candidate is frozen in
`deploy_routed_analytic_verifier_temporal_gated_release_more100.yml`.

Without changing that candidate, a second fresh-seed pilot on fold-0 held-out
shoes 0/5 reaches 10/10 versus 9/10 (paired `+0.10`, bootstrap 95%
`[0.00,+0.30]`, exact sign `p=1.0`; one candidate-only success and no
regression). Across the two separately reported pilots the direction is
consistent, but neither ten-seed comparison is statistically decisive and
shoes 0/5 were used during earlier architecture development. The next valid
gate is a pre-registered 30--50-seed cross-fold evaluation, not more tuning on
these objects.

The most important current artifacts are:

- `outputs/geometry_flow/camera_relative_pose_allfold_summary.json`
- `outputs/geometry_flow/camera_relative_adapter_allfold_seed0_summary.json`
- `outputs/geometry_flow/camera_relative_adapter_fold0_3seed_summary.json`
- `outputs/geometry_flow/interaction_diffusion_factorized_pose6dv2_allfold_seed0_summary.json`
- `outputs/geometry_flow/action_recovery_coverage_fold0.json`
- `outputs/geometry_flow/analytic_goal_recovery_fold0.json`
- `outputs/geometry_flow/camera_routed_analytic_locked_fold0_five_challenge_seeds.json`
- `outputs/geometry_flow/camera_routed_analytic_locked_fold0_random10_summary.json`
- `outputs/geometry_flow/camera_routed_analytic_vs_nominal_locked_fold0_random10_paired.json`
- `outputs/geometry_flow/geometry_marker_more100_seed141_validation.json`
- `outputs/geometry_flow/candidate_pose_verifier_ensemble_more100_fold0.json`
- `outputs/geometry_flow/camera_routed_analytic_verifier_more100_locked10_summary.json`
- `outputs/geometry_flow/camera_routed_analytic_stalled_release_more100_locked10_development_summary.json`
- `outputs/geometry_flow/camera_routed_analytic_stalled_release_more100_vs_nominal_locked10_development_paired.json`
- `outputs/geometry_flow/camera_routed_analytic_no_release_more100_validation16_n20_summary.json`
- `outputs/geometry_flow/camera_routed_analytic_temporal_release_fit3e4_a05_s10_more100_validation16_n20_summary.json`
- `outputs/geometry_flow/camera_routed_analytic_temporal_release_fit3e4_a05_s10_more100_vs_no_release_validation16_n20_paired.json`
- `outputs/geometry_flow/camera_routed_temporal_release_vs_no_release_test27_n20_paired.json`
- `outputs/geometry_flow/camera_routed_gated_release_validation16_screen6_summary.json`
- `outputs/geometry_flow/camera_routed_gated_release_test27_fresh_n10_summary.json`
- `outputs/geometry_flow/camera_routed_gated_release_vs_no_release_test27_fresh_n10_paired.json`
- `outputs/geometry_flow/camera_routed_gated_release_test05_fresh_n10_summary.json`
- `outputs/geometry_flow/camera_routed_gated_release_vs_no_release_test05_fresh_n10_paired.json`
- `outputs/geometry_flow/online_flow_eval_seed300001_oracle_hypotheses.json`

The frozen seed-level candidate placement config is
`deploy_routed_analytic_verifier_temporal_gated_release_more100.yml`. Its
one-sided validation-calibrated predecessor is
`deploy_routed_analytic_verifier_temporal_release_more100.yml`. The earlier
test-informed development config is
`deploy_routed_analytic_verifier_release_more100.yml`. The original
locked placement config is `deploy_routed_analytic_recovery.yml`; the
rejected learned comparisons are `deploy_routed_recovery.yml` and
`deploy_routed_twist_recovery.yml`. The earlier paired adapter configs remain
as ablations. New relative-geometry checkpoints must declare
`functional_frame_encoding=relative_se3_pose6d_v2`; the runtime deliberately
rejects the legacy rotation layout.

`GeometryFlow` is the clean implementation of the current research route. It
uses object representations to construct explicit robot–object interaction
targets. It does **not** use generative flow matching.

For grasping, the model predicts a small set of asymmetric virtual-gripper
keypoints. A differentiable Kabsch fit turns those keypoints into a guaranteed
proper SE(3) pose. The asymmetry retains approach and closing-direction signs;
object normalization is applied only to coordinates and never multiplied into
an SE(3) matrix.

Two heads share the same pointwise encoder:

- `direct`: globally pool the object and regress candidate SE(3) poses.
- `keypoint_transport`: attend from labeled gripper-frame queries to object
  points, predict surface anchors plus offsets, then fit SE(3).
- `ndf_directional`: use NDF's equivariant vector to construct the two signed
  grasp-axis candidates, then learn only contact translation and ranking.
- `pca_directional`: architecture-matched raw-geometry control using the
  point-cloud principal axis instead of NDF.

Both support multiple deterministic candidates. Training uses winner-take-all
assignment plus a learned candidate score. Evaluation reports both top-1 and
oracle@K so proposal quality and ranking quality remain distinguishable.

## Inputs and supervision

Policy inputs are the segmented `{A}` camera point cloud, active-arm identity,
and optionally frozen pointwise NDF features. Simulator object pose and shoe ID
are stored only for split construction, diagnostics, and oracle evaluation.
The grasp label is the measured end-effector pose at the first fully closed
gripper frame; across the 50 demonstrations, closing-motion drift is below
5 mm and 0.7 degrees.

The NDF checkpoint at `/shared2/sz/model/ndf/shoe.pth` contains both a 256-D
invariant descriptor and a learned 3-D equivariant direction. Older pointwise
preprocessing retained only the 256-D part. The new adapter stores all 259
channels and exposes `raw`, `ndf_scalar`, `ndf_vector_only`, and `all`
ablations over identical sampled XYZ points.

## Reproduce the initial cache

```bash
CUDA_VISIBLE_DEVICES=0 /root/miniconda3/envs/RoboTwin/bin/python \
  -m policy.GeometryFlow.build_grasp_dataset \
  --episodes 50 --num-points 256 \
  --ndf-checkpoint /shared2/sz/model/ndf/shoe.pth \
  --ndf-device cuda:0 \
  --output outputs/geometry_flow/grasp_initial_ndf_vector_50.npz
```

The first object-disjoint fold uses shoes `2 3 4 7 8 9` for training, `1 6`
for validation, and `0 5` for final testing. No frame or episode from a test
shoe enters training or early stopping.

## Initial evidence

See [RESULTS_INITIAL.md](RESULTS_INITIAL.md) for the five-fold results,
representation-only diagnostics, demonstration-conditioned relation transfer,
and the 90-degree pose-shift stress test. The short conclusion is that explicit
geometric candidates are promising, while the current shoe NDF checkpoint has
not yet beaten simple geometry in-distribution. Its measurable advantage is
rotation robustness, which now needs closed-loop simulator validation.

## Object task-flow screening

The latest strict learned-policy gate adds a no-bypass point-flow + explicit
SE(3) token bottleneck and evaluates matched zero, shuffled, PCA, and oracle
conditions.  It produces strong held-out action-prediction gains but still
fails an oracle-flow closed-loop survival test, localizing the next bottleneck
to recovery-aware sequential learning.  See
[RESULTS_FLOW_SE3_BOTTLENECK_20260802.md](RESULTS_FLOW_SE3_BOTTLENECK_20260802.md).

`build_task_flow_dataset.py` and `train_task_flow_benchmark.py` implement the
new flow-conditioned route. They first test an oracle 3D object task flow
against no-flow and information-equivalent SE(3)-trajectory conditions on five
object-disjoint folds. See [RESULTS_TASK_FLOW.md](RESULTS_TASK_FLOW.md). The
oracle test is positive, especially during held-object placement, but is not a
camera-only or closed-loop claim yet.

## Camera-predicted task flow

The follow-up replaces simulator flow at inference with camera geometry:

```bash
/root/miniconda3/envs/RoboTwin/bin/python \
  -m policy.GeometryFlow.build_flow_prediction_dataset \
  --oracle-flow outputs/geometry_flow/task_flow_oracle_50.npz \
  --ndf-cache outputs/geometry_flow/grasp_initial_ndf_vector_50.npz \
  --output outputs/geometry_flow/task_flow_prediction_50.npz

/root/miniconda3/envs/RoboTwin/bin/python \
  -m policy.GeometryFlow.evaluate_flow_transport \
  --dataset outputs/geometry_flow/task_flow_prediction_50.npz \
  --output-dir outputs/geometry_flow/task_flow_transport_v2
```

`target_frame.py` fits the visible target marker from XYZRGB, and
`evaluate_flow_transport.py` constructs guaranteed-rigid flow with raw, PCA,
or NDF cross-object relations. `evaluate_predicted_flow_actions.py` injects
those held-out flows into the frozen action models. See
[RESULTS_PREDICTED_FLOW.md](RESULTS_PREDICTED_FLOW.md) for the full five-fold,
two-policy-seed result. PCA transport currently retains about 79%/89% of the
oracle endpoint translation/rotation gains; this supports a medium-scale
closed-loop experiment, not yet a final end-to-end success claim.

## Flow-token and UTONIA follow-up

The next controlled pilot compares global pooled flow with local anchor-flow
tokens and replaces NDF with the public UTONIA encoder under the same
object-disjoint protocol. Local tokens do not beat global pooling, while UTONIA
clearly improves rotation over NDF. A guaranteed-rigid factorization using the
UTONIA centroid path and PCA rotation path gives the best predicted-flow
rotation result without a significant translation penalty. See
[RESULTS_FLOW_TOKEN_UTONIA.md](RESULTS_FLOW_TOKEN_UTONIA.md) for metrics,
bootstrap intervals, limitations, and the next experimental gate.

## Zero-initialized DP3 adapter gate

The follow-up DP3 experiment replaces direct flow-token concatenation with a
zero-initialized additive geometry adapter and compares parameter-matched zero,
oracle, fused UTONIA/PCA, and NDF endpoint flows from a common frozen policy
checkpoint. The architecture runs safely, but the two-seed object-disjoint
screen finds no reliable raw-action improvement: oracle is `-0.21%`, UTONIA is
`-0.04%`, and NDF is `+0.31%` relative to zero. See
[RESULTS_DP3_GEOMETRY_ADAPTER.md](RESULTS_DP3_GEOMETRY_ADAPTER.md) for the
protocol, intervals, artifacts, and next recommended manipulation-specific
relation-token route.

## Point-representation token benchmark

The follow-up directly tests whether global pooling explains the weak frozen
feature results. Raw XYZ, RGB, PCA-local coordinates, NDF, and UTONIA use the
same 64-D interface, 657k-parameter grasp head, two seeds, and five
object-disjoint folds. Raw XYZ point tokens improve orientation over raw global
pooling, but frozen NDF and full UTONIA tokens remain substantially worse than
raw tokens. Clean-versus-point-shuffled controls do not establish useful local
correspondence for either encoder. UTONIA's shallow s0 layer is the only
promising exploratory candidate, and still does not beat the absolute raw
token result. See
[RESULTS_POINT_REPRESENTATION_TOKENS.md](RESULTS_POINT_REPRESENTATION_TOKENS.md)
for metrics, shoe-cluster intervals, gated-fusion diagnostics, and the narrowed
task-specific part-refinement route.

A separate three-seed functional-region probe reaches the same decision: Raw
localizes the demonstration-derived grasp region better than NDF/UTONIA, and
clean descriptors do not beat point-shuffled controls. See
[RESULTS_INTERACTION_POINT_PROBE.md](RESULTS_INTERACTION_POINT_PROBE.md).

This point-token result does not evaluate the original NDF implicit grasp
optimizer. That method queries the field at a demonstrated virtual-gripper
point set, uses the default multi-layer `acts=all` activation, and searches
over SE(3) at inference. The next NDF comparison must port that exact candidate
generator to the same partial-camera, held-out-object protocol before making a
claim about NDF itself.
