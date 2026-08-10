# Context-gated TAGRT recovery: development and frozen blind results

Date: 2026-08-10

## Frozen policy contract

- Camera-segmented A/B point clouds only at deployment.
- Shoe-specific NDF ensemble estimates the current-to-goal functional frame.
- TAGRT local relation tokens `[256, 4]`, global remaining-SE(3) token `[9]`,
  and 20D robot state condition a DP3 six-action EEF chunk policy.
- No simulator pose, stage label, analytic planner, or IK residual is used as a
  policy input.

The recovery experiment adds a learned context gate and a learned task-aligned
SE(3) action decoder. Its deployed EMA preserves every non-gate parameter of
the previously validated decoder checkpoint exactly. A two-consecutive-chunk
confidence requirement and a two-chunk cooldown debounce repeated recovery
requests; suppressed chunks execute the original DP3 motion and retain the
learned gripper output.

## Artifacts

- Base DP3 checkpoint:
  `/shared2/sz/TAGRT-v1/models/dp3/place_shoe_geometry_marker-tagrt_closedterminalrelease_invariant_contractfixed_seed0-10192_0/30.ckpt`
- Frozen recovery checkpoint:
  `/shared2/sz/TAGRT-v1/models/dp3/place_shoe_geometry_marker-tagrt_multiaxis_contextgate_olddecoder_seed0-12560_0/10.ckpt`
- Recovery dataset:
  `outputs/geometry_flow/tagrt_dp3_online_contract_fix/task_flow_fold0_ndf_camera_train_multiaxis_gated_recovery.npz`
- Frozen deployment config:
  `policy/GeometryFlow/deploy_tagrt_dp3_contextgate_olddecoder_90.yml`

## Development seeds

Fixed expert-feasible seeds: 100000, 100002, 100003; 90 policy steps.

| Method | 100000 | 100002 | 100003 | Total |
|---|---:|---:|---:|---:|
| Original DP3 | fail | success | fail | 1/3 |
| Failure-matched geometry-only gate | fail | success | success | 2/3 |
| Broad-axis joint head training | fail | success | fail | 1/3 |
| Frozen old decoder + context gate + temporal debounce | fail | success | success | 2/3 |

For seed 100003, the temporal gate suppressed the first request and accepted
the second persistent request. The learned recovery changed privileged logging
metrics from 10.66 cm / 42.8 degrees to 4.85 cm / 14.4 degrees. The nominal DP3
then reached 2.76 cm / 4.4 degrees and released, succeeding at policy step 55.
Simulator metrics in this paragraph are post-action diagnostics only.

Result log:
`eval_result/place_shoe_geometry_marker/GeometryFlow.deploy_tagrt_dp3/demo_clean_3d_object_pc_geometry_marker_unseen05/tagrt-contextgate-olddecoder-epoch10-threshold07-persistence2-cooldown2-dev3-90/standard/2026-08-10 16:15:39/episodes.jsonl`

## Frozen blind cohort 300xxx

Expert-only seed discovery selected 300000, 300003, 300006, 300007, and
300008. These seeds were not used to train or tune the frozen model.

| Method | 300000 | 300003 | 300006 | 300007 | 300008 | Total |
|---|---:|---:|---:|---:|---:|---:|
| Frozen recovery method | fail | fail | success | fail | fail | 1/5 |
| Original DP3 | fail | fail | success | fail | fail | 1/5 |

The recovery head was never requested on any blind trajectory. All 417
executed 14D actions were bit-identical to the baseline (maximum absolute
difference 0). The four failures ended mainly in a low-rotation translation
stall around 4.25--5.65 cm, outside the 18--45 degree recovery support.

Method log:
`eval_result/place_shoe_geometry_marker/GeometryFlow.deploy_tagrt_dp3/demo_clean_3d_object_pc_geometry_marker_unseen05/tagrt-contextgate-olddecoder-frozen-blind5-300000-90/standard/2026-08-10 16:24:06/episodes.jsonl`

Baseline log:
`eval_result/place_shoe_geometry_marker/GeometryFlow.deploy_tagrt_dp3/demo_clean_3d_object_pc_geometry_marker_unseen05/tagrt-contractfix-baseline-frozen-blind5-300000-90/standard/2026-08-10 16:35:12/episodes.jsonl`

## Current conclusion

The experiment supports a narrow claim: task-aligned geometric decoding can
recover a persistent high-rotation failure without changing nominal successful
behavior. It does **not** yet support a general success-rate improvement claim.
The new blind cohort shows that the remaining dominant failure is low-rotation
translation stalling, which the current recovery supervision and support do not
cover.

The next method revision stays on the same geometry-conditioned policy line:
add object-disjoint, uniformly directed translation-stall recovery supervision
on training shoes only, freeze the design, and evaluate once on a new 400xxx
expert-feasible blind cohort. The 300xxx cohort is diagnostic from this point
and cannot be reused as a blind test.

## Low-rotation translation follow-up

The first strict translation-only follow-up exposed two distinct problems.
The dataset generator originally constrained the physical object perturbation,
not the resulting remaining relation. It was corrected to sample the desired
remaining translation directly. The corrected training set has 1,120 synthetic
rows from training shoes 2/3/4/7/8/9, donor rotation below 2 degrees, and an
exact 3.006--7.990 cm remaining-translation range. Shoe 5 remains fully held
out and contributes 136 evaluation-only rows.

The old global-context translation MLP fitted the synthetic training and an
unseen-vector split, but generalized poorly to held-out shoe 5: 2.810 cm mean
chunk-endpoint error and 0.874 endpoint-direction cosine. In the targeted
seed-100003 rollout it predicted the wrong signed correction and increased the
true translation error from 3.65 cm to 7.22 cm. This head is rejected.

The accompanying representation audit found that the existing `[256, 4]`
"local" token is a concatenation of source and target coordinates plus a role
bit, followed by independent PointNet/max pooling. It contains neither anchor
correspondences nor pointwise displacement. Therefore it should not be called
a point-flow token.

## Explicit anchor-flow token and constrained decoder

A deployable `[128, 9]` token was added for each current object anchor:
anchor-to-left-EEF offset, anchor-to-right-EEF offset, and remaining rigid flow
at the anchor. All three terms use the camera-estimated NDF functional relation
and ordinary robot proprioception; simulator object pose remains logging-only.

On the 136 shoe-5 held-out rows, a ridge probe using this exact token reached
1.348 cm mean endpoint error and 0.975 direction cosine, versus 2.417 cm and
0.917 for the raw global relation. A generic neural point-pool/chunk decoder
did not retain this advantage (4.112 cm), and endpoint-loss fine-tuning alone
only reached 3.256 cm. This rules out the claim that token construction alone
is sufficient.

The retained decoder predicts one active-arm endpoint plus zero-sum temporal
residuals, so the six predicted translations sum exactly to the learned
geometry-conditioned endpoint. Its shoe-5 results are:

| Epoch | Endpoint error mean | Endpoint error p95 | Cosine mean | Cosine p05 |
|---:|---:|---:|---:|---:|
| 5 | 1.715 cm | 2.498 cm | 0.9654 | 0.8664 |
| 10 | 1.276 cm | 2.034 cm | 0.9831 | 0.9348 |
| 15 | 1.083 cm | 1.766 cm | 0.9887 | 0.9562 |
| 20 | 0.988 cm | 1.590 cm | 0.9909 | 0.9633 |
| 25 | 0.942 cm | 1.506 cm | 0.9917 | 0.9684 |
| 30 | **0.924 cm** | **1.462 cm** | **0.9920** | **0.9701** |

The gate at threshold 0.8 retains 82.4% recall on the shoe-5 translation rows
with a 0.277% false-positive rate on normal test rows. Epoch 30 therefore
passes the predeclared offline requirement of beating the old 2.810 cm head and
is the only low-translation candidate promoted to a targeted closed-loop
diagnostic. It is not yet a frozen success-rate result. The local H200 host
cannot launch SAPIEN's required Vulkan renderer, so that single diagnostic is
scheduled on the isolated RTX 4090 simulator host before any 400xxx blind run.

## Targeted RTX-4090 closed-loop diagnosis

The epoch-30 translation-only anchor-flow candidate failed on the pre-registered
development seed 100003. The high-rotation decoder remained effective, moving
the true state from 5.57 cm / 39.0 degrees to 3.57 cm / 16.5 degrees. The
low-rotation head then fired at a camera estimate of 3.13 cm / 17.8 degrees and
increased the true error to 4.47 cm / 19.2 degrees. Its active-arm endpoint was
`[+2.14,+2.50,-2.64] cm`. The old retention head simultaneously released after
the first action of the chunk. Final performance was 5.48 cm / 17.99 degrees,
0/1 success.

The retention implementation is a learned six-step head, not a one-step rule.
It had been frozen during the translation-head experiment and had never learned
the new recovery-state distribution. Retention-only training changed exactly
six tensors under `binary_gripper_retention_head` and no other model tensor.
Epoch 4 was selected before another rollout: on 136 held-out shoe-5 rows, 97.1%
predicted closed for all six steps and the per-step p05 probabilities remained
0.538--0.591.

The translation distribution was then expanded from donor rotation below 2
degrees to a 0--18 degree residual-rotation shell, still using training shoes
2/3/4/7/8/9 only. A new held-out shoe-5 set used the same shell. Translation-
only decoding reached 1.618 cm endpoint error and 0.948 cosine offline, but it
again failed online: its `[+1.66,+6.16,-2.81] cm` endpoint changed the true
state from 3.59 cm / 16.3 degrees to 6.79 cm / 18.4 degrees. Retention was fixed
and stayed closed for the full 90 steps, so release no longer explains the
failure.

This identifies a supervision/deployment contract error. The synthetic
geometry-goal translation label contains the grasp-point orbit paired with its
rotation label, while the deployed translation-only branch discards that
rotation and retains nominal DP3 rotation. A high-rotation + learned-retention
ablation (translation branch disabled) ended at 3.87 cm / 7.24 degrees at 90
steps and 4.78 cm / 10.91 degrees at 120 steps; extra horizon did not resolve
the nominal translation drift. The factorized low-rotation translation-only
branch is therefore rejected.

## Paired joint-SE(3) anchor-flow decoder

The corrected learned decoder retains the same camera NDF/TAGRT inputs and gate
but emits a paired active-arm `[translation, rotvec]` chunk. One head produces
`[B, 6, 2, 6]`; its temporal residuals are zero-sum, so the chunk sum exactly
matches the learned 6D endpoint. At deployment, a low-rotation gate activation
replaces both translation and rotation from this same prediction. No planner,
simulator-pose input, stage label, or analytic residual was added.

Held-out shoe-5 results on the 0--18 degree cascade distribution are:

| Epoch | Translation mean / p95 | Direction cosine | Rotation mean / p95 |
|---:|---:|---:|---:|
| 5 | 2.210 / 3.522 cm | 0.9227 | 7.07 / 12.22 deg |
| 10 | 1.849 / 3.075 cm | 0.9422 | 5.97 / 10.44 deg |
| 15 | 1.652 / 2.703 cm | 0.9537 | 5.26 / 9.17 deg |
| 20 | 1.570 / 2.555 cm | 0.9584 | 4.87 / 8.58 deg |
| 25 | 1.541 / 2.500 cm | 0.9599 | 4.65 / 8.31 deg |
| 30 | **1.532 / 2.479 cm** | **0.9602** | **4.58 / 8.29 deg** |

Epoch 30 is the sole joint-SE(3) candidate promoted to the same seed-100003
targeted diagnostic. No new blind cohort may be run until that diagnostic
shows that offline paired-action consistency transfers to closed loop.
