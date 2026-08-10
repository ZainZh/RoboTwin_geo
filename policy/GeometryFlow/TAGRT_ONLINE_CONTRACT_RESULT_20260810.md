# Camera-only TAGRT-DP3 online contract and recovery result (2026-08-10)

## Frozen method boundary

The evaluated method remains a learned geometry-conditioned manipulation
policy:

1. segmented camera point clouds for operated object A and target B;
2. a shoe-category NDF ensemble estimates the current functional frame of A;
3. the latched target marker frame and the NDF frame produce task-aligned local
   point-relation tokens and a global remaining SE(3) token;
4. proprioceptive EEF state and the two geometry levels condition DP3;
5. DP3 predicts six-step, 14D bimanual EEF action chunks.

There is no simulator object pose in the policy input, analytic planner, IK
trajectory, stage label, or analytic residual controller.

## Repaired train/deploy geometry contract

Two deployment mismatches were found and repaired.

- The NDF frame head was trained on deterministically sampled 128-point clouds,
  while deployment passed all 1024 points. On episode 0, the three-model frame
  disagreement changed from 1.575 degrees mean at 128 points to 6.307 degrees
  at 1024 points; the valid fraction fell from 95.5% to 27.3%.
- The DP3 archive used a three-model medoid ensemble with a 7.1408-degree p95
  gate and causal temporal stabilization. Deployment instead used a two-model
  projected mean with a 3.3528-degree p99 gate.

After restoring the exact 128-point sampling and the frozen three-model medoid
ensemble, recomputation over 175 episode-0 frames matched the training archive:

- source-frame rotation difference: 2.92e-5 degrees mean;
- p90 difference: 5.62e-5 degrees;
- maximum difference: 2.53e-4 degrees;
- valid geometry: 97.7% overall and 96.9% post-grasp.

This closes the camera-NDF-to-token implementation contract.

## Paired online causal result

Fixed replay seeds 100000, 100002, and 100003 were evaluated with correct
camera TAGRT and with the same geometry input forced to zero. Both conditions
used the same DP3 checkpoint and initial states.

| Metric | correct TAGRT | zero geometry | relative improvement |
|---|---:|---:|---:|
| final translation | 11.158 cm | 19.778 cm | 43.6% |
| final rotation | 21.760 deg | 114.427 deg | 81.0% |
| best translation | 10.981 cm | 14.287 cm | 23.1% |
| best rotation | 19.690 deg | 79.452 deg | 75.2% |
| best combined score | 3.705 | 7.477 | 50.5% |
| task success | 0/3 | 0/3 | not closed |

The online geometry condition therefore has a large causal alignment effect,
but the end-task success claim is not yet supported.

## Failure localization and rejected fixes

The original hybrid gripper head opened while its own geometry estimate still
reported approximately 12 cm remaining translation. Training post-grasp open
labels never exceeded 6.73 cm. This localized one failure to premature release.

Three diagnostic changes were tested.

1. Matching the online six-step execution cadence in the offline history
   improved far-distance false-open classification but still released at
   11.85 cm online.
2. Concatenating raw global SE(3), translation norm, and confidence into the
   ordinary visual gripper MLP released even earlier, at 9.59 cm. The high
   dimensional visual context continued to dominate online extrapolation.
3. Executing only the first action of each six-step prediction was strongly
   negative. At the same 60 executed actions, receding-one produced
   22.79 cm / 34.04 degrees versus 13.43 cm / 6.53 degrees for executing six.
   Repeatedly taking the low-motion prefix destroys the learned chunk dynamics.

These variants are rejected.

## Learned geometry retention branch

A structured, still fully learned gripper route separates two observable
cases without stage supervision:

- when a gripper is open, the shared visual policy head learns grasp/close;
- when that gripper is already closed, a geometry-only temporal head learns
  hold versus release from the remaining SE(3) relation and confidence.

The branch adds approximately 6k parameters. It contains no hand-coded
geometric release threshold. On the object-disjoint test split its active
gripper accuracy is 96.12%, compared with 94.18% for the retrained ordinary
stride-6 head. Online on seed 100002 it eliminated premature release for all
300 steps. Translation continued to 5.27 cm, but rotation drifted from a best
4.13 degrees to 28.00 degrees. Release was no longer the dominant failure;
off-demonstration motion recovery became the bottleneck.

## Narrow recovery fine-tuning

The legacy recovery augmenter reused donor episode and frame indices for
synthetic rows. This could make duplicated donor frames accidental temporal
neighbors. Dedicated `history_episode_index` and `history_frame_index` arrays
now isolate every synthetic recovery state, while retaining donor metadata for
auditing.

A deliberately narrow pilot added 699 post-grasp samples from training shoes
2, 3, 4, 7, 8, and 9 only:

- 7.3% of the 9579-row mixed archive;
- 1.62 cm mean grasp-preserving translation perturbation;
- 10--35 degree rotation, 22.53 degrees mean;
- geometry-goal recovery labels;
- held-out shoes 0 and 5 were never augmented.

On the untouched object-disjoint nominal test set:

| checkpoint | endpoint translation | endpoint rotation | gripper accuracy |
|---|---:|---:|---:|
| retention before recovery | 2.445 cm | 5.342 deg | 96.12% |
| narrow recovery fine-tune | **1.719 cm** | 5.686 deg | **96.16%** |

On fixed online seed 100002 at 180 actions:

| checkpoint | final translation | minimum translation | final rotation | minimum rotation | best combined |
|---|---:|---:|---:|---:|---:|
| retention before recovery | 7.604 cm | 7.604 cm | 20.984 deg | 4.126 deg | 3.586 |
| narrow recovery fine-tune | **4.257 cm** | **3.945 cm** | 21.276 deg | **3.338 deg** | **1.779** |

The narrow recovery data improves online progress and the best joint alignment,
but does not yet make translation and rotation enter the success region at the
same time. The policy correctly remains closed, and task success is still 0/1.

## Partial-fine-tuning contract repair

Two independent implementation errors made early head-only experiments change
the supposedly frozen motion policy.

- The raw model and deployed EMA model differed. Starting partial training from
  the raw copy allowed EMA updates to replace frozen deployed motion weights.
- The head-only dataset normalizer replaced the checkpoint normalizer, changing
  both normalized observations and unnormalized diffusion actions.

Partial training now initializes the raw copy from deployed EMA weights and
preserves the deployed EMA normalizer. With these repairs, the held-out motion
metrics of a head-only checkpoint reproduce the source checkpoint (about
1.587 cm endpoint translation and 5.27 degrees endpoint rotation).

## Closed-state terminal-release supervision

The first terminal-release archive was also structurally invalid: 75.7% of its
eligible near-goal donors had an active gripper state above 0.5. Those examples
trained the already-open visual gripper route, not the closed-gripper retention
route used online. Aggregate offline release accuracy therefore overstated the
retention branch's ability to release a still-held object.

The corrected archive filters donors to active-gripper state at most 0.1 and
uses 1,312 small grasp-preserving perturbations of 164 valid training-shoe
states. The main grasp head retains its closed-class weight of 3.0; the
retention head uses a separately configurable balanced weight of 1.0.

Closed-state offline audit:

- terminal synthetic step-level release recall: 98.77%;
- terminal synthetic all-six-step release recall: 98.70%;
- nominal pure-hold recall: 90.57%;
- held-out nominal motion: 1.587 cm / 5.261 degrees.

On frozen placement seed 100002, the corrected policy released at action 46,
entered the pose-success region at action 47, first met the raw task criterion
at action 49, and satisfied the 25-step stability criterion at action 55. Final
error was 2.63 cm / 5.87 degrees with ramp contact. This is the first end-task
success obtained by the camera-only learned TAGRT-DP3 route.

At a 60-action budget on fixed seeds 100000, 100002, and 100003, success was
1/3. The two failures did not release prematurely: their active grippers stayed
closed and their best NDF estimates remained outside the terminal region
(4.83 cm / 26.69 degrees and 5.52 cm / 35.99 degrees). The remaining failure is
therefore motion recovery coverage, not the release interface.

Extending the same seeds to 180 actions did not change success (1/3). Seed
100000 reached a best 4.24 cm / 13.38 degrees before drifting; seed 100003 never
improved beyond 5.69 cm / 32.93 degrees. This rejects the hypothesis that the
two failures are only a short evaluation budget.

## Right-arm recovery follow-up

The original 706-sample near-goal rotation archive was arm imbalanced: 519
synthetic rows used the left arm and only 187 used the right. A targeted archive
therefore added 592 grasp-preserving right-arm samples from 74 closed-gripper
training states (eight perturbations each, 12--30 degrees rotation). The token
scale was made an explicit dataset/checkpoint contract so changing augmentation
statistics cannot silently rescale global translation tokens.

Two adaptation scopes were tested from the successful release checkpoint.

1. Full motion-only fine-tuning kept both gripper heads bit-identical but changed
   the other 6.66M parameters. It improved the two right-arm best combined scores
   from 2.74/3.32 to 1.07/1.27, but destroyed the previously successful left-arm
   seed and scored 0/3. This variant is rejected as over-adaptation.
2. TAGRT-adapter-only fine-tuning changed exactly 26 local/global adapter tensors;
   every backbone and gripper tensor remained bit-identical. It preserved the
   successful seed and improved its final rotation to 3.87 degrees. Right-arm
   best scores improved to 1.86/3.06, but overall success remained 1/3.

The arm-balanced data carries useful recovery signal, but random recovery
perturbations and shared adapters are not yet sufficient for robust right-arm
closed-loop convergence. More epochs of the same augmentation are not justified;
the next dataset must be matched to the observed failed trajectory directions.

## Current claim and promotion gate

Supported now:

> Camera-only, task-aligned NDF relation tokens causally improve a learned DP3
> policy's online object alignment. A closed-state geometry-conditioned
> hold/release factorization repairs premature release and can complete a held-
> out-object placement without simulator pose, planning, or stage labels.

Not supported yet:

> The current checkpoint completes held-out-object placement reliably across
> initial states.

The corrected closed-terminal-release checkpoint is the current candidate, not
the default deployment checkpoint. It has crossed the single-seed end-task gate
but not the robustness gate. Longer rollout and generic right-arm augmentation
have now been tested. The next recovery archive should replay the camera-token
neighborhood and signed rotation directions observed in failed seeds 100000 and
100003, while keeping the successful retention decoder frozen. The adapter-only
scope is safer than full-motion fine-tuning and should be the first promotion
candidate for that targeted archive.
