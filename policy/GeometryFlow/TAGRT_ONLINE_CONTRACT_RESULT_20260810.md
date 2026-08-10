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

## Current claim and promotion gate

Supported now:

> Camera-only, task-aligned NDF relation tokens causally improve a learned DP3
> policy's online object alignment, and a geometry-conditioned hold/release
> factorization prevents visual distribution shift from causing premature
> release.

Not supported yet:

> The current checkpoint completes held-out-object placement reliably.

The narrow-recovery checkpoint is a candidate, not the default deployment
checkpoint. It may be promoted only after it produces simultaneous alignment
and release on the fixed diagnostic seed, then repeats the direction on the
three fixed seeds. Only after that gate should evaluation expand to more shoes,
training seeds, and tasks.
