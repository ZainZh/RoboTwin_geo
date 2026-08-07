# GeometryFlow paper claim/evidence matrix

This matrix prevents a component-level result from being promoted into a broader
paper claim.  “Complete” means supported by frozen, held-out evidence available
on 2026-08-01; it does not mean the full T-ASE paper is complete.

| proposed claim | current evidence | status | evidence still required |
|---|---|---|---|
| A bilateral, geometry-owned release transition improves a matched analytic-route placement policy. | Frozen fold0 48/50 vs 39/50; frozen fold3 48/50 vs 43/50; pooled paired +14 pp, exact p=0.0005188. Candidate/baseline share nominal network, relation bank, verifier and analytic motion. | **Strongly supported** | Finish reverse-order repeat and fold2 stress replication to quantify run-order/repeat variability. |
| Camera-selected object-in-gripper relation plus analytic rigid flow improves the nominal learned policy. | One fixed 10-seed pilot: 6/10 vs 0/10, exact paired p=0.03125. Offline analytic mapping reproduces rigid recovery labels numerically. | **Promising, underpowered** | Paper-scale nominal-only evaluation on the same fold0/fold2/fold3 seeds; report nominal→analytic and analytic→verified increments separately. |
| The complete GeometryFlow stack improves held-out object success. | Two held-out folds support the transition component, but the paper-scale baseline already contains relation estimation and analytic flow. | **Not yet isolated** | Complete the queued three-way ablation before assigning a full-stack effect size. |
| Frozen NDF or UTONIA features are better than raw XYZ for this interface. | Generic global/pointwise fusion is unstable; raw XYZ wins existing local probes; clean-vs-shuffled descriptors do not show positive correspondence use. | **Not supported** | Test learned features only as held-out correspondence/SE(3)-verification backends with the downstream interface fixed. |
| GeometryFlow is fully visual after grasp. | Camera point clouds initialize relation/target frame, but current object pose is propagated from EEF proprioception under a rigid-grasp assumption. Two candidate failures expose slip/bias. | **False for v2** | Live fresh-point-cloud SE(3) correction, uncertainty calibration, slip intervention, and recovery—not only failure detection. |
| GeometryFlow generalizes across object instances. | Same-direction gains on two object-disjoint shoe folds (four held-out shoe identities in total). | **Supported within one category/task** | Additional folds and categories; use object/fold as the generalization unit, not only episode-pooled statistics. |
| GeometryFlow is task-general or solves full grasp-to-place. | Current formal evidence starts after an expert-admitted grasp and covers only the placement phase of shoe-to-ramp. | **Not supported** | Add learned grasp-to-place evaluation and at least 3–4 distinct functional-relation tasks such as insertion, hanging, articulated alignment, and bimanual placement. |
| GeometryFlow transfers to real robots. | No current real-robot evidence. | **Not supported** | Real RGB-D perception, calibration/noise tests, several objects per task, and paired or interleaved real trials. |

## Paper-safe central statement today

> Explicit functional geometry is most reliable when it owns a verifiable skill
> transition rather than serving only as an auxiliary action condition.  On two
> held-out shoe folds, a bilateral geometry-owned terminal transition improves an
> otherwise identical camera-relation/analytic-flow policy by 14 percentage
> points over 100 paired placement trials.

## Desired final T-ASE statement

The stronger final statement should be used only if the queued and future evidence
supports it:

> A learned pointwise representation can be converted into an uncertainty-aware
> object relation and rigid action-flow interface that improves learned skills at
> both continuous recovery and discrete phase transitions across objects, tasks,
> perturbations, and real-robot executions.
