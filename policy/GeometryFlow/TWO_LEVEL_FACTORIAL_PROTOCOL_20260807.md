# Two-level Task-Aligned Geometric Relation Token factorial protocol

Date frozen: 2026-08-07, before training the missing matched conditions.

## Question

Which part of the two-level representation causes the closed-loop improvement:
plain point tokens, learned local interaction relations, auxiliary rigid-flow
supervision, or the explicit global NDF functional SE(3) relation?

## Fixed fold and data contract

- Fold 1 object split and the already frozen nominal/recovery camera datasets.
- Policy phase: post-grasp relation/placement.
- Six-step, bimanual 14-D EEF action chunk; execute one step and reobserve.
- World-frame robot state/actions for every condition.
- Three independent optimization seeds: 0, 1, and 2.
- Nominal training followed by the same joint nominal/recovery fine-tuning.
- Checkpoint selection uses only the original fold-1 validation objective.
- No external-shoe result may select a checkpoint, epoch, width, or loss weight.

## Frozen hierarchy

1. `point128`: raw point-token policy with the same 128-D width.
2. `pointcap148`: raw point-token capacity control (1.112 M parameters).
3. `local_relation`: 16 learned A/B anchor-relation tokens, no flow loss.
4. `local_flow`: the same relation model plus correct auxiliary flow loss.
5. `local_flow_zero_global`: the existing independently trained, exact-zero
   global-frame policy (same capacity as the full model).
6. `full_two_level`: the existing local-flow policy plus the correct camera NDF
   relative functional-frame token.

Conditions 3 and 4 have identical parameter count. Conditions 5 and 6 have
identical parameter count, architecture, observations, and action supervision.
The wider point baseline is secondary because width and representation topology
cannot both be held identical.

## Evaluation

- Development contract check: the frozen fold-1 pose-correction manifest.
- Mechanistic external evaluation: the already frozen 32-state, 16-GSO-shoe
  manifest used by the confirmatory clean-vs-zero blind test.
- The external set is no longer called a new blind test for these newly trained
  ablations. It is a frozen post-hoc factorial evaluation; the original
  clean-vs-zero comparison remains the preregistered confirmatory result.
- Every variant starts from the same in-process serialized SAPIEN snapshot.
- Online inputs are camera A/B point clouds plus proprioception only.
- Simulator object poses are metrics/success labels only.
- Six policy calls, one executed step per call, target latched from three early
  observations, 4 cm / 15 degree success threshold.

## Primary mechanistic comparisons

- `local_relation - point128`: value of task-conditioned A/B anchor relations.
- `local_flow - local_relation`: value of correct auxiliary flow supervision.
- `full_two_level - local_flow_zero_global`: causal value of the global
  functional relation under exact capacity matching.
- `full_two_level - local_flow`: practical value of the complete second level.

Success rate is primary. Final translation and SO(3) error are fixed secondary
metrics. Results are reported per seed and per object; no failed object or
rollout is removed after evaluation.
