# Task Plan: Diagnose NDF Underperformance and Add a Hybrid NDF Pointwise Path

## Goal
Investigate why `train_ndf_pointwise` is flat or slightly worse than `train_objpc`, determine whether the main cause is evaluation setup, training recipe mismatch, incorrect usage, or the representation itself, and implement a new `ndf_pointwise_hybrid` path that preserves baseline merged object-PCD inputs while adding NDF pointwise features.

## Current Phase
Phase 6

## Phases

### Phase 1: Experiment Surface Audit
- [x] Restore planning context from prior work
- [x] Identify the relevant train/eval scripts, configs, preprocessing paths, and runtime observation injection logic
- [x] Confirm whether the current comparison is a pure representation comparison
- **Status:** complete

### Phase 2: Representation Path Comparison
- [x] Compare `objpc`, `ndf_pointwise`, and `semantic_pointwise` dataset construction
- [x] Compare online eval feature injection for NDF vs semantic
- [x] Compare encoder/model structure seen by DP3
- **Status:** complete

### Phase 3: Evidence Gathering
- [x] Inspect actual saved run overrides and dataset metadata for the existing `hanging_mug` experiments
- [x] Check whether the configured NDF checkpoint / backbone pairing is plausible
- [x] Check whether the current benchmark emphasizes novel-object generalization
- **Status:** complete

### Phase 4: Root-Cause Synthesis
- [x] Rank the most likely causes using code and artifact evidence
- [x] Distinguish “benchmark does not show NDF advantage” from “current NDF usage is flawed”
- [x] Propose a minimal next experiment matrix to disambiguate the causes
- **Status:** complete

### Phase 5: Delivery
- [x] Summarize findings with concrete file references
- [x] Call out actionable fixes vs. hypotheses
- **Status:** complete

### Phase 6: Hybrid NDF Implementation
- [x] Write and review a concrete design for `ndf_pointwise_hybrid`
- [x] Add a failing test that captures the intended hybrid observation semantics
- [x] Implement the new preprocess/train/eval/config path without changing existing `ndf_pointwise`
- [x] Verify the new test passes and the new shell/python entrypoints are syntactically valid
- **Status:** complete

## Key Questions
1. Is the current `objpc` vs `semantic_pointwise` vs `ndf_pointwise` comparison controlled, or are training recipes different?
2. Is the current NDF experiment using the intended checkpoint and encoder architecture?
3. Does the current eval setup actually test novel-object generalization, or mainly same-family / same-distribution performance?
4. Does the current NDF pointwise feature field look stable enough for DP3 to exploit with only 50 demos?

## Decisions Made
| Decision | Rationale |
|----------|-----------|
| Treat this as a root-cause investigation rather than jumping to code fixes | The user asked to understand why NDF is underperforming, not just patch scripts blindly |
| Use the saved zarr/checkpoint artifacts as primary evidence | The current raw demo folder is no longer the same complete source dataset used to build the existing 50-episode DP3 artifacts |
| Distinguish benchmark-design explanations from implementation/usage explanations | Both are plausible, and the user explicitly asked about novel-object evaluation vs. misuse vs. weak representation |
| Add a separate `ndf_pointwise_hybrid` path instead of redefining `ndf_pointwise` | This isolates the new experiment and keeps old checkpoints / notes interpretable |

## Errors Encountered
| Error | Attempt | Resolution |
|-------|---------|------------|
| Previous `task_plan.md` still described the actor-segmentation implementation workstream | 1 | Replaced it with a task-specific investigation plan for the NDF underperformance diagnosis |

## Notes
- Re-read this file before major decisions.
- Write concrete evidence to `findings.md`.
- Log commands and verification in `progress.md`.
- Keep speculative conclusions out of this file unless they are explicitly marked as hypotheses.
