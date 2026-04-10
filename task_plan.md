# Task Plan: Replace SAM With Sim Segmentation For Fused Object-PCD VLA Train/Eval

## Goal
Use simulator-provided segmentation instead of SAM-based segmentation to build fused object point clouds for DP3 VLA training and evaluation, and provide matching train/eval bash entrypoints for the new pipeline.

## Current Phase
Phase 4

## Phases

### Phase 1: Requirements & Discovery
- [x] Restore planning context from the previous session
- [x] Confirm the new task to execute
- [x] Identify existing train/eval/object-PCD code paths relevant to simulator segmentation
- [x] Confirm which simulator segmentation source should define the new pipeline contract
- **Status:** complete

### Phase 2: Design & Structure
- [x] Compare implementation approaches for replacing SAM with simulator segmentation
- [x] Define the fused object-PCD generation path for offline preprocessing
- [x] Define the online eval observation path
- [x] Define bash/config entrypoints and naming
- **Status:** complete

### Phase 3: Implementation
- [x] Implement simulator-segmentation fused object-PCD extraction utilities
- [x] Add or adapt preprocessing scripts for train-time zarr generation
- [x] Add or adapt eval-time online observation processing
- [x] Create train/eval bash scripts for the new pipeline
- **Status:** complete

### Phase 4: Testing & Verification
- [x] Run syntax and compile checks
- [x] Run a targeted preprocessing dry-run
- [ ] Run a targeted eval dry-run or smoke test
- [ ] Confirm remaining risks and unsupported cases
- **Status:** in_progress

### Phase 5: Delivery
- [ ] Summarize the design, implementation, and verification
- [ ] Call out follow-up work for NDF/semantic integration
- **Status:** pending

## Key Questions
1. Should the new pipeline use simulator `actor_segmentation` or `mesh_segmentation` as the canonical segmentation source?
2. Should the train and eval paths both use fused multi-camera segmentation-project logic, or should one side continue to use existing oracle `object_pointcloud` data?
3. What script/config naming keeps the new path distinct from existing `objpc` and `objpc_sam3` flows?

## Decisions Made
| Decision | Rationale |
|----------|-----------|
| Continue using `planning-with-files` for this new workstream | The user explicitly requested it for a multi-step code change |
| Treat this as a new task rather than extending the old SAM-only investigation plan | The objective has changed from validating SAM to replacing it with simulator segmentation |
| Preserve the prior session details in `findings.md` and `progress.md` | The old investigation still informs the new design and should remain available for reference |
| Use `actor_segmentation` as the simulator segmentation source for the new fused object-PCD pipeline | It matches actor/entity instances and aligns with the existing oracle `object_pointcloud` actor-id filtering path |
| Add a distinct `objpc_actorseg` pipeline instead of overloading the existing `objpc` flow | The existing `objpc` name is already associated with oracle object point clouds and older single-camera segmentation fallback behavior |
| Reuse incremental replay-buffer writing for the new offline preprocessing path | This preserves partial progress during preprocessing and matches the improved SAM3 pipeline robustness |

## Errors Encountered
| Error | Attempt | Resolution |
|-------|---------|------------|
| `task_plan.md` from the previous task no longer matched the new workstream | 1 | Replaced it with a task-specific plan for simulator segmentation |

## Notes
- Re-read this file before major decisions.
- Write discoveries to `findings.md`.
- Log work and verification in `progress.md`.
- Keep untrusted external content out of this file.
