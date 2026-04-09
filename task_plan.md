# Task Plan: Validate DP3 SAM3 Object-PCD Training Path

## Goal
Determine whether `policy/DP3/train_objpc_sam3.sh` can run correctly in the current repository, identify any blockers in its preprocessing or training path, and estimate the per-frame time needed to fuse object point clouds when using the SAM3 segment-and-project pipeline.

## Current Phase
Phase 5

## Phases

### Phase 1: Requirements & Discovery
- [x] Confirm planning workflow is requested
- [x] Check whether planning files already exist
- [x] Record initialization findings in `findings.md`
- [x] Capture the concrete user task to execute next
- [x] Trace `train_objpc_sam3.sh` dependencies and runtime assumptions
- [x] Identify the object-PCD fusion implementation and available benchmark path
- **Status:** complete

### Phase 2: Planning & Structure
- [x] Define how to validate the shell script without launching a full training run
- [x] Define how to measure or approximate per-frame object-PCD fusion cost
- [x] Record decisions and rationale
- **Status:** complete

### Phase 3: Implementation
- [x] Run syntax and dependency checks on the training/preprocessing path
- [x] Execute targeted dry-run or benchmark commands where feasible
- [x] Patch issues if a clear root cause is found
- **Status:** complete

### Phase 4: Testing & Verification
- [x] Verify script/config references resolve correctly
- [x] Record benchmark or timing evidence for fusion cost
- [x] Confirm remaining risks and untested assumptions
- **Status:** complete

### Phase 5: Delivery
- [ ] Review outputs and remaining risks
- [ ] Summarize changes and verification
- [ ] Deliver results to the user
- **Status:** in_progress

## Key Questions
1. Does `policy/DP3/train_objpc_sam3.sh` invoke valid preprocessing and training targets with the current repo layout?
2. What runtime dependencies or missing assets can still prevent the script from completing?
3. Where is per-frame SAM3 object-PCD fusion performed, and what evidence can be gathered for its runtime cost?

## Decisions Made
| Decision | Rationale |
|----------|-----------|
| Initialize planning files in the project root | Matches the `planning-with-files` workflow and keeps state local to the repo |
| Keep external or untrusted content out of `task_plan.md` | The skill warns that `task_plan.md` is high-sensitivity because it is repeatedly re-read |
| Investigate the SAM3 training path before changing code | The user first asked whether the current script can work, which requires root-cause-oriented validation instead of speculative edits |
| Use the dedicated SAM3 benchmark as the runtime probe | It exercises the same segment-and-project extraction core without paying the cost of a full DP3 training run |
| Make the SAM3 shell entrypoints self-locating | The original scripts were fragile because they silently depended on being launched from `policy/DP3` |

## Errors Encountered
| Error | Attempt | Resolution |
|-------|---------|------------|
| None so far | 0 | N/A |
| `NameError: name 'Path' is not defined` in `sam3_pointcloud_utils.py` | 1 | Added the missing `from pathlib import Path` import |
| `ModuleNotFoundError: No module named 'ftfy'` during SAM3 model load | 1 | Installed `ftfy` into the active `RoboTwin` environment |
| `TypeError: 'SimpleTokenizer' object is not callable` in Ultralytics SAM3 text encoder | 1 | Added a compatibility shim for the local CLIP tokenizer |
| `AssertionError` on bbox prompt shape `[1, 6]` | 1 | Normalized cached SAM3 boxes to XYXY (4 values) |

## Notes
- Re-read this file before major decisions.
- Write discoveries to `findings.md`.
- Log work and verification in `progress.md`.
- Add new phases if the task grows beyond the current scaffold.
