# Task Plan: Implement `pour_kettle_mug`

## Goal

Implement a new RoboTwin task `pour_kettle_mug` that uses only the left arm to grasp `009_kettle`, move its spout above a randomly placed `039_mug`, and tilt into a geometry-only pouring pose. The task should integrate with the existing environment loader, language generation pipeline, object-pointcloud placeholder mapping, and eval step-limit config.

## Current Phase

Complete

## Phases

### Phase 1: Planning & Test Scope
- [x] Write and approve a task design spec
- [x] Switch planning files to the new task
- [x] Identify the smallest meaningful failing test for the new task integration
- **Status:** complete

### Phase 2: Red Test
- [x] Add a failing test that proves the new task is not yet fully integrated
- [x] Run the test and confirm it fails for the expected reason
- **Status:** complete

### Phase 3: Implementation
- [x] Add `envs/pour_kettle_mug.py`
- [x] Add `description/task_instruction/pour_kettle_mug.json`
- [x] Register default object-pointcloud placeholder mappings
- [x] Add eval step-limit entry
- **Status:** complete

### Phase 4: Verification
- [x] Re-run the new targeted test and confirm it passes
- [x] Run syntax / import verification on modified Python files
- [x] Summarize residual risks if full simulator rollout is not executed
- **Status:** complete

## Decisions Made

| Decision | Rationale |
|----------|-----------|
| Keep the task single-arm | The user explicitly reverted from dual-arm flow to a left-arm-only pouring task |
| Keep the mug static and untouched | The user explicitly asked to operate only the kettle |
| Use geometry-only pouring success criteria | The user explicitly rejected liquid simulation |
| Keep mug position randomized within a small region | Preserves some generalization without turning the task into a search-heavy benchmark |
| Start with a fixed pour orientation | More stable than solving a fully general spout-to-cup orientation in the first version |

## Key Questions

1. What is the smallest realistic test that proves the new task is wired into the repo correctly?
2. Is the kettle functional point reliable enough as a spout proxy for success checking across all 3 kettle instances?
3. Can the left-arm fixed pour pose remain stable across the chosen kettle and mug spawn ranges?

## Errors Encountered

| Error | Attempt | Resolution |
|-------|---------|------------|
| Existing planning files were still scoped to an unrelated DP3 workstream | 1 | Replaced `task_plan.md` with a task-specific plan for `pour_kettle_mug` |
| New integration test failed before implementation | 1 | Expected; missing env file, instruction template, pointcloud mapping, and eval step limit are the intended red state |

## Notes

- Re-read this file before major edits.
- Log discoveries in `findings.md`.
- Log commands and verification in `progress.md`.
