# Findings & Decisions

## Requirements
- User explicitly invoked `planning-with-files`.
- Initialize persistent planning files in the project root for this repository.
- Keep future discovery, especially external or untrusted content, in this file rather than `task_plan.md`.
- Current repository goal: evaluate whether the 3D representation method improves VLA task success rate, with edits mainly under `policy/DP3`.
- Current near-term task: check whether `policy/DP3/train_objpc_sam3.sh` can work and estimate how much time is needed to fuse object PCDs for one frame using the segment-and-project path.
- Longer-term direction: improve NDF and semantic training separately, then integrate them together and compare against baseline.

## Research Findings
- No existing `task_plan.md`, `findings.md`, or `progress.md` were present in the repository root at initialization.
- The `planning-with-files` templates are available under `/home/zheng/.agents/skills/planning-with-files/templates/`.
- The workflow expects project-local planning files and ongoing updates after each phase or meaningful discovery.
- `policy/DP3/train_objpc_sam3.sh` first preprocesses data into `./data/${task_name}-${task_config}-${expert_data_num}-objpc-sam3.zarr` by calling `process_data_objpc_sam3.sh` if the zarr directory does not already exist.
- The same script then trains inside `policy/DP3/3D-Diffusion-Policy` with `python train_dp3.py --config-name=robot_dp3_objpc.yaml`, overriding `task.dataset.zarr_path` to the generated `-objpc-sam3.zarr` dataset.
- Relevant SAM3 object-PCD code exists under `policy/DP3/scripts/sam3_pointcloud_utils.py` and a dedicated benchmark entrypoint exists at `policy/DP3/scripts/benchmark_objpc_sam3.py`.
- `policy/DP3/process_data_objpc_sam3.sh` runs `python scripts/process_data_objpc_sam3.py` and therefore assumes the caller's working directory is `policy/DP3`.
- `policy/DP3/train_objpc_sam3.sh` also assumes the caller's working directory is `policy/DP3`: it invokes `bash process_data_objpc_sam3.sh`, checks `./data/...`, and runs `cd 3D-Diffusion-Policy`.
- The raw dataset directory `data/hanging_mug/demo_clean_3d_object_pc` exists in the repo root, and prior DP3 zarr outputs already exist under `policy/DP3/data/`.
- The SAM3 model file `/home/zheng/Datasets/sam3/sam3.pt` exists.
- `robot_dp3_objpc_sam3.yaml` differs from `robot_dp3_objpc.yaml` mainly by using `demo_task_objpc_sam3` and adding the `SAM3` tag; the task configs have the same shape metadata and differ mainly in the default dataset suffix (`-objpc.zarr` vs `-objpc-sam3.zarr`).
- The DP3 dataset loader resolves `zarr_path` relative to `diffusion_policy_3d/dataset/robot_dataset.py`, so `../../../data/...` correctly points to `policy/DP3/data`.
- Before patching, running `bash policy/DP3/train_objpc_sam3.sh ...` from the repo root failed immediately because the script could not locate `process_data_objpc_sam3.sh`, could not `cd` into `3D-Diffusion-Policy`, and then tried to open `train_dp3.py` from the wrong directory.
- Runtime blockers found and addressed in the SAM3 extraction path:
- Missing `Path` import in `sam3_pointcloud_utils.py`
- Missing `ftfy` package in the active `RoboTwin` environment
- Local CLIP `SimpleTokenizer` was not callable in the way Ultralytics SAM3 expects
- SAM3 returned 6-value boxes while the tracker's bbox reuse path required 4-value XYXY boxes
- After patching, `policy/DP3/scripts/benchmark_objpc_sam3.py` completed successfully for `hanging_mug / demo_clean_3d_object_pc / episode 0` over 5 frames using prompts `{A}: mug` and `{B}: rack`.
- Measured benchmark timing on this machine:
- `frame_total_ms`: first `28719.51`, mean `22544.42`, p95 `27344.03`
- `frame_total_ms_steady`: mean `21000.64`, p95 `21711.73`
- `{A}_ms`: first `17554.46`, mean `11924.64`, p95 `16277.47`
- `{B}_ms`: first `11165.04`, mean `10619.78`, p95 `11122.44`
- The benchmark ran with `torch.cuda.is_available() == False` and `torch.cuda.device_count() == 0`, so the measured fusion cost is CPU-only.
- There is already a visualization utility at `policy/DP3/scripts/visualize_objpc_zarr.py` that reads a DP3 zarr, optionally overlays per-placeholder point clouds, and can export `.ply` files.
- `open3d` is installed in the active `RoboTwin` environment, so the visualizer can run here.
- Dry-running the visualizer on `policy/DP3/data/hanging_mug-demo_clean_3d_object_pc-50-objpc.zarr` showed that older `objpc` zarrs may only contain the merged `data/point_cloud` dataset and may not include `object_point_cloud_A/B`.
- `process_data_objpc_sam3.py` writes per-placeholder datasets like `object_point_cloud_A` and `object_point_cloud_B` when `--save_placeholder_point_clouds` is enabled, and `process_data_objpc_sam3.sh` already passes that flag.
- The repeated warning `imgsz=[640] must be multiple of max stride 14, updating to [644]` comes from Ultralytics `check_imgsz(...)`.
- Current `SAM3ProjectiveTracker` creates `SAM3SemanticPredictor` with overrides that do not set `imgsz`, so the predictor falls back to `640`.
- For this SAM3 path, stride is `14`, so `640` is invalid and Ultralytics promotes it to `644` while printing a warning on each invocation.
- The correct fix is to set an explicit valid `imgsz` in the predictor overrides, rather than suppressing logs globally.
- Mid-preprocessing, `hanging_mug-demo_clean_3d_object_pc-50-objpc-sam3.zarr` contained only `.zgroup`, `data/.zgroup`, and `meta/.zgroup`. That is expected with the old preprocessing implementation because it buffered arrays in memory and only created `state`, `action`, `point_cloud`, and `episode_ends` after all episodes finished.
- The earlier `KeyError('state')` is consistent with an interrupted preprocessing run: the final zarr directory existed, but required datasets had never been written.
- `train_objpc_sam3.sh` now validates that `data/point_cloud`, `data/state`, `data/action`, and `meta/episode_ends` exist before skipping preprocessing. If not, it invokes preprocessing again and lets the preprocessing script resume or reset as needed.
- A stale `...objpc-sam3.zarr.tmp` from the older temp-publish approach is now treated as legacy state; the current implementation writes directly to the final `.zarr` per episode.
- `robot_dp3_objpc_sam3.yaml` now includes `_self_` in the Hydra defaults list, which removes the missing-`_self_` warning.
- Current recheck: `policy/DP3/data/hanging_mug-demo_clean_3d_object_pc-50-objpc-sam3.zarr` does not exist.
- Current recheck: `policy/DP3/data/hanging_mug-demo_clean_3d_object_pc-50-objpc-sam3.zarr.tmp` exists, but it only contains `.zgroup`, `data/.zgroup`, and `meta/.zgroup`; `data/` and `meta/` still have no arrays.
- This means the preprocess job has not yet reached the final write stage. If the user-facing job is still printing `processing episode: ...`, this state is consistent with the current implementation. If the user-facing job has already exited, then it stopped before publishing any datasets.
- The preprocessing path has now been changed to write incrementally per episode through DP3's `ReplayBuffer.add_episode(...)` instead of buffering all episodes in Python lists.
- New helper module: `policy/DP3/scripts/incremental_objpc_zarr.py`.
- The incremental-write test passes: `policy/DP3/scripts/test_incremental_objpc_zarr.py`.
- `process_data_objpc_sam3.py` now opens or resets a resumable replay buffer at the final `.zarr` path, resumes from `buffer.n_episodes`, and writes metadata JSON after each completed episode.
- `train_objpc_sam3.sh` no longer deletes incomplete SAM3 zarr outputs before preprocessing; it now reports that preprocessing will resume.
- This change reduces risk substantially: if preprocessing stops after episode `k`, episodes `0..k-1` remain persisted in the final `.zarr` and are visible immediately on disk.
- Ultralytics `SAM3VideoSemanticPredictor` resets its source on every `predictor(...)` call because it goes through `BasePredictor.stream_inference(...) -> setup_source(...)`, so tracker state is not preserved automatically across separate calls.
- `SAM3VideoSemanticPredictor.init_state()` asserts `dataset.mode == "video"` and allocates prompt state as `[None] * num_frames`, which means the stock video-tracker path is designed for a finite indexed sequence, not an open-ended live stream.
- The same class does support prompt updates within one loaded sequence through `add_prompt(frame_idx=..., text=..., bboxes=..., labels=...)`; text prompts are global to the sequence, while geometric prompts are frame-indexed.
- `LoadStreams` in Ultralytics uses `mode = "stream"` and can represent effectively infinite sources, but that does not match the stock SAM3 video predictor's `dataset.mode == "video"` requirement.
- Practical implication for this repo: using the SAM3 video tracker per episode or per camera sequence is structurally compatible, but true endless online tracking would require a wrapper, chunked/sliding-window processing, or lower-level state management instead of repeated top-level `predictor(source=..., stream=True)` calls.
- The current eval path already has an online SAM3 hook: `policy/DP3/eval_objpc_sam3.sh` launches `script/eval_policy.py` with `--config_name robot_dp3_objpc_sam3`.
- In eval, `policy/DP3/deploy_policy.py` sets `use_sam3_objpc = "objpc_sam3" in usr_args["config_name"]`, builds one `SAM3ProjectiveTracker`, and stores it on the loaded policy model.
- During each eval step, `encode_obs(...)` calls `extract_placeholder_point_cloud_sam3_online(...)` for every placeholder, merges the per-placeholder clouds, and writes the merged result into `obs["point_cloud"]`.
- The current online SAM3 eval path does not use `SAM3VideoSemanticPredictor`; it uses `SAM3SemanticPredictor` through `SAM3ProjectiveTracker`, with a simple tracking heuristic: reuse the previous frame's bbox when available, otherwise refresh with text, and force text refresh every `sam3_text_refresh_every` frames.
- Eval resets SAM3 state at episode boundaries in `reset_model(...)` by clearing `sam3_tracking_state` and setting `sam3_frame_idx = 0`.
- The eval shell entrypoint exposes the main SAM3 knobs already: `sam3_prompt_map`, `sam3_model`, `sam3_camera_names`, and `sam3_text_refresh_every`.
- The current `hanging_mug/demo_clean_3d_object_pc` dataset stores `head_camera` and `front_camera` RGB frames as JPEG-encoded `240x320` images; direct decode of episode 0 frame 0 confirmed shape `(240, 320, 3)` for both cameras.
- The default camera config in `task_config/_camera_config.yml` uses `D435: w=320, h=240` and also provides `Large_D435: w=640, h=480`.
- `task_config/demo_clean_3d_object_pc.yml` currently sets `head_camera_type: D435` and `wrist_camera_type: D435`, so the low resolution is coming from task config rather than from the SAM3 code path.
- `sam3_pointcloud_utils.py` projects the full 3D scene point cloud into mask pixels using the stored `intrinsic_cv`/`extrinsic_cv` matrices and the mask image size; lower RGB resolution therefore directly reduces mask detail and the number/stability of points selected for the object cloud, especially for small or thin structures.
- Raising RGB resolution should help SAM-based segmentation and 3D selection quality, but the benefit is on the segment-and-project stage, not on the DP3 policy directly. If eval also uses online SAM, the eval environment resolution needs to be raised too; improving only the offline training demos will not fix low-resolution online eval masks.
- Because `Large_D435` is `640x480`, moving from `320x240` to `640x480` increases input pixels by 4x, so SAM runtime and storage/render cost will rise substantially. A small A/B pilot before recollecting the full dataset is the pragmatic next step.

## Technical Decisions
| Decision | Rationale |
|----------|-----------|
| Start with a generic scaffold instead of guessing a task-specific plan | The user has requested the planning workflow but has not yet provided the concrete implementation task |
| Track initialization state immediately | This creates a clean handoff point for the next task in the session |
| Follow the `train_objpc_sam3.sh -> process_data_objpc_sam3.sh -> sam3_pointcloud_utils.py` chain first | This is the shortest path to determine if the script can run and where fusion timing should be measured |
| Prefer a targeted benchmark before full training | The user asked for per-frame fusion time, and the repo already includes a dedicated benchmark path that isolates this cost |
| Patch repo-side SAM3 compatibility issues before judging the pipeline | The initial failures were caused by fixable code and environment mismatches rather than the core segmentation-project idea itself |
| Prefer fixing the warning at the predictor configuration layer | This removes the spam without hiding potentially useful warnings from other parts of the pipeline |
| Replace all-at-end writes with per-episode replay-buffer appends | The user explicitly wants already-computed episodes persisted and visible during long SAM3 preprocessing runs |
| Treat SAM3 video tracking as a per-sequence tool, not a drop-in endless-stream API | The installed Ultralytics implementation requires `dataset.mode == "video"` and initializes state against a fixed `num_frames` |

## Issues Encountered
| Issue | Resolution |
|-------|------------|
| No prior planning context existed in the repo | Created a fresh planning scaffold |
| `train_objpc_sam3.sh` is cwd-sensitive | Treat this as a likely usability bug and verify it through execution checks |
| `sam3_pointcloud_utils.py` referenced `Path` without importing it | Fixed in repo |
| Local CLIP tokenizer did not match Ultralytics SAM3 expectations | Added an in-repo compatibility shim and installed `ftfy` |
| SAM3 bbox reuse path cached 6-value boxes | Trimmed boxes to 4-value XYXY before reuse |
| Incomplete final `.zarr` directories were reused as if preprocessing had succeeded | Added completeness checks in `train_objpc_sam3.sh` and moved SAM3 preprocessing to resumable per-episode persistence |
| All-at-end writes still left the user exposed to total progress loss before the final publish stage | Reworked SAM3 preprocessing to persist each episode incrementally and resume from partial replay-buffer state |

## Resources
- Project root: `/home/zheng/github/RoboTwin_geo`
- Skill file: `/home/zheng/.agents/skills/planning-with-files/SKILL.md`
- Template file: `/home/zheng/.agents/skills/planning-with-files/templates/task_plan.md`
- Template file: `/home/zheng/.agents/skills/planning-with-files/templates/findings.md`
- Template file: `/home/zheng/.agents/skills/planning-with-files/templates/progress.md`
- Training script: `/home/zheng/github/RoboTwin_geo/policy/DP3/train_objpc_sam3.sh`
- Benchmark script: `/home/zheng/github/RoboTwin_geo/policy/DP3/scripts/benchmark_objpc_sam3.py`
- Fusion utilities: `/home/zheng/github/RoboTwin_geo/policy/DP3/scripts/sam3_pointcloud_utils.py`
- Raw dataset example: `/home/zheng/github/RoboTwin_geo/data/hanging_mug/demo_clean_3d_object_pc`
- Existing DP3 outputs: `/home/zheng/github/RoboTwin_geo/policy/DP3/data`
- SAM3 weights: `/home/zheng/Datasets/sam3/sam3.pt`
- Dataset loader: `/home/zheng/github/RoboTwin_geo/policy/DP3/3D-Diffusion-Policy/diffusion_policy_3d/dataset/robot_dataset.py`
- Visualizer: `/home/zheng/github/RoboTwin_geo/policy/DP3/scripts/visualize_objpc_zarr.py`
- Incremental zarr helper: `/home/zheng/github/RoboTwin_geo/policy/DP3/scripts/incremental_objpc_zarr.py`
- Incremental zarr test: `/home/zheng/github/RoboTwin_geo/policy/DP3/scripts/test_incremental_objpc_zarr.py`

## Visual/Browser Findings
- None yet.

---
*Update this file after every 2 view/browser/search operations.*
