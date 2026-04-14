# Findings & Decisions

## Requirements
- User explicitly invoked `planning-with-files`.
- Initialize persistent planning files in the project root for this repository.
- Keep future discovery, especially external or untrusted content, in this file rather than `task_plan.md`.
- Current repository goal: evaluate whether the 3D representation method improves VLA task success rate, with edits mainly under `policy/DP3`.
- Current near-term task: check whether `policy/DP3/train_objpc_sam3.sh` can work and estimate how much time is needed to fuse object PCDs for one frame using the segment-and-project path.
- Longer-term direction: improve NDF and semantic training separately, then integrate them together and compare against baseline.
- New task on 2026-04-10: replace SAM-based segmentation with simulator-provided segmentation to build fused object point clouds for DP3 VLA training and evaluation, and add matching train/eval bash scripts.
- New task on 2026-04-10 (later): diagnose why `train_ndf_pointwise` is not improving over `train_objpc`, while `train_semantic_pointwise` improves substantially.

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
- The simulator already exposes two segmentation sources through `envs/_base_task.py`: `mesh_segmentation` and `actor_segmentation`.
- The simulator also already exposes direct per-placeholder oracle object point clouds through `observation["object_pointcloud"]` when `data_type.object_pointcloud=true`.
- Existing non-SAM training and eval entrypoints already exist:
- `policy/DP3/train_objpc.sh`
- `policy/DP3/eval_objpc.sh`
- Existing online eval logic in `policy/DP3/deploy_policy.py` already prefers `observation["object_pointcloud"]` when `use_object_pointcloud` is enabled and `use_sam3_objpc` is false.
- Existing offline fallback logic in `policy/DP3/scripts/object_pointcloud_utils.py` can derive object point clouds from simulator segmentation, but it currently only loads `head_camera` segmentation from the dataset and does not implement the newer SAM-style multi-camera fused path.
- `envs/camera/camera.py` returns segmentation as a colorized image derived from the simulator's `Segmentation` buffer:
- channel 0 for mesh-level labels
- channel 1 for actor-level labels
- `mesh_segmentation` and `actor_segmentation` are not equivalent in this repo:
- `mesh_segmentation` uses `Segmentation[..., 0]` and corresponds to visual-shape / mesh-level labels.
- `actor_segmentation` uses `Segmentation[..., 1]` and corresponds to actor/entity-level labels, which is the closer match to per-object instance masks.
- Current simulator segmentation export colorizes the label map for storage/visualization instead of preserving raw integer ids; for a robust placeholder-specific projection path, using raw actor ids internally would be preferable to reusing the colorized output as the canonical representation.
- The oracle `object_pointcloud` path filters camera point clouds directly by actor ids using `Segmentation[..., 1]`, so it is naturally aligned with `actor_segmentation`, not `mesh_segmentation`.
- Collected `scene_info.json` files for `*_object_pc` datasets already contain placeholder-to-actor-id mappings under `episode_k.object_pointcloud.targets.{placeholder}.actor_ids`; this is sufficient to drive offline actor-segmentation projection without using stored oracle point clouds.
- There is currently no local dataset under `data/` that already contains saved `actor_segmentation` or `mesh_segmentation` frames, so end-to-end validation of the new pipeline will require recollecting a small dataset or enabling segmentation in an eval task config.
- Implemented a new `objpc_actorseg` pipeline:
- offline utility: `policy/DP3/scripts/actorseg_pointcloud_utils.py`
- offline preprocessing: `policy/DP3/scripts/process_data_objpc_actorseg.py`
- shell entrypoints: `policy/DP3/process_data_objpc_actorseg.sh`, `policy/DP3/train_objpc_actorseg.sh`, `policy/DP3/eval_objpc_actorseg.sh`
- Hydra configs: `robot_dp3_objpc_actorseg.yaml`, `demo_task_objpc_actorseg.yaml`
- collection/eval task config templates: `task_config/demo_clean_3d_actorseg.yml`, `task_config/demo_randomized_3d_actorseg.yml`
- Online eval support for the new pipeline is now wired into `policy/DP3/deploy_policy.py` via a dedicated `use_actorseg_objpc` branch.
- `envs/_base_task.py` now preserves object-target metadata when `actor_segmentation` or `mesh_segmentation` is enabled, even if `data_type.object_pointcloud` is false; this is intended so actor-segmentation datasets can still save placeholder->actor_ids in `scene_info.json` without storing per-frame oracle object point clouds.
- The new offline actor-segmentation preprocessing path is fail-fast: if the requested cameras do not contain saved `actor_segmentation`, it raises an explicit error instead of silently generating zero point clouds.
- Negative verification against the existing `demo_clean_3d_object_pc` dataset confirmed this guard works as intended: preprocessing aborts with `missing_actor_segmentation` for `head_camera` and `front_camera`.
- The new actor-segmentation extraction path uses the stored colorized segmentation images and placeholder actor ids to select projected scene points. This mirrors the SAM segment-project-fuse structure, but depends on the current color-palette encoding of simulator labels.
- For the new requested work, the likely cleanest replacement for SAM is not direct `object_pointcloud`, but a new fused multi-camera simulator-segmentation projection path that mirrors the structure of the SAM3 pipeline while removing the learned 2D segmenter.
- `train_objpc.sh`, `train_ndf_pointwise.sh`, and `train_semantic_pointwise.sh` are not a controlled apples-to-apples comparison:
- `train_semantic_pointwise.sh` changes the training recipe by default (`training.use_ema=false`, `dataloader.batch_size=110`, `val_dataloader.batch_size=110`), while `objpc` and `ndf_pointwise` keep the base DP3 defaults (`use_ema=True`, `batch_size=256`).
- The saved Hydra overrides confirm the existing local runs used those differing defaults:
- `objpc`: only dataset path / setting overrides
- `ndf_pointwise`: only dataset path / extra obs key overrides
- `semantic_pointwise`: dataset path / extra obs key overrides plus `training.use_ema=false` and `batch_size=110`
- For the local `hanging_mug` experiments currently on disk, both `ndf_pointwise` and `semantic_pointwise` only augment placeholder `{A}` (the mug), not `{B}` (the rack). `feature_placeholders` in both metadata files are `['{A}']`.
- In `deploy_policy.py`, pointwise NDF / semantic runs do not keep the raw `{A}` object cloud inside the primary `point_cloud`. Instead:
- primary `point_cloud` becomes only the context placeholders (here `{B}` rack)
- the mug is provided separately via `ndf_point_cloud_A` or `semantic_point_cloud_A`
- Therefore this is not “baseline objpc plus extra feature”; it is a different observation factorization.
- `DP3Encoder` creates one PointNet branch per point-cloud observation key. So baseline `objpc` has one point-cloud branch, while `ndf_pointwise` / `semantic_pointwise` have two point-cloud branches (`point_cloud` + feature cloud), meaning the encoder architecture also changes.
- The pointwise feature branches are both consumed by `PointNetEncoderXYZRGB`, which is a generic per-point MLP + max-pool encoder. There is no NDF-specific alignment or correspondence module on top of the NDF field inside DP3.
- The saved NDF zarr and semantic zarr both contain 50 episodes (`meta/episode_ends` length 50) and the same step counts as the baseline objpc zarr, so the current comparison is using aligned DP3 training data volumes.
- The saved metadata for both pointwise variants shows the training set covered mug assets:
- `039_mug/base0`
- `039_mug/base1`
- `039_mug/base2`
- `039_mug/base3`
- `039_mug/base4`
- `039_mug/base6`
- `039_mug/base7`
- `039_mug/base8`
- `039_mug/base9`
- This means the training artifacts already span 9 mug instances from the same asset family, not a single mug instance.
- `envs/hanging_mug.py` samples `self.mug_id = np.random.choice([i for i in range(10)])`, so eval also draws from the same 10-instance `039_mug` family. This is not a strong novel-object benchmark; at most one mug instance (`base5`) is absent from the saved training metadata.
- `policy/DP3/deploy_policy.yml` sets `instruction_type: unseen`, and `script/eval_policy.py` writes this value into `_result.txt`. In this repo, “unseen” refers to unseen language instructions, not unseen object instances.
- The saved semantic eval result file reports `Instruction Type: unseen` and success `0.36`; this should not be interpreted as a novel-object result.
- `train_ndf_pointwise.sh` contains a bug in the preprocessing call: it passes `"${ndf_dgcnn_/placeholders}"` instead of `"${ndf_dgcnn_placeholders}"`. This prevents the user from correctly forwarding `ndf_dgcnn_placeholders` through the train script.
- That forwarding bug does not affect the user's current pointnet-backed NDF setup when `ndf_dgcnn_placeholders` is intentionally empty, but it would break any future DGCNN-backed NDF run.
- The currently used NDF checkpoint in `policy/DP3/Command.md` is `/home/zheng/train_results/Mug/checkpoints/model_current.pth`, not `model_best.pth`.
- Checkpoint-architecture compatibility check:
- `model_current.pth` cleanly matches `dgcnn=False` and mismatches under `dgcnn=True`
- `model_best.pth` cleanly matches `dgcnn=True` and mismatches under `dgcnn=False`
- This means the local NDF experiment is currently using a pointcloud-backbone “current” checkpoint rather than the DGCNN-backed “best” checkpoint.
- Because of the `train_ndf_pointwise.sh` forwarding bug, switching to the DGCNN-backed best checkpoint through the train script is currently blocked.
- The saved pointwise feature statistics suggest semantic features are much more temporally stable than NDF features in the current pipeline:
- over the first 200 frames, NDF frame-mean feature std is about `0.0108`
- over the first 200 frames, semantic frame-mean feature std is about `0.0012`
- flattened adjacent-frame cosine similarity is about `0.764` for NDF vs `0.999` for semantic
- These are rough probes, not full invariance metrics, but they are consistent with semantic features behaving like a much more stable object/part code and NDF behaving like a more frame-sensitive local geometry field.
- Both NDF and semantic pointwise features are unit-norm on disk (mean feature norm about `1.0`), so the underperformance signal is not explained by a simple feature-scale explosion.
- The current raw dataset folder under `data/hanging_mug/demo_clean_3d_object_pc` now contains only `episode0.hdf5` and `scene_info.json` with only `episode_0`, while the saved DP3 zarrs clearly contain 50 episodes. Therefore the current raw demo directory is no longer the exact source snapshot used to build the existing 50-episode artifacts.
- The requested fairness cleanup has now been applied:
- `train_ndf_pointwise.sh` correctly forwards `ndf_dgcnn_placeholders`
- `train_semantic_pointwise.sh` now defaults back to `batch_size=256`, `val_batch_size=batch_size`, and `use_ema=true`
- `use_ema` in this codebase means training and saving an exponential moving average copy of the policy weights. In practice it usually stabilizes eval and often gives a modest improvement, but the impact is task-dependent rather than guaranteed to be large.
- New task on 2026-04-13: explain how `train_ndf_pointwise_hybrid.sh` combines baseline object-PCD observations with NDF pointwise features, then build the analogous `semantic_pointwise_hybrid` path.
- New task on 2026-04-14: add actorseg-based hybrid training/eval for both NDF and semantic, mirroring the successful `objpc + feature-branch` hybrid structure while using mask-project-fuse object point clouds like `train_objpc_actorseg`.
- User-provided experiment evidence: the existing global-NDF path (`train_ndf.sh` / `robot_dp3_ndf`) was already tested using `policy/DP3/checkpoints/hanging_mug-demo_clean_3d_object_pc-objpc-ndf-50_0`, and it caused a severe behavioral regression: the robot often failed to approach the mug, consistent with losing or underusing object location information.
- This is consistent with the current global-NDF integration design:
- `compute_ndf_feature(...)` explicitly normalizes the object point cloud by centering and scaling it before extracting the descriptor, making the NDF feature pose-invariant by construction, see `normalize_object_point_cloud(...)` in [ndf_feature_utils.py](/home/zheng/github/RoboTwin_geo/policy/DP3/scripts/ndf_feature_utils.py#L431).
- `process_data_ndf.py` stores that descriptor as a low-dimensional extra observation `ndf_feat_A`, not as a localized point set, while the scene point cloud remains a merged `{A}+{B}` cloud, see [process_data_ndf.py](/home/zheng/github/RoboTwin_geo/policy/DP3/scripts/process_data_ndf.py#L153).
- Therefore the policy receives an object-identity / geometry descriptor with no explicit correspondence back to the mug points inside the merged scene cloud. This makes “global NDF causes localization regression” a credible integration failure mode rather than evidence that NDF is fundamentally useless.
- Current best diagnosis for the NDF underperformance is:
- Most likely: the current NDF integration path is suboptimal for DP3.
- Evidence:
- NDF pointwise observations are fed as a separate high-dimensional point-cloud branch into the generic `PointNetEncoderXYZRGB` path, not a correspondence-aware or geometry-aware downstream head.
- The primary `point_cloud` no longer contains the `{A}` object when NDF pointwise is enabled; `{A}` is isolated into its own branch, so the policy sees a different factorization than baseline `objpc`.
- NDF pointwise features are materially less temporally stable than semantic pointwise features in the current saved dataset probes.
- Secondary likely cause: the current benchmark underexposes the kind of generalization where NDF would be expected to help most.
- Evidence:
- current eval is “unseen instruction”, not unseen object
- train/eval both operate on the same `039_mug` family with broad overlap of mug ids
- Less likely but still plausible: the current pointnet-backed NDF checkpoint is not the strongest available NDF model for this use case.
- Evidence:
- command examples use `model_current.pth`
- `model_best.pth` corresponds to a different backbone family (`dgcnn=True`)
- There is not enough evidence to conclude “NDF cannot help this task.” The current evidence only supports “this particular NDF integration + benchmark setup is not showing an advantage.”
- The approved mitigation experiment has now been implemented as a separate `ndf_pointwise_hybrid` path rather than changing the existing `ndf_pointwise` behavior in place.
- `policy/DP3/scripts/pointwise_context_utils.py` now owns the shared logic for deciding whether feature placeholders stay in the main merged `point_cloud`.
- `policy/DP3/deploy_policy.py` now treats config names containing `ndf_pointwise_hybrid` as a distinct runtime mode:
- `ndf_point_cloud_A/B` generation is unchanged
- the main `point_cloud` keeps all placeholders instead of dropping feature placeholders
- `policy/DP3/scripts/process_data_ndf_pointwise.py` now supports `--keep_feature_placeholders_in_context`, and the new wrapper `policy/DP3/scripts/process_data_ndf_pointwise_hybrid.py` uses that flag together with the `-objpc-ndf-pointwise-hybrid` dataset suffix.
- New hybrid entrypoints now exist:
- `policy/DP3/process_data_ndf_pointwise_hybrid.sh`
- `policy/DP3/train_ndf_pointwise_hybrid.sh`
- `policy/DP3/eval_ndf_pointwise_hybrid.sh`
- `policy/DP3/3D-Diffusion-Policy/diffusion_policy_3d/config/robot_dp3_ndf_pointwise_hybrid.yaml`
- `policy/DP3/3D-Diffusion-Policy/diffusion_policy_3d/config/task/demo_task_ndf_pointwise_hybrid.yaml`
- The focused regression test `policy/DP3/scripts/test_ndf_pointwise_hybrid.py` now proves:
- old `ndf_pointwise` excludes `{A}` from the main `point_cloud`
- new `ndf_pointwise_hybrid` keeps `{A}` and `{B}` in the main `point_cloud`
- both still emit `ndf_point_cloud_A`
- Fresh verification evidence for the hybrid work:
- `python policy/DP3/scripts/test_ndf_pointwise_hybrid.py` passed with `2` tests
- `python -m py_compile policy/DP3/deploy_policy.py policy/DP3/scripts/process_data_ndf_pointwise.py policy/DP3/scripts/process_data_ndf_pointwise_hybrid.py policy/DP3/scripts/pointwise_context_utils.py policy/DP3/scripts/test_ndf_pointwise_hybrid.py` passed
- `bash -n policy/DP3/process_data_ndf_pointwise_hybrid.sh` passed
- `bash -n policy/DP3/train_ndf_pointwise_hybrid.sh` passed
- `bash -n policy/DP3/eval_ndf_pointwise_hybrid.sh` passed
- `train_ndf_pointwise_hybrid.sh` keeps the baseline merged raw `point_cloud` because its preprocess wrapper adds `--keep_feature_placeholders_in_context`, while still injecting `ndf_point_cloud_A/B` through `+task.dataset.extra_obs_keys=[...]` and dynamic shape overrides. This is the exact pattern to mirror for a semantic hybrid.
- The current semantic pointwise path still removes feature placeholders from the main `point_cloud`:
  - offline preprocessing in `process_data_semantic_pointwise.py` sets `context_placeholders = placeholders - feature_placeholders`
  - runtime `deploy_policy.py` only keeps feature placeholders in the main `point_cloud` when `use_ndf_pointwise_hybrid=True`
  - there is no `use_semantic_pointwise_hybrid` branch yet
- The minimum semantic-hybrid delta is therefore:
  - allow semantic preprocessing to accept `--keep_feature_placeholders_in_context`
  - add a thin wrapper `process_data_semantic_pointwise_hybrid.py`
  - add matching `train/eval/process` shell entrypoints and Hydra config/task-config names
  - extend runtime `deploy_policy.py` with a `use_semantic_pointwise_hybrid` mode that reuses the same context-building helper as NDF hybrid
- The requested `semantic_pointwise_hybrid` path has now been implemented with the same observation structure as the NDF hybrid:
  - main `point_cloud` keeps the baseline merged raw ObjPC
  - `semantic_point_cloud_A/B` remain extra point-cloud branches
  - low-dim `agent_pos` is unchanged
- Important encoder clarification:
  - DP3 does not concatenate raw point cloud, semantic pointwise cloud, and agent state at the point level
  - instead, [DP3Encoder](/home/zheng/github/RoboTwin_geo/policy/DP3/3D-Diffusion-Policy/diffusion_policy_3d/model/vision/pointnet_extractor.py#L194) encodes each point-cloud key with its own PointNet extractor, encodes low-dim state with a separate MLP, then concatenates the branch-level features
  - therefore `semantic_pointwise_hybrid` means `f_raw_objpc || f_semantic_pointwise || f_agent_pos`, not `[xyz|rgb|semantic|state]` inside one shared PointNet
- New semantic hybrid entrypoints now exist:
  - `policy/DP3/scripts/process_data_semantic_pointwise_hybrid.py`
  - `policy/DP3/process_data_semantic_pointwise_hybrid.sh`
  - `policy/DP3/train_semantic_pointwise_hybrid.sh`
  - `policy/DP3/eval_semantic_pointwise_hybrid.sh`
  - `policy/DP3/3D-Diffusion-Policy/diffusion_policy_3d/config/robot_dp3_semantic_pointwise_hybrid.yaml`
  - `policy/DP3/3D-Diffusion-Policy/diffusion_policy_3d/config/task/demo_task_semantic_pointwise_hybrid.yaml`
- Fresh verification evidence for the semantic hybrid work:
  - `python policy/DP3/scripts/test_semantic_pointwise_hybrid.py` passed with `3` tests
  - `python policy/DP3/scripts/test_ndf_pointwise_hybrid.py` still passed with `3` tests
  - `python -m py_compile policy/DP3/deploy_policy.py policy/DP3/scripts/process_data_semantic_pointwise.py policy/DP3/scripts/process_data_semantic_pointwise_hybrid.py policy/DP3/scripts/test_semantic_pointwise_hybrid.py` passed
  - `bash -n policy/DP3/process_data_semantic_pointwise_hybrid.sh` passed
  - `bash -n policy/DP3/train_semantic_pointwise_hybrid.sh` passed
  - `bash -n policy/DP3/eval_semantic_pointwise_hybrid.sh` passed
- New debugging result on 2026-04-14 for `eval_ndf_pointwise_hybrid.sh`:
  - The checkpoint mismatch was not caused by a changed model definition in training.
  - The user invocation omitted the explicit empty `ndf_dgcnn_placeholders` argument:
    - intended order: `... ndf_device ndf_dgcnn_placeholders object_placeholders checkpoint_num`
    - actual call placed `"{A},{B}"` into `ndf_dgcnn_placeholders` and `3000` into `object_placeholders`
  - Evidence:
    - eval log showed `[DP3Encoder] point cloud keys: ['point_cloud']`, which means `ndf_point_cloud_A` was never injected into the eval model shape meta
    - the checkpoint state dict *does* contain `obs_encoder.extractors.ndf_point_cloud_A.*`, which proves the training run that produced it had the NDF hybrid branch enabled
  - Therefore the checkpoint is structurally consistent with hybrid training; the eval-time CLI parsing was wrong.
  - Fix implemented: `policy/DP3/ndf_pointwise_arg_utils.sh` now normalizes both the explicit and legacy call forms, and the following scripts now auto-correct the omitted-empty-argument form:
    - `policy/DP3/eval_ndf_pointwise_hybrid.sh`
    - `policy/DP3/train_ndf_pointwise_hybrid.sh`
    - `policy/DP3/eval_ndf_pointwise.sh`
    - `policy/DP3/train_ndf_pointwise.sh`
- Current actorseg+hybrid design constraint:
  - training must use actorseg-projected fused object clouds as the main `point_cloud`
  - hybrid branches must still add `ndf_point_cloud_A/B` or `semantic_point_cloud_A/B` on top of that raw actorseg point cloud
  - eval must perform the same online actorseg extraction once per placeholder and reuse that same per-placeholder cloud for both the main merged `point_cloud` and the feature branches
- The cleanest implementation path is to add separate pipelines rather than mutating existing working ones:
  - `objpc_actorseg_ndf_pointwise_hybrid`
  - `objpc_actorseg_semantic_pointwise_hybrid`
- Actorseg hybrid runtime support is now implemented in `deploy_policy.py` by reusing the per-placeholder actorseg-extracted point clouds for both:
  - the merged raw `point_cloud`
  - the `ndf_point_cloud_A/B` or `semantic_point_cloud_A/B` feature branches
  This keeps online eval semantics aligned with offline preprocessing.
- New actorseg hybrid preprocessing entrypoints now exist:
  - `policy/DP3/scripts/process_data_ndf_pointwise_actorseg_hybrid.py`
  - `policy/DP3/scripts/process_data_semantic_pointwise_actorseg_hybrid.py`
  - `policy/DP3/process_data_ndf_pointwise_actorseg_hybrid.sh`
  - `policy/DP3/process_data_semantic_pointwise_actorseg_hybrid.sh`
- New actorseg hybrid train/eval entrypoints now exist:
  - `policy/DP3/train_ndf_pointwise_actorseg_hybrid.sh`
  - `policy/DP3/train_semantic_pointwise_actorseg_hybrid.sh`
  - `policy/DP3/eval_ndf_pointwise_actorseg_hybrid.sh`
  - `policy/DP3/eval_semantic_pointwise_actorseg_hybrid.sh`
- New Hydra configs now exist:
  - `policy/DP3/3D-Diffusion-Policy/diffusion_policy_3d/config/robot_dp3_objpc_actorseg_ndf_pointwise_hybrid.yaml`
  - `policy/DP3/3D-Diffusion-Policy/diffusion_policy_3d/config/robot_dp3_objpc_actorseg_semantic_pointwise_hybrid.yaml`
  - `policy/DP3/3D-Diffusion-Policy/diffusion_policy_3d/config/task/demo_task_objpc_actorseg_ndf_pointwise_hybrid.yaml`
  - `policy/DP3/3D-Diffusion-Policy/diffusion_policy_3d/config/task/demo_task_objpc_actorseg_semantic_pointwise_hybrid.yaml`
- Targeted regression coverage now exists in `policy/DP3/scripts/test_actorseg_pointwise_hybrid.py`, proving:
  - actorseg+NDF hybrid keeps merged raw actorseg `point_cloud` and emits `ndf_point_cloud_A`
  - actorseg+semantic hybrid keeps merged raw actorseg `point_cloud` and emits `semantic_point_cloud_A`
- Fresh verification evidence for the actorseg hybrid work:
  - `python policy/DP3/scripts/test_actorseg_pointwise_hybrid.py` passed with `2` tests
  - `python policy/DP3/scripts/test_ndf_pointwise_hybrid.py` still passed with `3` tests
  - `python policy/DP3/scripts/test_semantic_pointwise_hybrid.py` still passed with `3` tests
  - `bash policy/DP3/scripts/test_ndf_pointwise_arg_utils.sh` passed
  - `python -m py_compile policy/DP3/deploy_policy.py policy/DP3/scripts/process_data_ndf_pointwise_actorseg_hybrid.py policy/DP3/scripts/process_data_semantic_pointwise_actorseg_hybrid.py policy/DP3/scripts/test_actorseg_pointwise_hybrid.py` passed
  - `bash -n` passed for:
    - `policy/DP3/process_data_ndf_pointwise_actorseg_hybrid.sh`
    - `policy/DP3/process_data_semantic_pointwise_actorseg_hybrid.sh`
    - `policy/DP3/train_ndf_pointwise_actorseg_hybrid.sh`
    - `policy/DP3/train_semantic_pointwise_actorseg_hybrid.sh`
    - `policy/DP3/eval_ndf_pointwise_actorseg_hybrid.sh`
    - `policy/DP3/eval_semantic_pointwise_actorseg_hybrid.sh`
- Follow-up bug found during first real user run: `process_data_ndf_pointwise_hybrid.py` originally forwarded `--output_suffix` as two argv items, and the value `-objpc-ndf-pointwise-hybrid` was parsed as a new option because it starts with `-`.
- Root-cause fix: `policy/DP3/scripts/process_data_ndf_pointwise_hybrid.py` now builds argv through `build_hybrid_argv(...)` and passes `--output_suffix=-objpc-ndf-pointwise-hybrid` as an attached option.
- The regression test `policy/DP3/scripts/test_ndf_pointwise_hybrid.py` now includes a dedicated wrapper-argv check for this case and passes with `3` tests.
- Additional NDF-loading finding: the warning
  `unexpected_keys=['decoder.vector_basis.map_to_feat.weight', 'decoder.fc_vec_alpha.weight', 'decoder.fc_vec_alpha.bias']`
  is generally benign for the current DP3 pointwise NDF path when `missing_keys=[]`.
- Reason:
- `policy/DP3/scripts/ndf_feature_utils.py` builds `VNNOccNet(..., return_features=True, return_vector_features=False)` and loads checkpoints with `strict=False`.
- The reported unexpected keys belong to the optional decoder vector-feature head, which is only instantiated when `return_vector_features=True`, see `vnn_occupancy_net_pointnet_dgcnn.py`.
- Current DP3 feature extraction only calls `model.forward_latent(z, pts)` / `model.forward_latent(z, query)` and uses the regular `features` output, not `vector_features`.
- Therefore this exact warning means “the checkpoint contains extra decoder weights that the current loader ignores,” not “the main encoder/decoder weights are missing.”
- Risk rule:
- acceptable: `missing_keys=[]` and only optional vector-feature decoder keys are unexpected
- suspicious: any `missing_keys`, or unexpected keys under encoder / core decoder blocks / backbone family mismatch

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
| Treat the new request as “replace the 2D segmenter, keep the segment-project-fuse idea” unless the user explicitly asks to switch to oracle `object_pointcloud` | The user explicitly asked to use simulator segmentation instead of SAM, not to bypass segmentation entirely |
| Treat the NDF question as a diagnosis problem, not a one-step benchmark-design explanation | The current code and saved runs already expose multiple confounders beyond “NDF only helps on novel objects” |
| Add `ndf_pointwise_hybrid` as a new named path instead of redefining `ndf_pointwise` | This preserves the meaning of existing checkpoints while isolating the “keep raw `{A}` in the main cloud” hypothesis |

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
