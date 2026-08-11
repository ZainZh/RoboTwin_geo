# TAGRT Artifact Storage Policy

Status: mandatory.  The repository-level source of truth is `AGENTS.md`.

Durable experiment artifacts live under `/shared2/sz/TAGRT-v1`:

- `models/`: policy and geometric-estimator checkpoints;
- `simulator_data/`: raw simulator demonstrations and frozen snapshots;
- `processed_datasets/`: Zarr/NPZ/HDF5 policy datasets;
- `experiment_outputs/`: new run metrics, logs, and videos;
- `manifests/`: migration and checksum inventories.

The default root may be overridden only through `TAGRT_ARTIFACT_ROOT`; its
default value is always `/shared2/sz/TAGRT-v1`.

Workspace paths may be absolute symlinks into this root to preserve existing
commands and metadata.  Source code, tests, protocols, and compact JSON/Markdown
results remain in the Git workspace.  New training and collection commands
must write to the shared root unless a measured I/O bottleneck requires a local
scratch copy.  Local scratch is temporary and must be synchronized to the data
disk after a run.

`curate_tagrt_artifacts.py` manages the 2026-08-09 checkpoint migration.  It is
dry-run by default and requires both `--apply` and `--confirm TAGRT-v1` before
it can replace or delete a weight file.  Preserved files are SHA256-verified on
the shared disk before their workspace copies are replaced by symlinks.
