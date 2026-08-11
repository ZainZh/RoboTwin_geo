# Repository Working Rules

## TAGRT artifact storage (mandatory)

All non-source artifacts produced by TAGRT data collection, preprocessing,
training, evaluation, or rendering **must** live under:

`/shared2/sz/TAGRT-v1`

Use these canonical subdirectories:

- `simulator_data/` for raw RoboTwin demonstrations and replay-admission data;
- `processed_datasets/` for NPZ, HDF5, Zarr, cached tokens, and derived datasets;
- `models/` for NDF, policy, adapter, optimizer, and calibration checkpoints;
- `experiment_outputs/` for logs, metrics, manifests, and full evaluation traces;
- `paper_media/` for rendered success/failure/comparison videos and figures.

Source code, tests, YAML configuration, small JSON summaries, and Markdown
reports stay in the Git workspace.  Model weights, videos, simulator data, and
large generated archives must not be committed to Git or left on the workspace
filesystem.  A workspace path may be a symlink into the canonical data root
when an existing tool requires the old relative path.

New scripts should read `TAGRT_ARTIFACT_ROOT` when provided and otherwise use
`/shared2/sz/TAGRT-v1`.  Temporary local scratch is allowed only when measured
I/O performance requires it; the artifact must be synchronized to the data
disk and the scratch copy removed or replaced by a symlink after the run.

Pre-existing external inputs may be read from their original locations, but
every newly trained replacement and every new experiment artifact follows this
rule.
