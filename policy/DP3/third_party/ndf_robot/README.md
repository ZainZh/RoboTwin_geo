# Vendored NDF runtime

This directory contains only the NDF model architecture required to load the
checkpoints used by the DP3 geometry experiments. It is intentionally smaller
than the full NDF robotics repository.

- Upstream package: `ndf_robot`
- Upstream license: MIT (see `LICENSE.md`)
- Source snapshot: `geometry_awareness_manipulation` commit
  `540534628930fb68dbbf9e5286552f7157dc0c9a`
- Vendored files: `model/layers_equi.py` and
  `model/vnn_occupancy_net_pointnet_dgcnn.py`

The model import in the upstream snapshot used a repository-layout-specific
absolute path. It is changed here to a package-relative import so this runtime
works from a standalone RoboTwin checkout.

The upstream graph helpers also allocated index tensors on a hard-coded CUDA
device. They now allocate on the input tensor's device, preserving GPU behavior
while allowing dependency checks and checkpoint smoke tests on CPU machines.
