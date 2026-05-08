# PartNext Hammer Eval Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an eval-only RobotWin path that prepares one PartNext hammer as a RobotWin asset and lets `beat_block_hammer` load it through task config without changing the existing task success logic.

**Architecture:** Build a small offline asset-preparation pipeline that parses PartNext `annotation.jsonl`, selects one hammer candidate, computes scale plus handle/head semantic points, and materializes a RobotWin-compatible asset package under `assets/objects/partnext_hammer_eval/`. Then add a narrow task-config override in `envs/beat_block_hammer.py` so eval can swap `020_hammer/base0` for the prepared asset while keeping policy code and success checking unchanged.

**Tech Stack:** Python 3, `unittest`, `numpy`, `trimesh`, `matplotlib` with Agg backend, JSON/YAML, SAPIEN/RobotWin task configs

---

## File Structure

- Create: `script/partnext_hammer_eval_utils.py`
  - Pure utilities for loading annotation rows, selecting one hammer candidate, extracting head/handle face ids, estimating scale, computing local semantic points, rendering a preview, and building `model_data0.json` content.
- Create: `script/test_partnext_hammer_eval_utils.py`
  - Unit tests for annotation parsing, label-priority behavior, scale estimation, preview/package writing, and metadata generation.
- Create: `script/prepare_partnext_hammer_eval_asset.py`
  - CLI entrypoint that uses the utility module to build `assets/objects/partnext_hammer_eval/` and write preview artifacts.
- Modify: `envs/beat_block_hammer.py`
  - Add a pure asset-resolution helper, store `custom_hammer_eval` on the task during setup, and use it inside `load_actors()` and `play_once()` so the task can load either the default hammer or the prepared custom hammer.
- Create: `script/test_beat_block_hammer_eval_override.py`
  - Unit tests for the pure override helper used by `beat_block_hammer`.
- Create: `task_config/demo_clean_3d_partnext_hammer_eval.yml`
  - Eval config that turns on the custom hammer override without altering the eval CLI shape.
- Generated at runtime: `assets/objects/partnext_hammer_eval/visual/base0.glb`, `assets/objects/partnext_hammer_eval/collision/base0.glb`, `assets/objects/partnext_hammer_eval/model_data0.json`, `assets/objects/partnext_hammer_eval/points_info.json`, `assets/objects/partnext_hammer_eval/source_meta.json`, `assets/objects/partnext_hammer_eval/preview/overview.png`
  - Materialized asset package produced by the CLI; do not hand-edit these files after generation.

### Task 1: Build the PartNext Hammer Utility Layer

**Files:**
- Create: `script/partnext_hammer_eval_utils.py`
- Test: `script/test_partnext_hammer_eval_utils.py`

- [ ] **Step 1: Write the failing tests for annotation parsing and scale logic**

```python
import json
import unittest

import numpy as np

from partnext_hammer_eval_utils import (
    collect_region_face_ids,
    estimate_uniform_scale,
    find_annotation_row,
    pick_striking_label,
)


class TestPartNextHammerEvalUtils(unittest.TestCase):
    def test_collect_region_face_ids_merges_nested_handle_labels(self):
        row = {
            "glb_dst": "candidate.glb",
            "hierarchyList": json.dumps([
                {
                    "name": "Hammer",
                    "children": [
                        {
                            "name": "Head",
                            "children": [{"name": "Hammer Head", "maskId": 0}],
                        },
                        {
                            "name": "Handle",
                            "children": [
                                {"name": "Grip", "maskId": 2},
                                {"name": "Shaft", "maskId": 3},
                            ],
                        },
                    ],
                }
            ]),
            "masks": json.dumps({
                "0": {"0": [0, 1]},
                "2": {"0": [20, 21]},
                "3": {"0": [30, 31, 32]},
            }),
        }

        handle_faces = collect_region_face_ids(row, region="handle")
        self.assertEqual(handle_faces, {20, 21, 30, 31, 32})

    def test_pick_striking_label_prefers_hammer_head_over_nail_puller(self):
        label_to_faces = {
            "Nail Puller": {4, 5, 6},
            "Hammer Head": {7, 8, 9},
        }

        self.assertEqual(pick_striking_label(label_to_faces), "Hammer Head")

    def test_estimate_uniform_scale_matches_reference_dominant_extent(self):
        reference_loaded_extents = np.asarray([0.031, 0.178, 0.136], dtype=np.float32)
        candidate_extents = np.asarray([0.140, 0.560, 0.062], dtype=np.float32)

        scale = estimate_uniform_scale(reference_loaded_extents, candidate_extents)
        scaled = candidate_extents * scale

        self.assertAlmostEqual(float(np.max(scaled)), float(np.max(reference_loaded_extents)), places=5)

    def test_find_annotation_row_matches_by_glb_dst(self):
        rows = [
            {"glb_dst": "other.glb", "model_id": "aaa"},
            {"glb_dst": "candidate.glb", "model_id": "bbb"},
        ]

        row = find_annotation_row(rows, glb_name="candidate.glb")
        self.assertEqual(row["model_id"], "bbb")


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the tests to verify they fail before implementation**

Run: `python script/test_partnext_hammer_eval_utils.py`
Expected: FAIL with `ModuleNotFoundError: No module named 'partnext_hammer_eval_utils'`

- [ ] **Step 3: Write the minimal utility implementation**

```python
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np

HEAD_LABELS = ("Hammer Head", "Head", "Nail Puller")
HANDLE_LABELS = ("Handle", "Shaft", "Grip", "Handle End")


def _parse_json_field(value):
    if isinstance(value, str):
        return json.loads(value)
    return value


def load_annotation_rows(annotation_path: Path) -> list[dict]:
    return [json.loads(line) for line in annotation_path.read_text(encoding="utf-8").splitlines() if line.strip()]


def find_annotation_row(rows: list[dict], glb_name: str) -> dict:
    for row in rows:
        if str(row.get("glb_dst", "")) == glb_name:
            return row
    raise KeyError(f"annotation row not found for {glb_name}")


def _flatten_mask_indices(mask_entry) -> set[int]:
    faces: set[int] = set()
    if isinstance(mask_entry, dict):
        for nested in mask_entry.values():
            faces.update(_flatten_mask_indices(nested))
    elif isinstance(mask_entry, list):
        faces.update(int(idx) for idx in mask_entry)
    return faces


def _walk_nodes(nodes: Iterable[dict], parent_names: tuple[str, ...] = ()):
    for node in nodes:
        current_names = parent_names + (str(node.get("name", "")),)
        yield node, current_names
        for child in node.get("children", []) or []:
            yield from _walk_nodes([child], current_names)


def collect_region_face_ids(row: dict, region: str) -> set[int]:
    hierarchy = _parse_json_field(row["hierarchyList"])
    masks = _parse_json_field(row["masks"])
    labels = HANDLE_LABELS if region == "handle" else HEAD_LABELS
    face_ids: set[int] = set()

    for node, names in _walk_nodes(hierarchy):
        if not any(name in labels for name in names):
            continue
        mask_id = node.get("maskId")
        if mask_id is None:
            continue
        face_ids.update(_flatten_mask_indices(masks.get(str(mask_id), {})))
    return face_ids


def pick_striking_label(label_to_faces: dict[str, set[int]]) -> str:
    for label in ("Hammer Head", "Head", "Nail Puller"):
        if label in label_to_faces and label_to_faces[label]:
            return label
    raise ValueError("no striking label candidates available")


def estimate_uniform_scale(reference_loaded_extents: np.ndarray, candidate_extents: np.ndarray) -> float:
    ref = float(np.max(reference_loaded_extents))
    cand = float(np.max(candidate_extents))
    if cand <= 0:
        raise ValueError("candidate extents must be positive")
    scale = ref / cand
    return float(np.clip(scale, 1e-3, 1e3))
```

- [ ] **Step 4: Run the tests to verify the utility layer passes**

Run: `python script/test_partnext_hammer_eval_utils.py`
Expected: `OK`

- [ ] **Step 5: Commit the utility-layer foundation**

```bash
git add script/partnext_hammer_eval_utils.py script/test_partnext_hammer_eval_utils.py
git commit -m "feat: add PartNext hammer eval utility layer"
```

### Task 2: Materialize a RobotWin Asset Package from One PartNext Hammer

**Files:**
- Modify: `script/partnext_hammer_eval_utils.py`
- Create: `script/prepare_partnext_hammer_eval_asset.py`
- Test: `script/test_partnext_hammer_eval_utils.py`

- [ ] **Step 1: Extend the test file with a failing asset-package test**

```python
import tempfile
from pathlib import Path

from partnext_hammer_eval_utils import write_asset_package


class TestPartNextHammerEvalUtils(unittest.TestCase):
    def test_write_asset_package_creates_robotwin_layout(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            source_glb = tmpdir / "candidate.glb"
            source_glb.write_bytes(b"glTF")
            asset_dir = tmpdir / "assets" / "objects" / "partnext_hammer_eval"
            model_data = {
                "scale": [0.1, 0.1, 0.1],
                "contact_points_pose": [np.eye(4).tolist()],
                "functional_matrix": [np.eye(4).tolist()],
                "orientation_point": np.eye(4).tolist(),
                "target_pose": [np.eye(4).tolist()],
                "center": [0.0, 0.0, 0.0],
                "extents": [1.0, 2.0, 3.0],
                "stable": False,
            }

            write_asset_package(
                source_glb=source_glb,
                asset_dir=asset_dir,
                model_data=model_data,
                points_info={"contact": ["handle"], "functional": ["head"]},
                source_meta={"glb_dst": "candidate.glb"},
                preview_png=b"fake-png",
            )

            self.assertTrue((asset_dir / "visual" / "base0.glb").exists())
            self.assertTrue((asset_dir / "collision" / "base0.glb").exists())
            self.assertTrue((asset_dir / "model_data0.json").exists())
            self.assertTrue((asset_dir / "points_info.json").exists())
            self.assertTrue((asset_dir / "source_meta.json").exists())
            self.assertEqual((asset_dir / "preview" / "overview.png").read_bytes(), b"fake-png")
```

- [ ] **Step 2: Run the tests to verify the new package test fails**

Run: `python script/test_partnext_hammer_eval_utils.py`
Expected: FAIL with `ImportError` or `AttributeError` for `write_asset_package`

- [ ] **Step 3: Implement preview generation, package writing, and the preparation CLI**

```python
# script/partnext_hammer_eval_utils.py
from dataclasses import dataclass
from pathlib import Path
import io
import json
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import trimesh


@dataclass
class PreparedHammerAsset:
    source_glb: Path
    model_data: dict
    points_info: dict
    source_meta: dict
    preview_png: bytes


def render_preview_png(mesh: trimesh.Trimesh, contact_point: np.ndarray, functional_point: np.ndarray, handle_axis: np.ndarray) -> bytes:
    sampled = mesh.sample(4000)
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(sampled[:, 0], sampled[:, 1], sampled[:, 2], s=1, c="lightgray", alpha=0.2)
    ax.scatter(*contact_point.tolist(), c="tab:blue", s=80)
    ax.scatter(*functional_point.tolist(), c="tab:red", s=80)
    ax.quiver(contact_point[0], contact_point[1], contact_point[2], handle_axis[0], handle_axis[1], handle_axis[2], length=0.05, color="tab:green")
    ax.set_axis_off()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


def write_asset_package(source_glb: Path, asset_dir: Path, model_data: dict, points_info: dict, source_meta: dict, preview_png: bytes):
    visual_dir = asset_dir / "visual"
    collision_dir = asset_dir / "collision"
    preview_dir = asset_dir / "preview"
    visual_dir.mkdir(parents=True, exist_ok=True)
    collision_dir.mkdir(parents=True, exist_ok=True)
    preview_dir.mkdir(parents=True, exist_ok=True)

    shutil.copy2(source_glb, visual_dir / "base0.glb")
    shutil.copy2(source_glb, collision_dir / "base0.glb")
    (asset_dir / "model_data0.json").write_text(json.dumps(model_data, indent=2), encoding="utf-8")
    (asset_dir / "points_info.json").write_text(json.dumps(points_info, indent=2), encoding="utf-8")
    (asset_dir / "source_meta.json").write_text(json.dumps(source_meta, indent=2), encoding="utf-8")
    (preview_dir / "overview.png").write_bytes(preview_png)


def select_candidate_glb(candidate_glbs: list[Path], reference_loaded_extents: np.ndarray) -> Path:
    best = None
    ref_ratio = np.sort(reference_loaded_extents / np.max(reference_loaded_extents))
    for glb in candidate_glbs:
        mesh = trimesh.load(glb, force='mesh')
        extents = np.asarray(mesh.bounding_box.extents, dtype=np.float32)
        if np.any(extents <= 0):
            continue
        ratio = np.sort(extents / np.max(extents))
        score = float(np.linalg.norm(ratio - ref_ratio))
        item = (score, str(glb))
        if best is None or item < best:
            best = item
    if best is None:
        raise ValueError('no usable hammer candidates found')
    return Path(best[1])


def _region_vertices(mesh: trimesh.Trimesh, face_ids: set[int]) -> np.ndarray:
    if not face_ids:
        raise ValueError('region face ids are empty')
    submesh = mesh.submesh([sorted(face_ids)], append=True, repair=False)
    return np.asarray(submesh.vertices, dtype=np.float32)


def compute_handle_contact_point(mesh: trimesh.Trimesh, row: dict) -> tuple[np.ndarray, np.ndarray]:
    vertices = _region_vertices(mesh, collect_region_face_ids(row, 'handle'))
    center = vertices.mean(axis=0)
    _, _, vh = np.linalg.svd(vertices - center, full_matrices=False)
    axis = vh[0] / np.linalg.norm(vh[0])
    return center.astype(np.float32), axis.astype(np.float32)


def compute_head_functional_point(mesh: trimesh.Trimesh, row: dict, handle_axis: np.ndarray) -> np.ndarray:
    vertices = _region_vertices(mesh, collect_region_face_ids(row, 'head'))
    center = vertices.mean(axis=0)
    lateral = vertices - center
    lateral = lateral - (lateral @ handle_axis)[:, None] * handle_axis[None, :]
    return vertices[np.argmax(np.linalg.norm(lateral, axis=1))].astype(np.float32)


def make_local_pose(point: np.ndarray, x_axis_hint: np.ndarray) -> list[list[float]]:
    z_axis = np.asarray(x_axis_hint, dtype=np.float32)
    z_axis = z_axis / np.linalg.norm(z_axis)
    up = np.asarray([0.0, 0.0, 1.0], dtype=np.float32)
    if abs(float(np.dot(z_axis, up))) > 0.95:
        up = np.asarray([0.0, 1.0, 0.0], dtype=np.float32)
    x_axis = np.cross(up, z_axis)
    x_axis = x_axis / np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)
    pose = np.eye(4, dtype=np.float32)
    pose[:3, 0] = x_axis
    pose[:3, 1] = y_axis
    pose[:3, 2] = z_axis
    pose[:3, 3] = point
    return pose.tolist()


def build_model_data(mesh: trimesh.Trimesh, scale: float, contact_point_local: np.ndarray, functional_point_local: np.ndarray, handle_axis: np.ndarray) -> dict:
    extents = np.asarray(mesh.bounding_box.extents, dtype=np.float32)
    center = np.asarray(mesh.bounding_box.centroid, dtype=np.float32)
    contact_pose = make_local_pose(contact_point_local, handle_axis)
    functional_pose = make_local_pose(functional_point_local, handle_axis)
    return {
        'center': center.tolist(),
        'extents': extents.tolist(),
        'scale': [scale, scale, scale],
        'target_pose': [np.eye(4, dtype=np.float32).tolist()],
        'contact_points_pose': [contact_pose],
        'functional_matrix': [functional_pose],
        'orientation_point': np.eye(4, dtype=np.float32).tolist(),
        'stable': False,
    }


def build_partnext_hammer_asset(partnext_dir: Path, annotation_path: Path) -> PreparedHammerAsset:
    reference_loaded_extents = np.asarray([0.031, 0.178, 0.136], dtype=np.float32)
    candidate_glbs = sorted(partnext_dir.glob('*.glb'))
    rows = load_annotation_rows(annotation_path)
    candidate = select_candidate_glb(candidate_glbs, reference_loaded_extents)
    row = find_annotation_row(rows, candidate.name)
    mesh = trimesh.load(candidate, force='mesh')
    candidate_extents = np.asarray(mesh.bounding_box.extents, dtype=np.float32)
    scale = estimate_uniform_scale(reference_loaded_extents, candidate_extents)
    contact_point_local, handle_axis = compute_handle_contact_point(mesh, row)
    functional_point_local = compute_head_functional_point(mesh, row, handle_axis)
    preview_png = render_preview_png(mesh, contact_point_local, functional_point_local, handle_axis)
    model_data = build_model_data(mesh, scale, contact_point_local, functional_point_local, handle_axis)
    return PreparedHammerAsset(
        source_glb=candidate,
        model_data=model_data,
        points_info={"contact": ["hammer handle grasp point"], "functional": ["hammer striking head point"]},
        source_meta={"glb_dst": candidate.name, "model_id": row.get('model_id')},
        preview_png=preview_png,
    )
```

```python
# script/prepare_partnext_hammer_eval_asset.py
import argparse
from pathlib import Path

from partnext_hammer_eval_utils import build_partnext_hammer_asset, write_asset_package


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--partnext_dir", required=True)
    parser.add_argument("--annotation_path", required=True)
    parser.add_argument("--output_modelname", default="partnext_hammer_eval")
    return parser.parse_args()


def main():
    args = parse_args()
    prepared = build_partnext_hammer_asset(
        partnext_dir=Path(args.partnext_dir),
        annotation_path=Path(args.annotation_path),
    )
    asset_dir = Path("assets") / "objects" / args.output_modelname
    write_asset_package(
        source_glb=prepared.source_glb,
        asset_dir=asset_dir,
        model_data=prepared.model_data,
        points_info=prepared.points_info,
        source_meta=prepared.source_meta,
        preview_png=prepared.preview_png,
    )
    print(f"wrote asset package to {asset_dir}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run unit tests and then run the real asset-preparation smoke command**

Run: `python script/test_partnext_hammer_eval_utils.py`
Expected: `OK`

Run:
```bash
python script/prepare_partnext_hammer_eval_asset.py \
  --partnext_dir /home/zheng/Datasets/PartNext_mesh/Hammer \
  --annotation_path /home/zheng/Datasets/PartNext_mesh/Hammer/annotation.jsonl \
  --output_modelname partnext_hammer_eval
```
Expected:
- prints the selected candidate id and output directory
- creates `assets/objects/partnext_hammer_eval/model_data0.json`
- creates `assets/objects/partnext_hammer_eval/preview/overview.png`

- [ ] **Step 5: Commit the asset-preparation path**

```bash
git add script/partnext_hammer_eval_utils.py script/test_partnext_hammer_eval_utils.py script/prepare_partnext_hammer_eval_asset.py
git commit -m "feat: add PartNext hammer asset preparation CLI"
```

### Task 3: Add the Eval-Time Hammer Override to `beat_block_hammer`

**Files:**
- Modify: `envs/beat_block_hammer.py`
- Create: `script/test_beat_block_hammer_eval_override.py`
- Create: `task_config/demo_clean_3d_partnext_hammer_eval.yml`

- [ ] **Step 1: Write the failing pure override tests**

```python
import unittest

from envs.beat_block_hammer import resolve_hammer_asset_spec


class TestBeatBlockHammerEvalOverride(unittest.TestCase):
    def test_default_asset_spec_when_override_missing(self):
        spec = resolve_hammer_asset_spec(None)
        self.assertEqual(spec["modelname"], "020_hammer")
        self.assertEqual(spec["model_id"], 0)
        self.assertEqual(spec["info_label"], "020_hammer/base0")

    def test_custom_asset_spec_when_override_enabled(self):
        spec = resolve_hammer_asset_spec(
            {
                "enabled": True,
                "modelname": "partnext_hammer_eval",
                "model_id": 0,
            }
        )
        self.assertEqual(spec["modelname"], "partnext_hammer_eval")
        self.assertEqual(spec["model_id"], 0)
        self.assertEqual(spec["info_label"], "partnext_hammer_eval/base0")


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the override tests to verify they fail**

Run: `python script/test_beat_block_hammer_eval_override.py`
Expected: FAIL with `ImportError` or `AttributeError` for `resolve_hammer_asset_spec`

- [ ] **Step 3: Implement the pure override helper, persist the config on the task, and add the task config**

```python
# envs/beat_block_hammer.py

def resolve_hammer_asset_spec(custom_cfg):
    default = {
        "modelname": "020_hammer",
        "model_id": 0,
        "info_label": "020_hammer/base0",
    }
    if not custom_cfg or not custom_cfg.get("enabled", False):
        return default
    modelname = str(custom_cfg.get("modelname", default["modelname"]))
    model_id = int(custom_cfg.get("model_id", default["model_id"]))
    return {
        "modelname": modelname,
        "model_id": model_id,
        "info_label": f"{modelname}/base{model_id}",
    }


class beat_block_hammer(Base_Task):
    def setup_demo(self, **kwags):
        self.custom_hammer_eval = kwags.get("custom_hammer_eval")
        super()._init_task_env_(**kwags)

    def load_actors(self):
        asset_spec = resolve_hammer_asset_spec(self.custom_hammer_eval)
        self.hammer_asset_spec = asset_spec
        self.hammer = create_actor(
            scene=self,
            pose=sapien.Pose([0, -0.06, 0.783], [0, 0, 0.995, 0.105]),
            modelname=asset_spec["modelname"],
            convex=True,
            model_id=asset_spec["model_id"],
        )

    def play_once(self):
        self.info["info"] = {"{A}": self.hammer_asset_spec["info_label"], "{a}": str(arm_tag)}
        return self.info
```

```yaml
# task_config/demo_clean_3d_partnext_objpc_hammer_eval.yml
render_freq: 15
episode_num: 50
use_seed: false
save_freq: 15
embodiment: [aloha-agilex]
language_num: 100
domain_randomization:
  random_background: false
  cluttered_table: false
  clean_background_rate: 1
  random_head_camera_dis: 0
  random_table_height: 0
  random_light: false
  crazy_random_light_rate: 0
camera:
  head_camera_type: D435
  wrist_camera_type: D435
  collect_head_camera: true
  collect_wrist_camera: false
data_type:
  rgb: true
  third_view: true
  depth: false
  pointcloud: true
  observer: false
  endpose: true
  qpos: true
  mesh_segmentation: true
  actor_segmentation: false
pcd_down_sample_num: 1024
pcd_crop: true
save_path: ./data
clear_cache_freq: 5
collect_data: true
eval_video_log: true
custom_hammer_eval:
  enabled: true
  modelname: partnext_hammer_eval
  model_id: 0
```

- [ ] **Step 4: Run the override tests and a config sanity check**

Run: `python script/test_beat_block_hammer_eval_override.py`
Expected: `OK`

Run:
```bash
python - <<'PY'
import yaml
from pathlib import Path
cfg = yaml.safe_load(Path('task_config/demo_clean_3d_partnext_hammer_eval.yml').read_text())
assert cfg['custom_hammer_eval']['enabled'] is True
assert cfg['custom_hammer_eval']['modelname'] == 'partnext_hammer_eval'
print('config ok')
PY
```
Expected: `config ok`

- [ ] **Step 5: Commit the task-side override**

```bash
git add envs/beat_block_hammer.py script/test_beat_block_hammer_eval_override.py task_config/demo_clean_3d_partnext_objpc_hammer_eval.yml
git commit -m "feat: add beat_block_hammer custom eval asset override"
```

### Task 4: Run End-to-End Smoke Validation with the Real Asset

**Files:**
- Use: `assets/objects/partnext_hammer_eval/model_data0.json`
- Use: `assets/objects/partnext_hammer_eval/preview/overview.png`
- Use: `task_config/demo_clean_3d_partnext_hammer_eval.yml`

- [ ] **Step 1: Regenerate the asset package from the real PartNext source**

Run:
```bash
python script/prepare_partnext_hammer_eval_asset.py \
  --partnext_dir /home/zheng/Datasets/PartNext_mesh/Hammer \
  --annotation_path /home/zheng/Datasets/PartNext_mesh/Hammer/annotation.jsonl \
  --output_modelname partnext_hammer_eval
```
Expected:
- `assets/objects/partnext_hammer_eval/visual/base0.glb` exists
- `assets/objects/partnext_hammer_eval/collision/base0.glb` exists
- `assets/objects/partnext_hammer_eval/model_data0.json` exists
- `assets/objects/partnext_hammer_eval/preview/overview.png` exists

- [ ] **Step 2: Inspect the generated metadata and confirm the point semantics**

Run:
```bash
python - <<'PY'
import json
from pathlib import Path
model_data = json.loads(Path('assets/objects/partnext_hammer_eval/model_data0.json').read_text())
print('scale', model_data['scale'])
print('contact_points', len(model_data['contact_points_pose']))
print('functional_points', len(model_data['functional_matrix']))
PY
```
Expected:
- prints one valid scale triplet
- prints `contact_points 1`
- prints `functional_points 1`

Manual check:
- open [overview.png](/home/zheng/github/RoboTwin_geo/assets/objects/partnext_hammer_eval/preview/overview.png)
- confirm the blue handle point lies on the handle region
- confirm the red functional point lies on the striking head side, not the claw side

- [ ] **Step 3: Run an environment-only smoke test that instantiates the task with the custom asset**

Run:
```bash
python - <<'PY'
import os
from pathlib import Path
import yaml

from envs import CONFIGS_PATH
from envs.beat_block_hammer import beat_block_hammer
from script.eval_policy import get_embodiment_config

args = yaml.safe_load(Path('task_config/demo_clean_3d_partnext_hammer_eval.yml').read_text())
with open(os.path.join(CONFIGS_PATH, '_embodiment_config.yml'), 'r', encoding='utf-8') as f:
    embodiment_types = yaml.safe_load(f)
robot_file = embodiment_types[args['embodiment'][0]]['file_path']
args['left_robot_file'] = robot_file
args['right_robot_file'] = robot_file
args['left_embodiment_config'] = get_embodiment_config(robot_file)
args['right_embodiment_config'] = get_embodiment_config(robot_file)
args['dual_arm_embodied'] = True
args['task_name'] = 'beat_block_hammer'
args['task_config'] = 'demo_clean_3d_partnext_hammer_eval'
args['seed'] = 0
args['now_ep_num'] = 0
args['render_freq'] = 0
args['save_data'] = False
args['eval_mode'] = False

env = beat_block_hammer()
env.setup_demo(**args)
print(env.hammer_asset_spec)
env.close_env()
PY
```
Expected:
- prints `{'modelname': 'partnext_hammer_eval', 'model_id': 0, 'info_label': 'partnext_hammer_eval/base0'}`
- does not raise missing-asset or missing-metadata exceptions

- [ ] **Step 4: Run one policy-specific eval command against the new task config**

Run the exact eval wrapper you intend to use for inference. Example for DP3:

```bash
cd policy/DP3
bash eval.sh beat_block_hammer demo_clean_3d_partnext_hammer_eval debug_smoke 0 0
```
Expected:
- the environment runs end-to-end with the custom hammer asset
- success or failure is policy-dependent, but the environment path must stay stable

- [ ] **Step 5: Do not commit generated binary assets until size and provenance are reviewed**

```bash
git status --short
```
Expected:
- source code changes are already committed from Tasks 1-3
- generated `assets/objects/partnext_hammer_eval/*` can be reviewed separately before deciding whether to commit or ignore them
