import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parents[1] / "script"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import partnext_hammer_eval_utils as utils

from partnext_hammer_eval_utils import (
    build_model_data,
    build_partnext_hammer_asset,
    collect_region_face_ids,
    compute_handle_contact_point,
    compute_head_functional_point,
    estimate_uniform_scale,
    find_annotation_row,
    load_annotation_rows,
    pick_striking_label,
    select_candidate_glb,
)


class TestPartNextHammerEvalUtils(unittest.TestCase):
    def test_load_annotation_rows_reads_jsonl(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            annotation_path = Path(tmpdir) / "annotation.jsonl"
            annotation_path.write_text(
                "\n".join(
                    [
                        '{"glb_dst": "one.glb", "model_id": "a1"}',
                        '{"glb_dst": "two.glb", "model_id": "b2"}',
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            rows = load_annotation_rows(annotation_path)

        self.assertEqual([row["glb_dst"] for row in rows], ["one.glb", "two.glb"])
        self.assertEqual(rows[1]["model_id"], "b2")

    def test_collect_region_face_ids_merges_nested_handle_labels(self):
        row = {
            "glb_dst": "candidate.glb",
            "hierarchyList": json.dumps(
                [
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
                ]
            ),
            "masks": json.dumps(
                {
                    "0": {"0": [0, 1]},
                    "2": {"0": [20, 21]},
                    "3": {"0": [30, 31, 32]},
                }
            ),
        }

        handle_faces = collect_region_face_ids(row, region="handle")

        self.assertEqual(handle_faces, {20, 21, 30, 31, 32})

    def test_collect_region_face_specs_preserves_submesh_indices(self):
        row = {
            "glb_dst": "candidate.glb",
            "hierarchyList": json.dumps(
                [
                    {
                        "name": "Hammer",
                        "children": [
                            {
                                "name": "Head",
                                "children": [{"name": "Hammer Head", "maskId": 0}],
                            },
                            {
                                "name": "Handle",
                                "children": [{"name": "Grip", "maskId": 3}],
                            },
                        ],
                    }
                ]
            ),
            "masks": json.dumps(
                {
                    "0": {"0": [5, 6]},
                    "3": {"1": [7, 8], "2": [9]},
                }
            ),
        }

        face_specs = utils.collect_region_face_specs(row, region="handle")

        self.assertEqual(face_specs, {1: {7, 8}, 2: {9}})


    def test_collect_region_face_ids_rejects_invalid_region_before_parsing(self):
        row = {
            "hierarchyList": "not-json",
            "masks": "also-not-json",
        }

        with self.assertRaisesRegex(ValueError, "unsupported region: claw"):
            collect_region_face_ids(row, region="claw")

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

        self.assertAlmostEqual(
            float(np.max(scaled)),
            float(np.max(reference_loaded_extents)),
            places=5,
        )

    def test_find_annotation_row_matches_by_glb_dst(self):
        rows = [
            {"glb_dst": "other.glb", "model_id": "aaa"},
            {"glb_dst": "candidate.glb", "model_id": "bbb"},
        ]

        row = find_annotation_row(rows, glb_name="candidate.glb")

        self.assertEqual(row["model_id"], "bbb")

    def test_find_annotation_row_can_fallback_to_model_id(self):
        rows = [
            {"glb_dst": "other.glb", "model_id": "aaa"},
            {"glb_dst": "missing.glb", "model_id": "partnext_123"},
        ]

        row = find_annotation_row(rows, glb_name="candidate.glb", fallback_model_id="partnext_123")

        self.assertEqual(row["glb_dst"], "missing.glb")

    def test_build_partnext_hammer_asset_honors_requested_glb(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            partnext_dir = root / "partnext"
            partnext_dir.mkdir()
            first_glb = partnext_dir / "first.glb"
            second_glb = partnext_dir / "second.glb"
            utils.trimesh.creation.box(extents=(0.1, 0.4, 0.12)).export(first_glb)
            utils.trimesh.creation.box(extents=(0.2, 0.5, 0.18)).export(second_glb)

            annotation_path = root / "annotation.jsonl"
            rows = [
                {
                    "glb_dst": "first.glb",
                    "model_id": "first",
                    "hierarchyList": json.dumps(
                        [{"name": "Hammer", "children": [{"name": "Head", "children": [{"name": "Hammer Head", "maskId": 0}]}, {"name": "Handle", "children": [{"name": "Grip", "maskId": 3}]}]}]
                    ),
                    "masks": json.dumps({"0": {"0": [0, 1]}, "3": {"0": [2, 3]}}),
                },
                {
                    "glb_dst": "second.glb",
                    "model_id": "second",
                    "hierarchyList": json.dumps(
                        [{"name": "Hammer", "children": [{"name": "Head", "children": [{"name": "Hammer Head", "maskId": 0}]}, {"name": "Handle", "children": [{"name": "Grip", "maskId": 3}]}]}]
                    ),
                    "masks": json.dumps({"0": {"0": [0, 1]}, "3": {"0": [2, 3]}}),
                },
            ]
            annotation_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

            reference_model_data_path = root / "model_data0.json"
            reference_model_data_path.write_text(
                json.dumps(
                    {
                        "extents": [0.1, 0.4, 0.12],
                        "scale": [1.0, 1.0, 1.0],
                        "stable": False,
                    }
                ),
                encoding="utf-8",
            )

            prepared_asset = build_partnext_hammer_asset(
                partnext_dir=partnext_dir,
                annotation_path=annotation_path,
                output_modelname="partnext_hammer_eval",
                reference_model_data_path=reference_model_data_path,
                requested_glb_name="second.glb",
            )

        self.assertEqual(prepared_asset.visual_glb_path, second_glb)
        self.assertEqual(prepared_asset.source_meta["glb_dst"], "second.glb")


    def test_select_candidate_glb_reports_candidate_rejection_details(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            glb_path = root / "candidate.glb"
            mesh = utils.trimesh.creation.box(extents=(0.1, 0.4, 0.12))
            mesh.export(glb_path)
            annotation_rows = [
                {
                    "glb_dst": "candidate.glb",
                    "model_id": "partnext_001",
                    "hierarchyList": json.dumps([{"name": "Hammer Head", "maskId": 0}]),
                    "masks": json.dumps({"0": {"0": [0, 1]}}),
                }
            ]

            with self.assertRaisesRegex(ValueError, "candidate\.glb: missing annotated handle or head faces"):
                select_candidate_glb(
                    partnext_dir=root,
                    annotation_rows=annotation_rows,
                    reference_loaded_extents=np.asarray([0.03, 0.18, 0.14], dtype=np.float32),
                )


    def test_compute_handle_contact_point_aggregates_selected_submesh_centers(self):
        head_mesh = utils.trimesh.creation.box(extents=(0.3, 0.2, 0.3))
        handle_mesh_a = utils.trimesh.creation.box(extents=(0.1, 0.6, 0.1))
        handle_mesh_b = utils.trimesh.creation.box(extents=(0.1, 0.4, 0.1))
        handle_mesh_a.apply_translation([0.0, -0.3, 0.0])
        handle_mesh_b.apply_translation([0.0, -0.25, 0.0])
        geometry_meshes = [head_mesh, handle_mesh_a, handle_mesh_b]
        handle_face_specs = {1: {0, 1}, 2: {0, 1}}

        contact_point, handle_axis = compute_handle_contact_point(geometry_meshes, handle_face_specs)

        expected_centers = np.vstack([
            handle_mesh_a.triangles_center[[0, 1]],
            handle_mesh_b.triangles_center[[0, 1]],
        ])
        self.assertTrue(np.allclose(contact_point, expected_centers.mean(axis=0)))
        self.assertGreater(abs(handle_axis[1]), 0.9)

    def test_compute_head_functional_point_uses_perpendicular_workface_center(self):
        head_mesh = utils.trimesh.creation.box(extents=(0.3, 0.4, 0.3))
        handle_mesh = utils.trimesh.creation.box(extents=(0.1, 0.6, 0.1))
        handle_mesh.apply_translation([0.0, -0.5, 0.0])
        geometry_meshes = [head_mesh, handle_mesh]
        head_face_specs = {0: set(range(len(head_mesh.faces)))}
        handle_axis = np.asarray([0.0, 1.0, 0.0], dtype=np.float64)
        handle_contact_point = np.asarray([0.0, -0.5, 0.0], dtype=np.float64)

        functional_point = compute_head_functional_point(
            geometry_meshes,
            head_face_specs,
            handle_axis,
            handle_contact_point,
        )

        self.assertLess(abs(functional_point[1]), 0.05)
        self.assertGreater(abs(functional_point[2]), 0.12)


    def test_compute_handle_contact_point_uses_handle_region_center(self):
        handle_mesh = utils.trimesh.creation.box(extents=(0.1, 1.0, 0.1))
        head_mesh = utils.trimesh.creation.box(extents=(0.3, 0.2, 0.3))
        head_mesh.apply_translation([0.0, 0.7, 0.0])
        mesh = utils.trimesh.util.concatenate([handle_mesh, head_mesh])
        handle_face_ids = set(range(len(handle_mesh.faces)))

        contact_point, handle_axis = compute_handle_contact_point(mesh, handle_face_ids)

        expected_handle_center = np.asarray(mesh.triangles_center[sorted(handle_face_ids)], dtype=np.float64).mean(axis=0)
        self.assertTrue(np.allclose(contact_point, expected_handle_center))
        self.assertTrue(np.allclose(np.abs(handle_axis), np.asarray([0.0, 1.0, 0.0]), atol=1e-6))

    def test_compute_head_functional_point_prefers_perpendicular_workface_for_merged_mesh(self):
        handle_mesh = utils.trimesh.creation.box(extents=(0.1, 1.0, 0.1))
        head_mesh = utils.trimesh.creation.box(extents=(0.3, 0.2, 0.3))
        head_mesh.apply_translation([0.0, 0.7, 0.0])
        mesh = utils.trimesh.util.concatenate([handle_mesh, head_mesh])
        head_face_ids = set(range(len(handle_mesh.faces), len(handle_mesh.faces) + len(head_mesh.faces)))

        functional_point = compute_head_functional_point(
            mesh,
            head_face_ids,
            np.asarray([0.0, 1.0, 0.0], dtype=np.float64),
            np.asarray([0.0, 0.0, 0.0], dtype=np.float64),
        )

        self.assertLess(abs(functional_point[1] - 0.7), 0.05)
        self.assertGreater(abs(functional_point[2]), 0.12)

    def test_build_model_data_uses_target_point_and_requested_descriptions(self):
        mesh = utils.trimesh.creation.box(extents=(0.1, 1.0, 0.1))
        contact_pose = np.eye(4, dtype=np.float64)
        functional_pose = np.eye(4, dtype=np.float64)
        target_point = np.asarray([0.0, 0.2, 0.0], dtype=np.float64)

        model_data = build_model_data(
            mesh=mesh,
            scale=1.0,
            contact_pose=contact_pose,
            functional_pose=functional_pose,
            target_point=target_point,
            stable=True,
        )

        self.assertTrue(np.allclose(np.asarray(model_data['target_pose'][0], dtype=np.float64)[:3, 3], target_point))
        self.assertEqual(
            model_data['contact_points_discription'][0],
            "Grab the hammer's handle with the head facing outward.",
        )
        self.assertEqual(model_data['target_point_discription'][0], 'The center of the handle part.')
        self.assertEqual(
            model_data['functional_point_discription'][0],
            'Point 0: The head of the hammer is facing outward.',
        )


    def test_write_asset_package_writes_package_files_and_preview_ply(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            visual_src = root / "candidate_visual.glb"
            collision_src = root / "candidate_collision.glb"
            visual_src.write_bytes(b"visual-glb")
            collision_src.write_bytes(b"collision-glb")

            prepared_asset = utils.PreparedHammerAsset(
                modelname="partnext_hammer_eval",
                visual_glb_path=visual_src,
                collision_glb_path=collision_src,
                model_data={
                    "scale": [0.5, 0.5, 0.5],
                    "center": [0.0, 0.0, 0.0],
                    "extents": [0.1, 0.5, 0.2],
                    "target_pose": [],
                    "contact_points_pose": [np.eye(4).tolist()],
                    "functional_matrix": [np.eye(4).tolist()],
                    "orientation_point": [np.eye(4).tolist()],
                    "stable": True,
                },
                points_info={
                    "contact_points": [{"id": 0, "description": "hammer handle grasp point"}],
                    "functional_points": [{"id": 0, "description": "hammer striking head point"}],
                },
                source_meta={
                    "model_id": "partnext_001",
                    "glb_dst": "candidate.glb",
                    "matched_labels": {
                        "handle": ["Handle"],
                        "head": ["Hammer Head"],
                    },
                },
            )
            preview_ply = b"ply\nformat ascii 1.0\nend_header\n"
            output_root = root / "assets" / "objects"

            asset_dir = utils.write_asset_package(
                output_root=output_root,
                prepared_asset=prepared_asset,
                preview_ply=preview_ply,
            )

            self.assertEqual(asset_dir, output_root / "partnext_hammer_eval")
            self.assertEqual((asset_dir / "visual" / "base0.glb").read_bytes(), b"visual-glb")
            self.assertEqual((asset_dir / "collision" / "base0.glb").read_bytes(), b"collision-glb")
            self.assertEqual(
                (asset_dir / "preview" / "overview.ply").read_bytes(),
                preview_ply,
            )

            model_data = json.loads((asset_dir / "model_data0.json").read_text(encoding="utf-8"))
            self.assertEqual(model_data["scale"], [0.5, 0.5, 0.5])
            self.assertEqual(len(model_data["contact_points_pose"]), 1)
            self.assertEqual(len(model_data["orientation_point"]), 1)

            points_info = json.loads((asset_dir / "points_info.json").read_text(encoding="utf-8"))
            self.assertEqual(points_info["contact_points"][0]["description"], "hammer handle grasp point")

            source_meta = json.loads((asset_dir / "source_meta.json").read_text(encoding="utf-8"))
            self.assertEqual(source_meta["model_id"], "partnext_001")


if __name__ == "__main__":
    unittest.main()
