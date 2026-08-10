from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np
import zarr

from audit_robotwin_dinov2_inputs import audit_robotwin_dinov2_inputs


class AuditRobotwinDinov2InputsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.dataset_dir = Path(self.temporary.name) / "dataset"
        self.data_dir = self.dataset_dir / "data"
        self.data_dir.mkdir(parents=True)
        self.lengths = [3, 4]
        for episode, length in enumerate(self.lengths):
            self._write_episode(episode, length, include_depth=True)
        self.field_zarr = Path(self.temporary.name) / "field.zarr"
        self._write_field_zarr([length - 1 for length in self.lengths])

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _write_episode(self, episode: int, length: int, *, include_depth: bool) -> None:
        with h5py.File(self.data_dir / f"episode{episode}.hdf5", "w") as root:
            joint = root.create_group("joint_action")
            joint.create_dataset("vector", data=np.zeros((length, 14), dtype=np.float32))
            observation = root.create_group("observation")
            for camera_name in ("head_camera", "front_camera"):
                camera = observation.create_group(camera_name)
                camera.create_dataset(
                    "rgb", data=np.zeros((length, 4, 5, 3), dtype=np.uint8)
                )
                if include_depth:
                    camera.create_dataset(
                        "depth", data=np.ones((length, 4, 5), dtype=np.float32)
                    )
                camera.create_dataset(
                    "intrinsic_cv",
                    data=np.repeat(np.eye(3, dtype=np.float32)[None], length, axis=0),
                )
                camera.create_dataset(
                    "extrinsic_cv",
                    data=np.repeat(np.eye(4, dtype=np.float32)[None, :3], length, axis=0),
                )
                camera.create_dataset(
                    "cam2world_gl",
                    data=np.repeat(np.eye(4, dtype=np.float32)[None], length, axis=0),
                )

    def _write_field_zarr(self, policy_lengths: list[int]) -> None:
        root = zarr.open(str(self.field_zarr), mode="w")
        data = root.create_group("data")
        meta = root.create_group("meta")
        episode_ends = np.cumsum(policy_lengths, dtype=np.int64)
        total = int(episode_ends[-1])
        meta.array("episode_ends", episode_ends)
        data.array("action", np.zeros((total, 14), dtype=np.float32))
        data.array("state", np.zeros((total, 14), dtype=np.float32))
        data.array("point_cloud", np.zeros((total, 8, 6), dtype=np.float32))
        data.array(
            "semantic_point_cloud_A",
            np.zeros((total, 4, 7), dtype=np.float32),
        )

    def test_complete_rgbd_and_matched_policy_pass(self) -> None:
        report = audit_robotwin_dinov2_inputs(
            dataset_dir=self.dataset_dir,
            field_zarr=self.field_zarr,
            expected_episodes=2,
            expected_query_points=4,
        )
        self.assertEqual(report["status"], "pass")
        self.assertTrue(report["eligible_for_visibility_aware_dinov2_zarr"])
        self.assertTrue(report["matched_policy"]["frame_mapping_matches"])
        self.assertEqual(report["blocking_reasons"], [])

    def test_missing_depth_is_fail_closed(self) -> None:
        path = self.data_dir / "episode1.hdf5"
        with h5py.File(path, "a") as root:
            del root["observation/front_camera/depth"]
        report = audit_robotwin_dinov2_inputs(
            dataset_dir=self.dataset_dir,
            field_zarr=self.field_zarr,
            expected_query_points=4,
        )
        self.assertEqual(report["status"], "blocked")
        codes = {item["code"] for item in report["blocking_reasons"]}
        self.assertIn("missing_dense_depth", codes)
        self.assertTrue(report["no_depth_synthesis_or_sparse_pointcloud_substitution"])

    def test_policy_episode_mismatch_is_blocked(self) -> None:
        root = zarr.open(str(self.field_zarr), mode="a")
        root["meta/episode_ends"][:] = np.asarray([2, 4], dtype=np.int64)
        report = audit_robotwin_dinov2_inputs(
            dataset_dir=self.dataset_dir,
            field_zarr=self.field_zarr,
            expected_query_points=4,
        )
        codes = {item["code"] for item in report["blocking_reasons"]}
        self.assertIn("policy_episode_length_mismatch", codes)
        self.assertEqual(report["status"], "blocked")

    def test_rgb_depth_resolution_mismatch_is_blocked(self) -> None:
        path = self.data_dir / "episode0.hdf5"
        with h5py.File(path, "a") as root:
            camera = root["observation/head_camera"]
            del camera["depth"]
            camera.create_dataset(
                "depth", data=np.ones((self.lengths[0], 3, 5), dtype=np.float32)
            )
        report = audit_robotwin_dinov2_inputs(
            dataset_dir=self.dataset_dir,
            field_zarr=self.field_zarr,
            expected_query_points=4,
        )
        codes = {item["code"] for item in report["blocking_reasons"]}
        self.assertIn("rgb_depth_shape_mismatch", codes)

    def test_limited_smoke_cannot_claim_complete_eligibility(self) -> None:
        report = audit_robotwin_dinov2_inputs(
            dataset_dir=self.dataset_dir,
            field_zarr=self.field_zarr,
            expected_query_points=4,
            max_episodes=1,
            max_frames_per_episode=1,
        )
        self.assertEqual(report["status"], "blocked")
        self.assertFalse(report["scope"]["complete_dataset_audit"])
        codes = {item["code"] for item in report["blocking_reasons"]}
        self.assertIn("incomplete_smoke_scope", codes)


if __name__ == "__main__":
    unittest.main()
