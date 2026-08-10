import argparse
import tempfile
import unittest
from pathlib import Path

import numpy as np

import build_utonia128_policy_zarr
from utonia_projection_utils import (
    build_frozen_orthogonal_projection,
    load_projection_artifact,
    project_utonia_cloud,
    save_projection_artifact,
)


class TestUtoniaProjectionUtils(unittest.TestCase):
    def test_projection_is_reproducible_orthogonal_and_norm_scaled(self):
        first = build_frozen_orthogonal_projection(12, 4, seed=17)
        second = build_frozen_orthogonal_projection(12, 4, seed=17)
        self.assertTrue(np.array_equal(first, second))
        self.assertTrue(
            np.allclose(first.T @ first, np.eye(4, dtype=np.float32) * 3.0, atol=1e-5)
        )

    def test_artifact_round_trip_and_cloud_projection_preserve_xyz(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "projection.npz"
            matrix = build_frozen_orthogonal_projection(6, 2, seed=3)
            save_projection_artifact(path, matrix, seed=3)
            artifact = load_projection_artifact(
                path, expected_input_dim=6, expected_output_dim=2
            )
            cloud = np.arange(18, dtype=np.float32).reshape(2, 9)
            result = project_utonia_cloud(cloud, artifact["matrix"])
        self.assertEqual((2, 5), result.shape)
        self.assertTrue(np.array_equal(cloud[:, :3], result[:, :3]))
        self.assertTrue(np.isfinite(result).all())

    def test_builder_keeps_policy_arrays_and_query_xyz_bitwise(self):
        try:
            import zarr
        except ImportError:
            self.skipTest("zarr is unavailable")
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "source-utonia.zarr"
            destination = root / "output-utonia128.zarr"
            projection = root / "projection.npz"
            source_meta = root / "source-utonia_meta.json"
            source_meta.write_text('{"format":"test"}\n', encoding="utf-8")
            zroot = zarr.open(str(source), mode="w")
            data = zroot.create_group("data")
            meta = zroot.create_group("meta")
            generator = np.random.default_rng(5)
            data.create_dataset(
                "action", data=generator.standard_normal((4, 14)).astype(np.float32)
            )
            data.create_dataset(
                "state", data=generator.standard_normal((4, 14)).astype(np.float32)
            )
            data.create_dataset(
                "point_cloud", data=generator.standard_normal((4, 8, 6)).astype(np.float32)
            )
            cloud = generator.standard_normal((4, 3, 9)).astype(np.float32)
            data.create_dataset("utonia_point_cloud_A", data=cloud, chunks=(2, 3, 9))
            meta.create_dataset("episode_ends", data=np.asarray([2, 4], dtype=np.int64))
            args = argparse.Namespace(
                source_zarr=source,
                output_zarr=destination,
                projection_path=projection,
                source_meta=source_meta,
                output_meta=None,
                key="utonia_point_cloud_A",
                output_dim=2,
                seed=7,
                batch_frames=2,
            )
            report = build_utonia128_policy_zarr.build(args)
            output = zarr.open(str(destination), mode="r")
            projected = np.asarray(output["data/utonia_point_cloud_A"][:])
            for key in ("action", "state", "point_cloud"):
                self.assertTrue(
                    np.array_equal(
                        np.asarray(zroot[f"data/{key}"][:]),
                        np.asarray(output[f"data/{key}"][:]),
                    )
                )
        self.assertEqual((4, 3, 5), projected.shape)
        self.assertTrue(np.array_equal(cloud[..., :3], projected[..., :3]))
        self.assertTrue(report["formal_comparison_eligible"])
        self.assertFalse(report["projection"]["test_data_used"])


if __name__ == "__main__":
    unittest.main()
