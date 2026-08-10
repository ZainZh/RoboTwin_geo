from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import zarr

from build_semantic_policy_ablation_zarr import transform_semantic_cloud
from validate_semantic_policy_routes import ROUTE_NAMES, validate_matched_routes


class ValidateSemanticPolicyRoutesTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.shuffle_seed = 91
        frames, points = 3, 4
        self.xyz = np.arange(frames * points * 3, dtype=np.float32).reshape(frames, points, 3)
        embeddings = np.linspace(-1.0, 1.0, frames * points * 4, dtype=np.float32).reshape(
            frames, points, 4
        )
        field = np.concatenate([self.xyz, embeddings], axis=-1)
        raw = np.zeros((frames, points, 2), dtype=np.float32)
        raw[..., 0] = np.arange(frames * points, dtype=np.float32).reshape(frames, points)
        probabilities = np.exp(raw - raw.max(axis=-1, keepdims=True))
        probabilities /= probabilities.sum(axis=-1, keepdims=True)
        partprob = np.concatenate([self.xyz, probabilities.astype(np.float32)], axis=-1)
        routes = {
            "field": field,
            "xyz": self.xyz,
            "partprob": partprob,
            "uniform": transform_semantic_cloud(partprob, mode="uniform_prob"),
            "shuffled": transform_semantic_cloud(
                partprob, mode="shuffled_prob", shuffle_seed=self.shuffle_seed
            ),
        }
        self.paths = {}
        for route in ROUTE_NAMES:
            path = self.root / f"{route}.zarr"
            root = zarr.open(str(path), mode="w")
            data = root.create_group("data")
            meta = root.create_group("meta")
            data.array("action", np.arange(frames * 14, dtype=np.float32).reshape(frames, 14))
            data.array("state", np.arange(frames * 14, dtype=np.float32).reshape(frames, 14))
            data.array("point_cloud", np.arange(frames * points * 6, dtype=np.float32).reshape(frames, points, 6))
            data.array("semantic_point_cloud_A", routes[route], chunks=(2, points, routes[route].shape[-1]))
            meta.array("episode_ends", np.asarray([1, frames], dtype=np.int64))
            self.paths[route] = path

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_valid_routes_pass_all_fairness_checks(self) -> None:
        report = validate_matched_routes(
            self.paths,
            batch_frames=2,
            shuffle_seed=self.shuffle_seed,
            expected_query_points=4,
            expected_field_feature_dim=4,
        )
        self.assertEqual(report["status"], "pass")
        result = report["semantic_arrays"]["semantic_point_cloud_A"]
        self.assertEqual(result["channels"], {
            "field": 7,
            "xyz": 3,
            "partprob": 5,
            "uniform": 5,
            "shuffled": 5,
        })
        self.assertTrue(result["uniform_exact"])
        self.assertTrue(result["shuffle_multiset_preserved"])
        self.assertTrue(result["shuffle_seeded_derangement_exact"])
        self.assertEqual(result["shuffle_fixed_indices"], 0)

    def test_invariant_action_mismatch_fails(self) -> None:
        root = zarr.open(str(self.paths["xyz"]), mode="a")
        root["data/action"][1, 0] += 1.0
        with self.assertRaisesRegex(ValueError, "xyz:data/action value mismatch"):
            validate_matched_routes(self.paths, shuffle_seed=self.shuffle_seed)

    def test_query_xyz_mismatch_fails(self) -> None:
        root = zarr.open(str(self.paths["uniform"]), mode="a")
        root["data/semantic_point_cloud_A"][0, 0, 0] += 1.0
        with self.assertRaisesRegex(ValueError, "uniform:semantic_point_cloud_A query XYZ mismatch"):
            validate_matched_routes(self.paths, shuffle_seed=self.shuffle_seed)

    def test_non_deranged_shuffle_fails(self) -> None:
        part = zarr.open(str(self.paths["partprob"]), mode="r")
        shuffled = zarr.open(str(self.paths["shuffled"]), mode="a")
        shuffled["data/semantic_point_cloud_A"][0] = part[
            "data/semantic_point_cloud_A"
        ][0]
        with self.assertRaisesRegex(ValueError, "seeded per-frame derangement"):
            validate_matched_routes(self.paths, shuffle_seed=self.shuffle_seed)


if __name__ == "__main__":
    unittest.main()
