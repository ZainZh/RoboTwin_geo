from __future__ import annotations

import os
import random
import sys
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from semantic_field_fast_path import (
    SEMANTIC_SAMPLE_KEYS,
    SEMANTIC_VIEW2_KEYS,
    make_zero_weight_aware_semantic_loss,
    make_rng_equivalent_query_sampler,
    semantic_only_collate_fn,
    semantic_only_encode_support,
    semantic_only_forward,
    semantic_only_probe_labels,
)
from train_semantic_field_loss_ablation_fast import _require_explicit_semantic_mode
from train_semantic_field_loss_ablation_safe import (
    _install_default_semantic_fast_path,
    _semantic_fast_path_requested,
)


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
RELEASE_ROOT = Path(
    os.environ.get(
        "SEMANTIC_FIELD_RELEASE_ROOT",
        REPO_ROOT / "include" / "3d_semantic_train" / "semantic_field_release",
    )
).resolve()
UTONIA_ROOT = Path(
    os.environ.get(
        "UTONIA_ROOT",
        REPO_ROOT / "include" / "3d_semantic_train" / "include" / "Utonia",
    )
).resolve()
DATASET_ROOT = Path(
    os.environ.get("PARTNEXT_DATASET_ROOT", "/home/zheng/Datasets/PartNext_mesh")
)
ALIAS_CONFIG = RELEASE_ROOT / "configs" / "hammer.json"


def _seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class _FakeMesh:
    bounds = np.asarray([[-1.0, -2.0, -3.0], [2.0, 3.0, 4.0]], dtype=np.float32)

    def __init__(self) -> None:
        self.contains_calls = 0

    def contains(self, points: np.ndarray) -> np.ndarray:
        self.contains_calls += 1
        np.random.random(3)
        return points[:, 0] > 0


def _fake_sample_surface(
    mesh: _FakeMesh, num_points: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    del mesh
    points = np.random.uniform(-1.0, 1.0, size=(num_points, 3)).astype(np.float32)
    return (
        points,
        np.zeros_like(points, dtype=np.float32),
        np.zeros_like(points, dtype=np.float32),
    )


class _CallRecorder:
    def __init__(self, result=None, *, fail: str | None = None) -> None:
        self.result = result
        self.fail = fail
        self.calls = 0

    def __call__(self, *args, **kwargs):
        self.calls += 1
        if self.fail is not None:
            raise AssertionError(self.fail)
        if callable(self.result):
            return self.result(*args, **kwargs)
        return self.result


class _ProjectorRecorder:
    def __init__(self) -> None:
        self.calls = 0

    def build(self, *, support_points, support_features):
        self.calls += 1
        return support_points + support_features


class TestSemanticFieldFastPath(unittest.TestCase):
    def assert_tree_equal(self, expected, actual, path: str = "root") -> None:
        if isinstance(expected, torch.Tensor):
            self.assertIsInstance(actual, torch.Tensor, path)
            self.assertEqual(expected.dtype, actual.dtype, path)
            self.assertEqual(tuple(expected.shape), tuple(actual.shape), path)
            self.assertTrue(torch.equal(expected, actual), path)
            return
        if isinstance(expected, np.ndarray):
            self.assertIsInstance(actual, np.ndarray, path)
            self.assertEqual(expected.dtype, actual.dtype, path)
            self.assertTrue(np.array_equal(expected, actual), path)
            return
        if isinstance(expected, dict):
            self.assertEqual(set(expected), set(actual), path)
            for key in expected:
                self.assert_tree_equal(expected[key], actual[key], f"{path}.{key}")
            return
        if isinstance(expected, (list, tuple)):
            self.assertEqual(len(expected), len(actual), path)
            for index, (left, right) in enumerate(zip(expected, actual)):
                self.assert_tree_equal(left, right, f"{path}[{index}]")
            return
        self.assertEqual(expected, actual, path)

    def test_query_sampler_preserves_points_and_rng_without_contains(self):
        if str(UTONIA_ROOT) not in sys.path:
            sys.path.insert(0, str(UTONIA_ROOT))
        if str(RELEASE_ROOT) not in sys.path:
            sys.path.insert(0, str(RELEASE_ROOT))
        from my_datasets import partnext_occupancy as occupancy_module

        original_sampler = occupancy_module._sample_queries
        original_surface_sampler = occupancy_module._sample_surface
        original_mesh = _FakeMesh()
        try:
            occupancy_module._sample_surface = _fake_sample_surface
            _seed_all(987)
            original_points, original_labels = original_sampler(
                original_mesh, 63, 0.75, 0.02, 0.15
            )
        finally:
            occupancy_module._sample_surface = original_surface_sampler
        original_next = np.random.random(16)

        fast_mesh = _FakeMesh()
        fast_sampler = make_rng_equivalent_query_sampler(_fake_sample_surface)
        _seed_all(987)
        fast_points, fast_labels = fast_sampler(
            fast_mesh, 63, 0.75, 0.02, 0.15
        )
        fast_next = np.random.random(16)

        self.assertTrue(np.array_equal(original_points, fast_points))
        self.assertTrue(np.array_equal(original_next, fast_next))
        self.assertEqual(1, original_mesh.contains_calls)
        self.assertEqual(0, fast_mesh.contains_calls)
        self.assertEqual(original_labels.shape, fast_labels.shape)
        self.assertTrue(np.all(fast_labels == 0))

    def test_probe_nearest_is_replaced_by_ignored_placeholders(self):
        points = np.ones((7, 3), dtype=np.float32)
        labels = semantic_only_probe_labels(
            points, np.zeros((4, 3)), np.arange(4), 0.1
        )
        self.assertTrue(np.array_equal(labels, np.full((7,), -1, dtype=np.int64)))

    def test_safe_entry_defaults_fast_for_future_semantic_ablation(self):
        for path in (str(UTONIA_ROOT), str(RELEASE_ROOT)):
            if path not in sys.path:
                sys.path.insert(0, path)
        from train import train_utonia_universal_field as trainer

        output = StringIO()
        with redirect_stdout(output):
            installed = _install_default_semantic_fast_path(
                trainer, ["--train-mode", "semantic"], {}
            )
        self.assertTrue(installed)
        self.assertIn("[semantic-fast-path]", output.getvalue())
        self.assertIn("safe-entry marker and identities verified", output.getvalue())
        self.assertIs(getattr(trainer, "SEMANTIC_FAST_PATH_ENABLED"), True)
        self.assertIs(trainer.UtoniaUniversalFieldNet.forward, semantic_only_forward)
        self.assertIs(
            trainer.UtoniaUniversalFieldNet.encode_support,
            semantic_only_encode_support,
        )

    def test_safe_entry_explicit_zero_disables_future_fast_install(self):
        self.assertFalse(
            _install_default_semantic_fast_path(
                object(),
                ["--train-mode=semantic"],
                {"SEMANTIC_FAST_PATH": "0"},
            )
        )
        self.assertFalse(
            _semantic_fast_path_requested(
                ["--train-mode", "pose"], {"SEMANTIC_FAST_PATH": "1"}
            )
        )

    def test_semantic_collate_drops_all_occ_and_probe_keys(self):
        item = {
            "utonia_input_view1": {"feat": torch.tensor([[1.0]])},
            "support_points_view1": torch.ones((2, 3)),
            "query_surface_points_view1": torch.ones((3, 3)),
            "query_surface_points_canonical": torch.ones((3, 3)),
            "query_surface_correspondence": torch.arange(3),
            "query_surface_labels": torch.arange(3),
            "category": "Hammer",
            "model_id": "m",
            "mesh_path": "m.glb",
            "query_occ_points_view1": torch.ones((5, 3)),
            "query_probe_points_view1": torch.ones((6, 3)),
        }

        def fake_utonia_collate(items):
            return {"feat": torch.cat([entry["feat"] for entry in items], dim=0)}

        batch = semantic_only_collate_fn([item, item], utonia_collate_fn=fake_utonia_collate)
        self.assertFalse(any("occ" in key or "probe" in key for key in batch))
        self.assertEqual((2, 2, 3), tuple(batch["support_points_view1"].shape))

    def test_semantic_support_never_calls_pose_modules(self):
        feature_extractor = _CallRecorder(
            lambda inputs, num_support_points: inputs["feat"][:, :num_support_points]
        )
        semantic_adapter = _CallRecorder(lambda features: features + 1)
        pose_adapter = _CallRecorder(fail="pose adapter must not run")
        semantic_projector = _ProjectorRecorder()
        pose_projector = _CallRecorder(fail="pose projector must not run")
        model = SimpleNamespace(
            feature_extractor=feature_extractor,
            semantic_adapter=semantic_adapter,
            pose_adapter=pose_adapter,
            semantic_projector=semantic_projector,
            pose_projector=pose_projector,
        )
        points = torch.zeros((1, 2, 3))
        cache = semantic_only_encode_support(
            model, {"feat": torch.ones((1, 2, 3))}, points
        )
        self.assertEqual({"semantic"}, set(cache))
        self.assertEqual(1, feature_extractor.calls)
        self.assertEqual(1, semantic_adapter.calls)
        self.assertEqual(1, semantic_projector.calls)
        self.assertEqual(0, pose_adapter.calls)
        self.assertEqual(0, pose_projector.calls)

    def test_forward_needs_no_occ_keys_and_rejects_nonsemantic_mode(self):
        class FakeModel:
            def __init__(self):
                self.calls = []

            def encode_view(self, **kwargs):
                self.calls.append(kwargs)
                return {"sem_logits": kwargs["query_surface_points"]}

        model = FakeModel()
        batch = {
            "utonia_input_view1": {},
            "support_points_view1": torch.zeros((1, 2, 3)),
            "query_surface_points_view1": torch.zeros((1, 4, 3)),
        }
        outputs = semantic_only_forward(model, batch)
        self.assertEqual({"view1"}, set(outputs))
        self.assertEqual((1, 0, 3), tuple(model.calls[0]["query_occ_points"].shape))
        self.assertFalse(model.calls[0]["compute_pose"])
        self.assertFalse(model.calls[0]["compute_occupancy"])
        with self.assertRaises(ValueError):
            semantic_only_forward(model, batch, train_mode="joint")

    def test_fast_entry_requires_explicit_semantic_mode(self):
        _require_explicit_semantic_mode(["--train-mode", "semantic"])
        _require_explicit_semantic_mode(["--train-mode=semantic"])
        with self.assertRaises(SystemExit):
            _require_explicit_semantic_mode([])
        with self.assertRaises(SystemExit):
            _require_explicit_semantic_mode(["--train-mode", "pose"])

    @unittest.skipUnless(
        DATASET_ROOT.joinpath("Hammer", "annotation.jsonl").is_file()
        and ALIAS_CONFIG.is_file()
        and UTONIA_ROOT.is_dir(),
        "real Hammer/Utonia/release data are unavailable",
    )
    def test_real_hammer_semantic_inputs_and_rng_are_bitwise_equal(self):
        for path in (str(UTONIA_ROOT), str(RELEASE_ROOT)):
            if path not in sys.path:
                sys.path.insert(0, path)
        from my_datasets import partnext_universal_field as dataset_module
        from my_datasets.partnext_canonical_field import (
            build_partnext_canonical_label_space,
        )
        from my_datasets.partnext_occupancy import _sample_surface
        from my_datasets.partnext_universal_field import (
            PartNextUniversalFieldDataset,
            partnext_universal_field_collate_fn,
        )

        labels = build_partnext_canonical_label_space(
            dataset_root=DATASET_ROOT,
            categories=["Hammer"],
            label_level="config",
            alias_config_path=ALIAS_CONFIG,
            ignore_labels=("Other",),
        )
        dataset = PartNextUniversalFieldDataset(
            dataset_root=DATASET_ROOT,
            split="train",
            categories=["Hammer"],
            num_support_points=5000,
            num_query_surface_points=2048,
            num_query_occ_points=1536,
            num_query_probe_points=1024,
            balanced_sampling_ratio=0.0,
            query_sampling_ratio=0.75,
            val_ratio=0.1,
            test_ratio=0.0,
            split_seed=42,
            label_level="config",
            alias_config_path=ALIAS_CONFIG,
            canonical_label_names=labels,
            rotation_mode="so3",
            jitter_std=0.005,
            second_view=True,
        )
        record = dataset.records[0]
        original_query_sampler = dataset_module._sample_queries
        original_nearest = dataset_module._assign_nearest_surface_labels
        seed = 20260807
        try:
            _seed_all(seed)
            original_sample = dataset._load_sample(record)
            original_np_next = np.random.random(16)
            original_torch_next = torch.rand(16)

            dataset_module._sample_queries = make_rng_equivalent_query_sampler(
                _sample_surface
            )
            dataset_module._assign_nearest_surface_labels = semantic_only_probe_labels
            _seed_all(seed)
            fast_sample = dataset._load_sample(record)
            fast_np_next = np.random.random(16)
            fast_torch_next = torch.rand(16)
        finally:
            dataset_module._sample_queries = original_query_sampler
            dataset_module._assign_nearest_surface_labels = original_nearest

        for key in SEMANTIC_SAMPLE_KEYS + SEMANTIC_VIEW2_KEYS:
            self.assert_tree_equal(original_sample[key], fast_sample[key], key)
        self.assertTrue(np.array_equal(original_np_next, fast_np_next))
        self.assertTrue(torch.equal(original_torch_next, fast_torch_next))

        original_batch = partnext_universal_field_collate_fn([original_sample])
        fast_batch = semantic_only_collate_fn(
            [fast_sample],
            utonia_collate_fn=dataset_module.UTONIA.data.collate_fn,
        )
        for key in fast_batch:
            self.assert_tree_equal(original_batch[key], fast_batch[key], f"batch.{key}")
        self.assertFalse(any("occ" in key or "probe" in key for key in fast_batch))

    def test_zero_weight_loss_and_gradients_match_release_total(self):
        for path in (str(UTONIA_ROOT), str(RELEASE_ROOT)):
            if path not in sys.path:
                sys.path.insert(0, path)
        from train import train_utonia_universal_field as trainer

        fast_loss = make_zero_weight_aware_semantic_loss(trainer)
        torch.manual_seed(44)
        labels = torch.tensor(
            [[0, 0, 1, 1, 0, 1, 0, 1], [1, 0, 1, 0, 1, 0, 1, 0]],
            dtype=torch.long,
        )
        template = {
            "logits1": torch.randn(2, 8, 2),
            "embedding1": torch.randn(2, 8, 6),
            "logits2": torch.randn(2, 8, 2),
            "embedding2": torch.randn(2, 8, 6),
        }

        def make_inputs():
            tensors = {
                key: value.clone().requires_grad_(True)
                for key, value in template.items()
            }
            outputs = {
                "view1": {
                    "sem_logits": tensors["logits1"],
                    "sem_embeddings": tensors["embedding1"],
                },
                "view2": {
                    "sem_logits": tensors["logits2"],
                    "sem_embeddings": tensors["embedding2"],
                },
            }
            batch = {
                "query_surface_labels": labels,
                "query_surface_points_view1": torch.zeros((2, 8, 3)),
                "query_surface_points_canonical": torch.zeros((2, 8, 3)),
                "query_surface_correspondence": torch.arange(8).repeat(2, 1),
            }
            return tensors, outputs, batch

        for ce_weight, supcon_weight in ((0.0, 0.2), (1.0, 0.0), (0.0, 0.0)):
            args = SimpleNamespace(
                train_mode="semantic",
                sem_ce_weight=ce_weight,
                sem_contrastive_weight=supcon_weight,
                sem_consistency_weight=0.1,
                sem_label_smoothing=0.0,
                sem_contrastive_temperature=0.1,
                sem_contrastive_max_points=4,
            )
            original_tensors, original_outputs, original_batch = make_inputs()
            fast_tensors, fast_outputs, fast_batch = make_inputs()
            torch.manual_seed(7788)
            original_terms = trainer._compute_loss_terms(
                original_outputs, original_batch, args
            )
            original_rng_state = torch.get_rng_state()
            torch.manual_seed(7788)
            fast_terms = fast_loss(fast_outputs, fast_batch, args)
            fast_rng_state = torch.get_rng_state()
            self.assertTrue(torch.equal(original_terms["loss"], fast_terms["loss"]))
            self.assertTrue(torch.equal(original_rng_state, fast_rng_state))

            original_grads = torch.autograd.grad(
                original_terms["loss"],
                list(original_tensors.values()),
                allow_unused=True,
            )
            fast_grads = torch.autograd.grad(
                fast_terms["loss"],
                list(fast_tensors.values()),
                allow_unused=True,
            )
            for original_grad, fast_grad, tensor in zip(
                original_grads, fast_grads, fast_tensors.values()
            ):
                expected = (
                    torch.zeros_like(tensor)
                    if original_grad is None
                    else original_grad
                )
                actual = torch.zeros_like(tensor) if fast_grad is None else fast_grad
                self.assertTrue(torch.equal(expected, actual))

    def test_z_installer_sets_verified_marker_and_function_identities(self):
        for path in (str(UTONIA_ROOT), str(RELEASE_ROOT)):
            if path not in sys.path:
                sys.path.insert(0, path)
        from train import train_utonia_universal_field as trainer
        from semantic_field_fast_path import install_semantic_only_fast_path

        install_semantic_only_fast_path(trainer)
        self.assertIs(getattr(trainer, "SEMANTIC_FAST_PATH_ENABLED"), True)
        self.assertIs(
            trainer.UtoniaUniversalFieldNet.forward,
            semantic_only_forward,
        )
        self.assertIs(
            trainer.UtoniaUniversalFieldNet.encode_support,
            semantic_only_encode_support,
        )


if __name__ == "__main__":
    unittest.main()
