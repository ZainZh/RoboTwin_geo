from __future__ import annotations

import argparse
import copy
import random
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from semantic_field_training_state import (
    EVALUATION_FORMAT,
    TRAINING_STATE_FORMAT,
    load_training_state,
    make_restore_checkpoint,
    validate_training_state,
)
from train_semantic_field_loss_ablation_safe import save_weights_only_checkpoint


class _Model(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.semantic_adapter = torch.nn.Linear(3, 4)
        self.pose_adapter = torch.nn.Linear(3, 4)
        self.semantic_local_fusion = torch.nn.Linear(4, 4)
        self.pose_local_fusion = torch.nn.Linear(4, 4)
        self.occupancy_local_fusion = torch.nn.Linear(4, 4)
        self.semantic_decoder = torch.nn.Linear(4, 2)
        self.pose_decoder = torch.nn.Linear(4, 2)
        self.occupancy_decoder = torch.nn.Linear(4, 1)
        self.feature_extractor = SimpleNamespace(
            freeze_encoder=True, encoder=torch.nn.Identity()
        )


def _args(
    root: Path,
    *,
    seed: int = 7,
    num_workers: int = 4,
    batch_size: int = 2,
    epochs: int = 5,
) -> argparse.Namespace:
    return argparse.Namespace(
        output_dir=root,
        run_name="resume_run",
        epochs=epochs,
        seed=seed,
        train_mode="semantic",
        batch_size=batch_size,
        num_workers=num_workers,
        sem_ce_weight=1.0,
        resume=None,
        resume_additional_epochs=0,
        resume_from=None,
        auto_resume=True,
    )


def _assert_tree_equal(test: unittest.TestCase, left, right) -> None:
    if isinstance(left, torch.Tensor):
        test.assertTrue(torch.equal(left, right))
    elif isinstance(left, dict):
        test.assertEqual(set(left), set(right))
        for key in left:
            _assert_tree_equal(test, left[key], right[key])
    elif isinstance(left, (list, tuple)):
        test.assertEqual(len(left), len(right))
        for item_left, item_right in zip(left, right):
            _assert_tree_equal(test, item_left, item_right)
    else:
        test.assertEqual(left, right)


class TestSemanticFieldTrainingState(unittest.TestCase):
    def test_checkpoint_pair_is_weights_only_and_restores_optimizer_and_rng(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run_dir = root / "resume_run"
            model = _Model()
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
            scaler = torch.amp.GradScaler("cpu", enabled=False)

            loss = sum(parameter.square().sum() for parameter in model.parameters())
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            random.seed(101)
            np.random.seed(202)
            torch.manual_seed(303)
            save_weights_only_checkpoint(
                run_dir / "last.pt",
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                epoch=2,
                global_step=16,
                best_sem_score=0.8,
                best_pose_score=-1.0,
                best_joint_score=0.8,
                args=_args(root),
                canonical_label_names=["Handle", "Head"],
            )

            evaluation = torch.load(
                run_dir / "last.pt", map_location="cpu", weights_only=True
            )
            resumed = torch.load(
                run_dir / "resume.pt", map_location="cpu", weights_only=True
            )
            self.assertEqual(EVALUATION_FORMAT, evaluation["format"])
            self.assertIs(evaluation["training_resume_supported"], False)
            self.assertNotIn("optimizer_state_dict", evaluation)
            self.assertEqual(TRAINING_STATE_FORMAT, resumed["format"])
            self.assertIs(resumed["training_resume_supported"], True)
            self.assertEqual(2, resumed["epoch"])
            self.assertEqual(16, resumed["global_step"])
            self.assertFalse(list(run_dir.glob(".*.tmp-*")))

            expected_python = random.random()
            expected_numpy = float(np.random.random())
            expected_torch = torch.rand(4)
            expected_model = {
                key: value.clone() for key, value in model.state_dict().items()
            }
            expected_optimizer = copy.deepcopy(resumed["optimizer_state_dict"])

            with torch.no_grad():
                for parameter in model.parameters():
                    parameter.add_(10.0)
            random.seed(1)
            np.random.seed(2)
            torch.manual_seed(3)
            other_optimizer = torch.optim.AdamW(model.parameters(), lr=0.5)
            other_scaler = torch.amp.GradScaler("cpu", enabled=False)

            def original_restore(checkpoint, *, model, optimizer, scaler, device):
                del device
                projector = checkpoint["projector_state_dict"]
                for name in (
                    "semantic_adapter", "pose_adapter", "semantic_local_fusion",
                    "pose_local_fusion", "occupancy_local_fusion",
                    "semantic_decoder", "pose_decoder", "occupancy_decoder",
                ):
                    getattr(model, name).load_state_dict(projector[name])
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                scaler.load_state_dict(checkpoint["scaler_state_dict"])
                return checkpoint["epoch"], checkpoint["global_step"], {
                    "sem": checkpoint["best_sem_score"],
                    "pose": checkpoint["best_pose_score"],
                    "joint": checkpoint["best_joint_score"],
                }

            restore = make_restore_checkpoint(original_restore)
            metadata = restore(
                load_training_state(run_dir / "resume.pt"),
                model=model,
                optimizer=other_optimizer,
                scaler=other_scaler,
                device=torch.device("cpu"),
            )
            self.assertEqual((2, 16), metadata[:2])
            _assert_tree_equal(self, expected_model, model.state_dict())
            _assert_tree_equal(self, expected_optimizer, other_optimizer.state_dict())
            self.assertEqual(expected_python, random.random())
            self.assertEqual(expected_numpy, float(np.random.random()))
            self.assertTrue(torch.equal(expected_torch, torch.rand(4)))

    def test_identity_rejects_seed_change_and_legacy_eval_checkpoint(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run_dir = root / "resume_run"
            model = _Model()
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
            scaler = torch.amp.GradScaler("cpu", enabled=False)
            save_weights_only_checkpoint(
                run_dir / "last.pt", model=model, optimizer=optimizer,
                scaler=scaler, epoch=1, global_step=8,
                best_sem_score=0.5, best_pose_score=-1.0,
                best_joint_score=0.5, args=_args(root),
                canonical_label_names=["Handle", "Head"],
            )
            validate_training_state(run_dir / "resume.pt", _args(root))
            validate_training_state(
                run_dir / "resume.pt", _args(root, num_workers=2)
            )
            validate_training_state(
                run_dir / "resume.pt", _args(root, epochs=3)
            )
            with self.assertRaisesRegex(RuntimeError, "may only stay fixed or decrease"):
                validate_training_state(run_dir / "resume.pt", _args(root, epochs=6))
            with self.assertRaisesRegex(RuntimeError, "scientific config"):
                validate_training_state(
                    run_dir / "resume.pt", _args(root, batch_size=3)
                )
            tampered = torch.load(
                run_dir / "resume.pt", map_location="cpu", weights_only=True
            )
            tampered["run_identity"]["config"]["num_workers"] = 99
            tampered_path = run_dir / "tampered_resume.pt"
            torch.save(tampered, tampered_path)
            with self.assertRaisesRegex(RuntimeError, "SHA mismatch"):
                validate_training_state(tampered_path, _args(root, num_workers=2))
            with self.assertRaisesRegex(RuntimeError, "identity mismatch"):
                validate_training_state(run_dir / "resume.pt", _args(root, seed=8))
            with self.assertRaisesRegex(RuntimeError, "not an exact training-state"):
                load_training_state(run_dir / "last.pt")



if __name__ == "__main__":
    unittest.main()
