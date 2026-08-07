from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch
from torch import nn

from .ndf_adapter import NdfFunctionalFrameAdapter
from .train_ndf_functional_frame_head import (
    frame_loss,
    load_frame_model,
    resolve_object_split,
    select_training_episode_indices,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
NDF_ROOT = REPO_ROOT / "policy" / "DP3" / "third_party"
if str(NDF_ROOT) not in sys.path:
    sys.path.insert(0, str(NDF_ROOT))

import ndf_robot.model.vnn_occupancy_net_pointnet_dgcnn as vnn  # noqa: E402


class NdfFunctionalFrameHeadTest(unittest.TestCase):
    def test_explicit_object_split_is_complete_unique_and_disjoint(self):
        split = resolve_object_split(
            0,
            train_ids=[2, 3, 4, 7],
            validation_ids=[1, 6],
            test_ids=[0, 5],
        )
        self.assertEqual(split["train"], (2, 3, 4, 7))
        self.assertEqual(split["validation"], (1, 6))
        self.assertEqual(split["test"], (0, 5))
        with self.assertRaisesRegex(ValueError, "supplied together"):
            resolve_object_split(0, train_ids=[2], validation_ids=[1])
        with self.assertRaisesRegex(ValueError, "disjoint"):
            resolve_object_split(
                0, train_ids=[2, 3], validation_ids=[1, 2], test_ids=[0]
            )
        with self.assertRaisesRegex(ValueError, "unique"):
            resolve_object_split(
                0, train_ids=[2, 2], validation_ids=[1], test_ids=[0]
            )

    def test_episode_budget_keeps_complete_frames_and_all_train_objects(self):
        import numpy as np

        # Two train objects, two episodes each, three frames per episode. The
        # validation/test episodes use disjoint objects and must never enter the
        # selected training rows.
        shoe = np.repeat(np.asarray([2, 2, 3, 3, 1, 0]), 3)
        episode = np.repeat(np.asarray([20, 21, 30, 31, 10, 0]), 3)
        train = np.flatnonzero(np.isin(shoe, [2, 3]))
        validation = np.flatnonzero(shoe == 1)
        test = np.flatnonzero(shoe == 0)
        selected, diagnostic = select_training_episode_indices(
            train,
            shoe,
            episode,
            (2, 3),
            episodes_per_object=1,
            fold=0,
            seed=7,
        )
        repeated, repeated_diagnostic = select_training_episode_indices(
            train,
            shoe,
            episode,
            (2, 3),
            episodes_per_object=1,
            fold=0,
            seed=7,
        )
        self.assertTrue(np.array_equal(selected, repeated))
        self.assertEqual(diagnostic, repeated_diagnostic)
        self.assertEqual(set(shoe[selected].tolist()), {2, 3})
        self.assertFalse(np.intersect1d(selected, validation).size)
        self.assertFalse(np.intersect1d(selected, test).size)
        for selected_episode in np.unique(episode[selected]):
            self.assertTrue(
                np.array_equal(
                    selected[episode[selected] == selected_episode],
                    np.flatnonzero(episode == selected_episode),
                )
            )
        self.assertEqual(diagnostic["train_episodes_after"], 2)
        self.assertEqual(diagnostic["train_samples_after"], 6)

    def test_episode_budget_rejects_partial_episode_and_all_is_identity(self):
        import numpy as np

        shoe = np.repeat(np.asarray([2, 3]), 3)
        episode = np.repeat(np.asarray([20, 30]), 3)
        train = np.arange(6)
        selected, diagnostic = select_training_episode_indices(
            train,
            shoe,
            episode,
            (2, 3),
            episodes_per_object=None,
            fold=0,
            seed=0,
        )
        self.assertTrue(np.array_equal(selected, train))
        self.assertEqual(diagnostic["mode"], "all_complete_episodes")
        with self.assertRaisesRegex(ValueError, "partial episode"):
            select_training_episode_indices(
                train[:-1],
                shoe,
                episode,
                (2, 3),
                episodes_per_object=1,
                fold=0,
                seed=0,
            )

    def test_episode_budget_rejects_cross_object_episode_and_missing_object(self):
        import numpy as np

        with self.assertRaisesRegex(ValueError, "spans object identities"):
            select_training_episode_indices(
                np.arange(4),
                np.asarray([2, 2, 3, 3]),
                np.asarray([20, 21, 21, 30]),
                (2, 3),
                episodes_per_object=1,
                fold=0,
                seed=0,
            )

        shoe = np.repeat(np.asarray([2, 3]), 3)
        episode = np.repeat(np.asarray([20, 30]), 3)
        with self.assertRaisesRegex(ValueError, "object coverage mismatch"):
            select_training_episode_indices(
                np.flatnonzero(shoe == 2),
                shoe,
                episode,
                (2, 3),
                episodes_per_object=1,
                fold=0,
                seed=0,
            )

    def test_decoder_preserves_legacy_shape_and_adds_two_vector_shape(self):
        model = vnn.VNNOccNet(
            latent_dim=12,
            return_features=True,
            return_vector_features=True,
            vector_feature_dim=4,
            acts="last",
        )
        latent = torch.randn(2, 12, 3)
        query = torch.randn(2, 7, 3)
        _, legacy = model.forward_latent(
            latent, query, return_vector_features=True
        )
        self.assertEqual(tuple(legacy.shape), (2, 7, 3))
        model.decoder.fc_vec_beta = nn.Linear(12, 4)
        _, full = model.forward_latent(
            latent, query, return_vector_features=True
        )
        self.assertEqual(tuple(full.shape), (2, 7, 2, 3))

    def test_frame_convention_is_right_handed_xyz(self):
        vectors = torch.zeros(2, 5, 2, 3)
        vectors[..., 0, 2] = 1.0  # alpha = Z
        vectors[..., 1, 1] = 1.0  # beta = Y
        frame = NdfFunctionalFrameAdapter.frame_from_vectors(vectors)
        expected = torch.eye(3)[None].expand(2, -1, -1)
        self.assertTrue(torch.allclose(frame, expected, atol=1e-6))
        self.assertTrue(torch.allclose(torch.det(frame), torch.ones(2), atol=1e-6))

    def test_full_frame_loss_is_signed_and_rotation_sensitive(self):
        target = torch.eye(3)[None]
        correct = torch.zeros(1, 6, 2, 3)
        correct[..., 0, 2] = 1.0
        correct[..., 1, 1] = 1.0
        wrong = correct.clone()
        wrong[..., 1, 1] = -1.0
        correct_loss, _ = frame_loss(
            correct, target, axis_weight=0.5, orthogonality_weight=0.05
        )
        wrong_loss, _ = frame_loss(
            wrong, target, axis_weight=0.5, orthogonality_weight=0.05
        )
        self.assertLess(float(correct_loss), 0.01)
        self.assertGreater(float(wrong_loss), float(correct_loss) + 1.0)

    def test_parallel_axes_have_finite_deterministic_fallback(self):
        vectors = torch.zeros(3, 4, 2, 3)
        vectors[..., 0, 2] = 1.0
        vectors[..., 1, 2] = 1.0
        first = NdfFunctionalFrameAdapter.frame_from_vectors(vectors)
        second = NdfFunctionalFrameAdapter.frame_from_vectors(vectors)
        self.assertTrue(torch.equal(first, second))
        self.assertTrue(torch.isfinite(first).all())
        self.assertTrue(torch.allclose(torch.det(first), torch.ones(3), atol=1e-6))

    def test_reverse_curriculum_preserves_hard_axis_and_trains_only_alpha(self):
        model = vnn.VNNOccNet(
            latent_dim=256,
            model_type="pointcloud",
            return_features=True,
            return_vector_features=True,
            vector_feature_dim=16,
            acts="last",
        )
        with torch.no_grad():
            model.decoder.fc_vec_alpha.weight.fill_(0.125)
            model.decoder.fc_vec_alpha.bias.fill_(0.25)
        import tempfile

        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "axis_y.pth"
            torch.save(model.state_dict(), checkpoint)
            promoted, trainable = load_frame_model(
                checkpoint,
                torch.device("cpu"),
                trainable_scope="alpha",
                promote_alpha_to_beta=True,
            )
        self.assertTrue(
            torch.equal(
                promoted.decoder.fc_vec_beta.weight,
                torch.full_like(promoted.decoder.fc_vec_beta.weight, 0.125),
            )
        )
        self.assertTrue(
            torch.equal(
                promoted.decoder.fc_vec_beta.bias,
                torch.full_like(promoted.decoder.fc_vec_beta.bias, 0.25),
            )
        )
        self.assertFalse(
            torch.equal(
                promoted.decoder.fc_vec_alpha.weight,
                promoted.decoder.fc_vec_beta.weight,
            )
        )
        self.assertEqual(
            {id(parameter) for parameter in trainable},
            {
                id(promoted.decoder.fc_vec_alpha.weight),
                id(promoted.decoder.fc_vec_alpha.bias),
            },
        )

    def test_random_initialization_is_capacity_matched_and_full_scope_only(self):
        model = vnn.VNNOccNet(
            latent_dim=256,
            model_type="pointcloud",
            return_features=True,
            return_vector_features=True,
            vector_feature_dim=16,
            acts="last",
        )
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.fill_(0.125)
        import tempfile

        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "constant.pth"
            torch.save(model.state_dict(), checkpoint)
            randomized, trainable = load_frame_model(
                checkpoint,
                torch.device("cpu"),
                trainable_scope="full",
                initialization="random",
            )
            with self.assertRaisesRegex(ValueError, "requires.*full"):
                load_frame_model(
                    checkpoint,
                    torch.device("cpu"),
                    trainable_scope="heads",
                    initialization="random",
                )
        self.assertEqual(
            sum(parameter.numel() for parameter in randomized.parameters()),
            sum(parameter.numel() for parameter in trainable),
        )
        self.assertFalse(
            torch.equal(
                randomized.decoder.fc_vec_alpha.weight,
                torch.full_like(randomized.decoder.fc_vec_alpha.weight, 0.125),
            )
        )

    def test_reinitialized_heads_preserve_backbone_and_remove_axis_bias(self):
        model = vnn.VNNOccNet(
            latent_dim=256,
            model_type="pointcloud",
            return_features=True,
            return_vector_features=True,
            vector_feature_dim=16,
            acts="last",
        )
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.fill_(0.125)
        original = {
            name: value.clone() for name, value in model.state_dict().items()
        }
        import tempfile

        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "constant.pth"
            torch.save(model.state_dict(), checkpoint)
            adapted, trainable = load_frame_model(
                checkpoint,
                torch.device("cpu"),
                trainable_scope="full",
                initialization="pretrained_reinit_heads",
            )
            with self.assertRaisesRegex(ValueError, "incompatible"):
                load_frame_model(
                    checkpoint,
                    torch.device("cpu"),
                    trainable_scope="full",
                    promote_alpha_to_beta=True,
                    initialization="pretrained_reinit_heads",
                )
        adapted_state = adapted.state_dict()
        for name, value in original.items():
            if name.startswith("decoder.fc_vec_alpha."):
                continue
            self.assertTrue(torch.equal(adapted_state[name], value), name)
        self.assertFalse(
            torch.equal(
                adapted.decoder.fc_vec_alpha.weight,
                original["decoder.fc_vec_alpha.weight"],
            )
        )
        self.assertTrue(
            torch.equal(
                adapted.decoder.fc_vec_alpha.bias,
                torch.zeros_like(adapted.decoder.fc_vec_alpha.bias),
            )
        )
        self.assertTrue(
            torch.equal(
                adapted.decoder.fc_vec_beta.bias,
                torch.zeros_like(adapted.decoder.fc_vec_beta.bias),
            )
        )
        self.assertEqual(
            sum(parameter.numel() for parameter in adapted.parameters()),
            sum(parameter.numel() for parameter in trainable),
        )


if __name__ == "__main__":
    unittest.main()
