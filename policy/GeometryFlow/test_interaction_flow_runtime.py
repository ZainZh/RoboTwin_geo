from __future__ import annotations

import numpy as np
import torch

from .interaction_flow_tokens import InteractionFlowOutput
from .interaction_flow_runtime import InteractionFlowRuntime
from .online_ndf_functional_frame import OnlineFunctionalFrameEstimate


def test_runtime_matches_direct_checkpoint_model(tmp_path):
    from .train_interaction_flow_tokens import build_model

    generator = np.random.default_rng(3)
    normalization = {
        "state_mean": np.zeros(20, dtype=np.float32),
        "state_std": np.ones(20, dtype=np.float32),
        "action_mean": np.zeros(14, dtype=np.float32),
        "action_std": np.ones(14, dtype=np.float32),
        "xyz_mean": np.zeros(3, dtype=np.float32),
        "xyz_std": np.ones(3, dtype=np.float32),
    }
    model = build_model(
        "interaction_flow",
        horizon=6,
        flow_steps=4,
        num_anchors=4,
        feature_dim=32,
        layers=1,
        flow_parameterization="free",
        xyz_std=normalization["xyz_std"],
    )
    checkpoint = tmp_path / "model.pt"
    torch.save(
        {
            "model": model.state_dict(),
            "condition": "interaction_flow",
            "horizon": 6,
            "flow_steps": 4,
            "num_anchors": 4,
            "feature_dim": 32,
            "layers": 1,
            "flow_parameterization": "free",
            "normalization": normalization,
        },
        checkpoint,
    )
    runtime = InteractionFlowRuntime(
        checkpoint, device="cpu", observation_points=16
    )
    points_a = generator.normal(size=(16, 3)).astype(np.float32)
    points_b = generator.normal(size=(16, 3)).astype(np.float32)
    state = generator.normal(size=20).astype(np.float32)
    action = runtime.predict(
        operated_point_cloud=points_a,
        target_point_cloud=points_b,
        eef_state20=state,
    )
    assert action.shape == (6, 14)
    assert np.all(np.isfinite(action))
    assert runtime.last_diagnostic["anchor_pair_distance_cm"] is not None


def test_runtime_preserves_aligned_rgb_channels(tmp_path):
    from .train_interaction_flow_tokens import build_model

    generator = np.random.default_rng(9)
    normalization = {
        "state_mean": np.zeros(20, dtype=np.float32),
        "state_std": np.ones(20, dtype=np.float32),
        "action_mean": np.zeros(14, dtype=np.float32),
        "action_std": np.ones(14, dtype=np.float32),
        "xyz_mean": np.zeros(3, dtype=np.float32),
        "xyz_std": np.ones(3, dtype=np.float32),
        "point_feature_mean": np.zeros(3, dtype=np.float32),
        "point_feature_std": np.ones(3, dtype=np.float32),
    }
    model = build_model(
        "interaction_flow",
        horizon=2,
        flow_steps=2,
        num_anchors=4,
        feature_dim=32,
        layers=1,
        flow_parameterization="free",
        xyz_std=normalization["xyz_std"],
        point_channels=6,
    )
    checkpoint = tmp_path / "rgb_model.pt"
    torch.save(
        {
            "model": model.state_dict(),
            "condition": "interaction_flow",
            "horizon": 2,
            "flow_steps": 2,
            "num_anchors": 4,
            "feature_dim": 32,
            "layers": 1,
            "point_channels": 6,
            "normalization": normalization,
        },
        checkpoint,
    )
    runtime = InteractionFlowRuntime(
        checkpoint, device="cpu", observation_points=16
    )
    cloud_a = generator.normal(size=(24, 6)).astype(np.float32)
    cloud_b = generator.normal(size=(24, 6)).astype(np.float32)
    action = runtime.predict(
        operated_point_cloud=cloud_a,
        target_point_cloud=cloud_b,
        eef_state20=np.zeros(20, dtype=np.float32),
    )
    assert action.shape == (2, 14)
    assert runtime.model.point_encoder.network[0].in_features == 6


def test_runtime_target_priority_retains_sparse_marker(tmp_path):
    from .train_interaction_flow_tokens import build_model

    normalization = {
        "state_mean": np.zeros(20, dtype=np.float32),
        "state_std": np.ones(20, dtype=np.float32),
        "action_mean": np.zeros(14, dtype=np.float32),
        "action_std": np.ones(14, dtype=np.float32),
        "xyz_mean": np.zeros(3, dtype=np.float32),
        "xyz_std": np.ones(3, dtype=np.float32),
        "point_feature_mean": np.zeros(3, dtype=np.float32),
        "point_feature_std": np.ones(3, dtype=np.float32),
    }
    model = build_model(
        "interaction_flow",
        horizon=2,
        flow_steps=2,
        num_anchors=4,
        feature_dim=32,
        layers=1,
        flow_parameterization="free",
        xyz_std=normalization["xyz_std"],
        point_channels=6,
    )
    checkpoint = tmp_path / "marker_model.pt"
    torch.save(
        {
            "model": model.state_dict(),
            "condition": "interaction_flow",
            "horizon": 2,
            "flow_steps": 2,
            "num_anchors": 4,
            "feature_dim": 32,
            "layers": 1,
            "point_channels": 6,
            "target_color_priority": True,
            "normalization": normalization,
        },
        checkpoint,
    )
    runtime = InteractionFlowRuntime(
        checkpoint, device="cpu", observation_points=16
    )
    generator = np.random.default_rng(13)
    cloud = generator.normal(size=(64, 6)).astype(np.float32)
    cloud[:, 3:6] = 0.1
    cloud[:7, 3:6] = np.asarray((0.9, 0.8, 0.05), dtype=np.float32)
    sampled = runtime._points(cloud, target=True)
    marker = sampled[:, 3] + sampled[:, 4] - 2.0 * sampled[:, 5] > 0.65
    assert sampled.shape == (16, 6)
    assert int(marker.sum()) == 7


def test_runtime_axis_policy_supports_exact_zero_and_online_ndf_stub(tmp_path):
    from .train_interaction_flow_tokens import build_model

    normalization = {
        "state_mean": np.zeros(20, dtype=np.float32),
        "state_std": np.ones(20, dtype=np.float32),
        "action_mean": np.zeros(14, dtype=np.float32),
        "action_std": np.ones(14, dtype=np.float32),
        "xyz_mean": np.zeros(3, dtype=np.float32),
        "xyz_std": np.ones(3, dtype=np.float32),
        "point_feature_mean": np.zeros(3, dtype=np.float32),
        "point_feature_std": np.ones(3, dtype=np.float32),
    }
    model = build_model(
        "interaction_flow",
        horizon=2,
        flow_steps=2,
        num_anchors=4,
        feature_dim=32,
        layers=1,
        flow_parameterization="free",
        xyz_std=normalization["xyz_std"],
        point_channels=6,
        source_axis_relation_adapter=True,
    )
    checkpoint = tmp_path / "axis_model.pt"
    torch.save(
        {
            "model": model.state_dict(),
            "condition": "interaction_flow",
            "horizon": 2,
            "flow_steps": 2,
            "num_anchors": 4,
            "feature_dim": 32,
            "layers": 1,
            "point_channels": 6,
            "source_axis_relation_adapter": True,
            "normalization": normalization,
        },
        checkpoint,
    )
    runtime = InteractionFlowRuntime(
        checkpoint,
        device="cpu",
        observation_points=16,
        source_axis_mode="zero",
    )
    generator = np.random.default_rng(21)
    cloud_a = generator.normal(size=(24, 6)).astype(np.float32)
    cloud_b = generator.normal(size=(24, 6)).astype(np.float32)
    action = runtime.predict(
        operated_point_cloud=cloud_a,
        target_point_cloud=cloud_b,
        eef_state20=np.zeros(20, dtype=np.float32),
    )
    assert action.shape == (2, 14)
    assert runtime.last_diagnostic["ndf_axis_norm"] == 0.0

    class StubNdf:
        @staticmethod
        def encode(support, query):
            feature = np.zeros((len(query), 259), dtype=np.float32)
            feature[:, -3:] = np.asarray([0.0, 0.0, 1.0], dtype=np.float32)
            return feature

    runtime.source_axis_mode = "clean"
    runtime.ndf_adapter = StubNdf()
    runtime.predict(
        operated_point_cloud=cloud_a,
        target_point_cloud=cloud_b,
        eef_state20=np.zeros(20, dtype=np.float32),
    )
    assert runtime.last_diagnostic["ndf_axis_z"] == 1.0
    runtime.source_axis_mode = "orthogonal"
    runtime.predict(
        operated_point_cloud=cloud_a,
        target_point_cloud=cloud_b,
        eef_state20=np.zeros(20, dtype=np.float32),
    )
    assert abs(runtime.last_diagnostic["ndf_axis_z"]) < 1e-6
    assert abs(runtime.last_diagnostic["ndf_axis_norm"] - 1.0) < 1e-6


def test_runtime_reconstructs_target_axis_from_sampled_marker(tmp_path):
    from .target_frame import T_TEMPLATE
    from .train_interaction_flow_tokens import build_model

    normalization = {
        "state_mean": np.zeros(20, dtype=np.float32),
        "state_std": np.ones(20, dtype=np.float32),
        "action_mean": np.zeros(14, dtype=np.float32),
        "action_std": np.ones(14, dtype=np.float32),
        "xyz_mean": np.zeros(3, dtype=np.float32),
        "xyz_std": np.ones(3, dtype=np.float32),
        "point_feature_mean": np.zeros(3, dtype=np.float32),
        "point_feature_std": np.ones(3, dtype=np.float32),
    }
    model = build_model(
        "interaction_flow",
        horizon=2,
        flow_steps=2,
        num_anchors=4,
        feature_dim=32,
        layers=1,
        flow_parameterization="free",
        xyz_std=normalization["xyz_std"],
        point_channels=6,
        source_axis_relation_adapter=True,
        target_axis_relation_adapter=True,
    )
    checkpoint = tmp_path / "dual_axis_model.pt"
    torch.save(
        {
            "model": model.state_dict(),
            "condition": "interaction_flow",
            "horizon": 2,
            "flow_steps": 2,
            "num_anchors": 4,
            "feature_dim": 32,
            "layers": 1,
            "point_channels": 6,
            "source_axis_relation_adapter": True,
            "target_axis_relation_adapter": True,
            "target_axis_marker_local3": [-1.0, 0.0, 0.0],
            "normalization": normalization,
        },
        checkpoint,
    )
    grid = np.stack(
        np.meshgrid(np.linspace(-0.12, 0.12, 12), np.linspace(-0.12, 0.12, 12)),
        axis=-1,
    ).reshape(-1, 2)
    background = np.concatenate((grid, np.zeros((len(grid), 1))), axis=-1)
    marker = np.concatenate(
        (T_TEMPLATE, np.full((len(T_TEMPLATE), 1), 0.006)), axis=-1
    )
    target = np.zeros((len(background) + len(marker), 6), dtype=np.float32)
    target[:, :3] = np.concatenate((background, marker), axis=0)
    target[: len(background), 3:6] = 0.1
    target[len(background) :, 3:6] = np.asarray((0.9, 0.8, 0.05))
    source = target.copy()
    runtime = InteractionFlowRuntime(
        checkpoint,
        device="cpu",
        observation_points=len(target),
        source_axis_mode="zero",
        target_axis_mode="clean",
    )
    runtime.predict(
        operated_point_cloud=source,
        target_point_cloud=target,
        eef_state20=np.zeros(20, dtype=np.float32),
    )
    assert abs(runtime.last_diagnostic["target_axis_x"] + 1.0) < 0.05
    assert abs(runtime.last_diagnostic["target_axis_norm"] - 1.0) < 1e-6
    runtime.target_axis_mode = "zero"
    runtime.predict(
        operated_point_cloud=source,
        target_point_cloud=target,
        eef_state20=np.zeros(20, dtype=np.float32),
    )
    assert runtime.last_diagnostic["target_axis_norm"] == 0.0
    runtime.target_axis_mode = "orthogonal"
    runtime.predict(
        operated_point_cloud=source,
        target_point_cloud=target,
        eef_state20=np.zeros(20, dtype=np.float32),
    )
    assert abs(runtime.last_diagnostic["target_axis_x"]) < 0.05
    assert abs(runtime.last_diagnostic["target_axis_norm"] - 1.0) < 1e-6


def test_runtime_goal_policy_frame_canonicalizes_input_and_decodes_action(tmp_path):
    from .train_interaction_flow_tokens import build_model

    normalization = {
        "state_mean": np.zeros(20, dtype=np.float32),
        "state_std": np.ones(20, dtype=np.float32),
        "action_mean": np.zeros(14, dtype=np.float32),
        "action_std": np.ones(14, dtype=np.float32),
        "xyz_mean": np.zeros(3, dtype=np.float32),
        "xyz_std": np.ones(3, dtype=np.float32),
    }
    model = build_model(
        "interaction_functional_direct_flow",
        horizon=2,
        flow_steps=2,
        num_anchors=4,
        feature_dim=32,
        layers=1,
        flow_parameterization="free",
        xyz_std=normalization["xyz_std"],
    )
    checkpoint = tmp_path / "goal_policy_frame.pt"
    torch.save(
        {
            "model": model.state_dict(),
            "condition": "interaction_functional_direct_flow",
            "horizon": 2,
            "flow_steps": 2,
            "num_anchors": 4,
            "feature_dim": 32,
            "layers": 1,
            "flow_parameterization": "free",
            "functional_frame_encoding": (
                "camera_ndf_current_marker_goal_se3_columns_v1"
            ),
            "functional_frame_translation_mode": "delta",
            "action_frame": "world",
            "policy_frame": "goal",
            "normalization": normalization,
        },
        checkpoint,
    )

    world_from_goal = np.asarray(
        ((0.0, -1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
        dtype=np.float32,
    )
    goal_position = np.asarray((0.4, -0.2, 0.7), dtype=np.float32)
    goal_transform = np.eye(4, dtype=np.float32)
    goal_transform[:3, :3] = world_from_goal
    goal_transform[:3, 3] = goal_position

    class StubProvider:
        def estimate(self, *, mode, **_kwargs):
            assert mode == "clean"
            return OnlineFunctionalFrameEstimate(
                frame9_metric=np.asarray(
                    (0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0),
                    dtype=np.float32,
                ),
                combined_confidence=1.0,
                source_confidence=1.0,
                target_confidence=1.0,
                source_disagreement_deg=0.0,
                source_rotation=np.eye(3, dtype=np.float32),
                goal_transform=goal_transform,
                mode="clean",
                target_latched=True,
                marker_points=30,
                marker_fit_score_m2=1e-6,
            )

        @staticmethod
        def reset():
            return None

    runtime = InteractionFlowRuntime(
        checkpoint,
        device="cpu",
        observation_points=128,
        dense_frame_mode="clean",
        dense_frame_provider=StubProvider(),
    )

    class CaptureModel:
        def __init__(self):
            self.batch = None

        def __call__(self, batch):
            self.batch = batch
            action = torch.zeros((1, 2, 14), dtype=torch.float32)
            action[..., 0] = 0.05
            return InteractionFlowOutput(action=action)

    capture = CaptureModel()
    runtime.model = capture
    local_point = np.asarray((0.1, 0.02, -0.03), dtype=np.float32)
    world_point = local_point @ world_from_goal.T + goal_position
    cloud = np.repeat(world_point[None], 128, axis=0)
    world_sixd = world_from_goal[:, :2].reshape(6)
    state = np.concatenate(
        (world_point, world_sixd, [0.0], world_point, world_sixd, [1.0])
    ).astype(np.float32)
    action = runtime.predict(
        operated_point_cloud=cloud,
        target_point_cloud=cloud,
        eef_state20=state,
    )
    np.testing.assert_allclose(
        capture.batch["points_a"].numpy()[0, 0], local_point, atol=1e-6
    )
    np.testing.assert_allclose(
        capture.batch["state"].numpy()[0, :3], local_point, atol=1e-6
    )
    np.testing.assert_allclose(action[0, :3], (0.0, 0.05, 0.0), atol=1e-6)


def test_runtime_supplies_current_ndf_basis_to_equivariant_adapter(tmp_path):
    from .train_interaction_flow_tokens import build_model

    normalization = {
        "state_mean": np.zeros(20, dtype=np.float32),
        "state_std": np.ones(20, dtype=np.float32),
        "action_mean": np.zeros(14, dtype=np.float32),
        "action_std": np.linspace(0.5, 1.8, 14, dtype=np.float32),
        "xyz_mean": np.zeros(3, dtype=np.float32),
        "xyz_std": np.ones(3, dtype=np.float32),
    }
    condition = "interaction_functional_local_equivariant_adapter_flow"
    model = build_model(
        condition,
        horizon=2,
        flow_steps=2,
        num_anchors=4,
        feature_dim=32,
        layers=1,
        flow_parameterization="free",
        xyz_std=normalization["xyz_std"],
        action_std=normalization["action_std"],
    )
    checkpoint = tmp_path / "equivariant_adapter.pt"
    torch.save(
        {
            "model": model.state_dict(),
            "condition": condition,
            "horizon": 2,
            "flow_steps": 2,
            "num_anchors": 4,
            "feature_dim": 32,
            "layers": 1,
            "flow_parameterization": "free",
            "functional_frame_encoding": (
                "camera_ndf_current_marker_goal_se3_columns_v1"
            ),
            "functional_frame_translation_mode": "delta",
            "recovery_action_frame": "source_functional",
            "recovery_local_frame_token": True,
            "action_frame": "world",
            "policy_frame": "world",
            "normalization": normalization,
        },
        checkpoint,
    )
    source_rotation = np.asarray(
        ((0.0, -1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
        dtype=np.float32,
    )

    class StubProvider:
        def estimate(self, *, mode, **_kwargs):
            assert mode == "clean"
            transform = np.eye(4, dtype=np.float32)
            return OnlineFunctionalFrameEstimate(
                frame9_metric=np.asarray(
                    (0.04, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0),
                    dtype=np.float32,
                ),
                combined_confidence=1.0,
                source_confidence=1.0,
                target_confidence=1.0,
                source_disagreement_deg=0.0,
                source_rotation=source_rotation,
                goal_transform=transform,
                mode="clean",
                target_latched=True,
                marker_points=30,
                marker_fit_score_m2=1e-6,
            )

        @staticmethod
        def reset():
            return None

    runtime = InteractionFlowRuntime(
        checkpoint,
        device="cpu",
        observation_points=128,
        dense_frame_mode="clean",
        dense_frame_provider=StubProvider(),
    )

    class CaptureModel:
        def __init__(self):
            self.batch = None

        def __call__(self, batch):
            self.batch = batch
            return InteractionFlowOutput(action=torch.zeros((1, 2, 14)))

    capture = CaptureModel()
    runtime.model = capture
    cloud = np.zeros((128, 3), dtype=np.float32)
    runtime.predict(
        operated_point_cloud=cloud,
        target_point_cloud=cloud,
        eef_state20=np.zeros(20, dtype=np.float32),
    )
    expected = np.concatenate(
        (source_rotation[:, 0], source_rotation[:, 1])
    )
    np.testing.assert_allclose(
        capture.batch["source_frame6_columns"].cpu().numpy()[0], expected
    )
    expected_local = np.asarray(
        (0.0, -0.04, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0),
        dtype=np.float32,
    )
    np.testing.assert_allclose(
        capture.batch["target_frame9_source_local"].cpu().numpy()[0],
        expected_local,
        atol=1e-6,
    )
