import torch

from .deploy_fulltask_tagrt_v2 import gripper_command_from_probability
from .fulltask_tagrt_v2 import (
    FullTaskTAGRTPolicy,
    merge_action14,
    merge_action26,
    split_action14,
    split_action26,
)
from .train_fulltask_tagrt_v2 import (
    Statistics,
    augment_open_approach_eef_translation,
)


def _batch(batch_size=2):
    return {
        "scene": torch.randn(batch_size, 3, 24, 6),
        "points_a": torch.randn(batch_size, 3, 16, 6),
        "points_b": torch.randn(batch_size, 3, 16, 6),
        "local": torch.randn(batch_size, 3, 8, 12),
        "global": torch.randn(batch_size, 3, 10),
        "state": torch.randn(batch_size, 3, 20),
    }


def test_regression_and_flow_shapes():
    for decoder_type in ("regression", "flow"):
        model = FullTaskTAGRTPolicy(
            horizon=5,
            feature_dim=48,
            heads=4,
            memory_layers=2,
            decoder_layers=2,
            decoder_type=decoder_type,
        )
        output = model(_batch(), noisy_motion=torch.randn(2, 5, 12), time=torch.rand(2))
        assert output.motion.shape == (2, 5, 12)
        assert output.gripper_logits.shape == (2, 5, 2)
        sampled = model.sample(_batch(), integration_steps=2)
        assert sampled.motion.shape == (2, 5, 12)


def test_split_merge_action_roundtrip():
    action = torch.randn(3, 5, 14)
    motion, gripper = split_action14(action)
    assert torch.equal(merge_action14(motion, gripper), action)


def test_dense_joint_action_roundtrip_and_model_shape():
    action = torch.randn(3, 15, 26)
    motion, gripper = split_action26(action)
    assert torch.equal(merge_action26(motion, gripper), action)
    model = FullTaskTAGRTPolicy(
        horizon=15,
        feature_dim=48,
        heads=4,
        memory_layers=1,
        decoder_layers=1,
        decoder_type="regression",
        motion_dim=24,
    )
    output = model(_batch(), noisy_motion=torch.randn(2, 15, 24))
    assert output.motion.shape == (2, 15, 24)
    assert output.gripper_logits.shape == (2, 15, 2)


def test_dense_gripper_decoder_preserves_continuous_drive_target():
    probability = torch.tensor([[0.9, 0.4]], dtype=torch.float32)
    torch.testing.assert_close(
        gripper_command_from_probability(probability, continuous=True), probability
    )
    torch.testing.assert_close(
        gripper_command_from_probability(probability, continuous=False),
        torch.tensor([[1.0, 0.0]]),
    )


def test_open_approach_jitter_has_corrective_first_action():
    statistics = Statistics(
        point_mean=[0.0] * 6,
        point_std=[1.0] * 6,
        local_mean=[0.0] * 12,
        local_std=[1.0] * 12,
        global_mean=[0.0] * 10,
        global_std=[1.0] * 10,
        state_mean=[0.0] * 20,
        state_std=[1.0] * 20,
        motion_mean=[0.0] * 12,
        motion_std=[1.0] * 12,
    )
    batch = {
        "state": torch.zeros(2, 3, 20),
        "local": torch.zeros(2, 3, 8, 12),
        "motion": torch.zeros(2, 6, 12),
        "current_gripper": torch.tensor([[1.0, 1.0], [0.0, 1.0]]),
    }
    torch.manual_seed(7)
    augmented = augment_open_approach_eef_translation(batch, statistics, 0.01)
    assert torch.allclose(
        augmented["state"][0, -1, :3], -augmented["motion"][0, 0, :3]
    )
    assert torch.allclose(
        augmented["local"][0, -1, 0, 3:6],
        augmented["motion"][0, 0, :3],
    )
    assert torch.count_nonzero(augmented["state"][1]) == 0
    assert torch.count_nonzero(augmented["motion"][1]) == 0
