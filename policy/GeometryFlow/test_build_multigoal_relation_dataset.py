import numpy as np
from scipy.spatial.transform import Rotation

from .build_multigoal_relation_dataset import (
    build_multigoal_payload,
    frame9_from_matrix,
    matrix_from_frame9,
    pose9_from_matrix,
    predicted_current_frame,
    relative_frame9,
    rigid_grasp_labels,
    select_goal_rows,
    select_source_rows,
)
from .canonicalize_task_flow_dataset import pose7_to_matrix
from .functional_action_frame import frame9_rotation


def _pose7(position=(0.0, 0.0, 0.0)) -> np.ndarray:
    return np.asarray((*position, 1.0, 0.0, 0.0, 0.0), dtype=np.float32)


def test_relative_contract_recovers_predicted_current() -> None:
    current = np.eye(4)
    current[:3, :3] = Rotation.from_euler("xyz", [10, -5, 25], degrees=True).as_matrix()
    current[:3, 3] = [0.1, -0.2, 0.3]
    goal = np.eye(4)
    goal[:3, :3] = Rotation.from_euler("xyz", [-4, 8, -15], degrees=True).as_matrix()
    goal[:3, 3] = [-0.05, 0.07, 0.4]
    relation = relative_frame9(goal, current)
    recovered = predicted_current_frame(frame9_from_matrix(goal), relation)
    np.testing.assert_allclose(recovered, current, atol=1e-6)


def test_rigid_grasp_labels_reach_goal_and_keep_passive_arm_fixed() -> None:
    current = np.eye(4)
    current[:3, 3] = [0.1, 0.0, 0.7]
    goal = np.eye(4)
    goal[:3, :3] = Rotation.from_euler("z", 30, degrees=True).as_matrix()
    goal[:3, 3] = [0.13, -0.02, 0.72]
    eef = np.stack((_pose7((-0.2, 0.0, 0.8)), _pose7((0.2, 0.0, 0.8))))
    anchors = np.asarray(
        [[0.09, 0.0, 0.7], [0.11, 0.0, 0.7], [0.1, 0.02, 0.7]],
        dtype=np.float32,
    )
    actions, future, flow, poses = rigid_grasp_labels(
        current_object=current,
        goal_object=goal,
        current_eef_pose7=eef,
        current_anchors=anchors,
        active_arm_right=True,
        grippers=(1.0, 0.0),
        action_horizon=6,
        flow_steps=16,
    )
    assert actions.shape == (6, 14)
    assert future.shape == (6, 2, 7)
    assert flow.shape == (3, 16, 3)
    assert poses.shape == (16, 9)
    np.testing.assert_allclose(future[-1, 0], eef[0], atol=1e-6)
    expected_delta = goal @ np.linalg.inv(current)
    expected_active = expected_delta @ pose7_to_matrix(eef[1])
    np.testing.assert_allclose(
        pose7_to_matrix(future[-1, 1]), expected_active, atol=1e-6
    )
    np.testing.assert_allclose(flow[:, 0], anchors, atol=1e-6)
    expected_anchors = anchors @ expected_delta[:3, :3].T + expected_delta[:3, 3]
    np.testing.assert_allclose(flow[:, -1], expected_anchors, atol=1e-6)
    np.testing.assert_allclose(
        matrix_from_frame9(frame9_from_matrix(goal)), goal, atol=1e-6
    )
    np.testing.assert_allclose(frame9_rotation(frame9_from_matrix(goal)), goal[:3, :3])


def test_goal_selection_keeps_original_and_prefers_distant_real_goal() -> None:
    identity = np.eye(4, dtype=np.float32)
    current = pose9_from_matrix(identity)
    goals = []
    for x in (0.0, 0.02, 0.08):
        transform = identity.copy()
        transform[0, 3] = x
        goals.append(frame9_from_matrix(transform))
    payload = {
        "episode_index": np.asarray((10, 11, 12), dtype=np.int64),
        "current_object_pose9": np.stack((current, current, current)),
        "goal_frame9": np.stack(goals),
    }
    selected = select_goal_rows(
        payload,
        source_index=0,
        candidate_rows=np.arange(3, dtype=np.int64),
        count=2,
    )
    assert selected == [0, 2]
    limited = select_goal_rows(
        payload,
        source_index=0,
        candidate_rows=np.arange(3, dtype=np.int64),
        count=2,
        maximum_translation_cm=5.0,
        maximum_rotation_deg=90.0,
    )
    assert limited == [0, 1]


def test_source_selection_balances_relation_rows_per_episode() -> None:
    payload = {
        "shoe_id": np.zeros(9, dtype=np.int64),
        "episode_index": np.asarray((0, 0, 0, 0, 0, 1, 1, 1, 1)),
        "relation_phase": np.asarray((0, 1, 1, 1, 1, 0, 1, 1, 1)),
    }
    selected = select_source_rows(
        payload,
        relation_only=True,
        max_sources_per_episode=2,
    )
    np.testing.assert_array_equal(selected, np.asarray((1, 4, 6, 8)))


def test_source_selection_zero_limit_keeps_every_eligible_row() -> None:
    payload = {
        "shoe_id": np.zeros(4, dtype=np.int64),
        "episode_index": np.asarray((0, 0, 1, 1)),
        "relation_phase": np.asarray((0, 1, 1, 0)),
    }
    selected = select_source_rows(
        payload,
        relation_only=True,
        max_sources_per_episode=0,
    )
    np.testing.assert_array_equal(selected, np.asarray((1, 2)))


def test_source_selection_applies_short_horizon_label_limits() -> None:
    payload = {
        "shoe_id": np.zeros(3, dtype=np.int64),
        "episode_index": np.arange(3, dtype=np.int64),
        "relation_phase": np.ones(3, dtype=np.float32),
        "goal_translation_error_xyz_m": np.asarray(
            ((0.02, 0.0, 0.0), (0.25, 0.0, 0.0), (0.01, 0.0, 0.0)),
            dtype=np.float32,
        ),
        "goal_rotation_error_rotvec": np.asarray(
            ((0.0, 0.0, 0.2), (0.0, 0.0, 0.1), (0.0, 0.0, 2.0)),
            dtype=np.float32,
        ),
    }
    selected = select_source_rows(
        payload,
        relation_only=True,
        max_sources_per_episode=0,
        maximum_translation_cm=20.0,
        maximum_rotation_deg=90.0,
    )
    np.testing.assert_array_equal(selected, np.asarray((0,)))


def test_builder_accepts_shoe_contract_without_optional_stage_labels() -> None:
    count = 3
    identity_frame = frame9_from_matrix(np.eye(4))
    identity_pose = pose9_from_matrix(np.eye(4))
    eef = np.stack((_pose7((-0.2, 0.0, 0.8)), _pose7((0.2, 0.0, 0.8))))
    payload = {
        "points_a": np.zeros((count, 4, 6), dtype=np.float32),
        "points_b": np.zeros((count, 4, 6), dtype=np.float32),
        "state": np.zeros((count, 20), dtype=np.float32),
        "action": np.zeros((count, 6, 14), dtype=np.float32),
        "current_anchors": np.zeros((count, 3, 3), dtype=np.float32),
        "current_object_pose9": np.stack([identity_pose] * count),
        "current_eef_pose7": np.stack([eef] * count),
        "future_eef_pose7": np.zeros((count, 6, 2, 7), dtype=np.float32),
        "active_arm_right": np.ones(count, dtype=np.float32),
        "episode_index": np.arange(count, dtype=np.int64),
        "shoe_id": np.arange(count, dtype=np.int64),
        "frame_index": np.zeros(count, dtype=np.int64),
        "relation_phase": np.ones(count, dtype=np.float32),
        "goal_frame9": np.stack([identity_frame] * count),
        "target_frame9": np.stack([identity_frame] * count),
        "source_frame6_columns": np.stack([identity_frame[3:]] * count),
        "goal_translation_error_xyz_m": np.zeros((count, 3), dtype=np.float32),
        "goal_rotation_error_rotvec": np.zeros((count, 3), dtype=np.float32),
    }
    output, metadata = build_multigoal_payload(
        payload,
        train_ids=[0],
        validation_ids=[1],
        test_ids=[2],
        goals_per_source=1,
        flow_steps=4,
    )
    assert len(output["shoe_id"]) == count
    assert metadata["source_samples"] == count
    assert "goal_aligned" not in output
