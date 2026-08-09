import numpy as np
from scipy.spatial.transform import Rotation

from .build_multigoal_relation_dataset import (
    frame9_from_matrix,
    matrix_from_frame9,
    pose9_from_matrix,
    predicted_current_frame,
    relative_frame9,
    rigid_grasp_labels,
    select_goal_rows,
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
