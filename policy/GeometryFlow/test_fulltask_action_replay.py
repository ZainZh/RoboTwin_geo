from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

import h5py
import numpy as np

from .deploy_fulltask_action_replay import eval as replay_eval
from .deploy_fulltask_action_replay import get_model
from .prepare_fulltask_dense_control_archive import (
    dense_action26,
    policy_observation_indices,
)


def test_qpos_replay_uses_next_recorded_drive_target():
    with TemporaryDirectory() as directory:
        path = Path(directory) / "episode2.hdf5"
        qpos = np.arange(4 * 14, dtype=np.float32).reshape(4, 14)
        with h5py.File(path, "w") as archive:
            archive.create_dataset("joint_action/vector", data=qpos)

        model = get_model(
            {
                "expert_replay_hdf5": str(path),
                "expert_replay_episode": 2,
                "expert_replay_execute_steps": 1,
            }
        )
        assert model.replay_mode == "qpos_drive_target"
        np.testing.assert_array_equal(model.transitions, qpos[1:])

        class Task:
            def __init__(self):
                self.actions = []
                self.take_action_cnt = 0
                self.step_lim = 20
                self.eval_success = False

            def take_action(self, action, action_type):
                self.actions.append((np.asarray(action), action_type))
                self.take_action_cnt += 1

        observation = {
            "endpose": {
                "left_endpose": np.zeros(7, dtype=np.float32),
                "right_endpose": np.zeros(7, dtype=np.float32),
            }
        }
        task = Task()
        replay_eval(task, model, observation)
        assert len(task.actions) == 1
        np.testing.assert_array_equal(task.actions[0][0], qpos[1])
        assert task.actions[0][1] == "qpos"
        assert model.cursor == 1


def test_dense_replay_executes_one_low_level_chunk_without_replanning():
    with TemporaryDirectory() as directory:
        path = Path(directory) / "episode0.hdf5"
        position = np.arange(5 * 14, dtype=np.float32).reshape(5, 14)
        velocity = np.arange(5 * 12, dtype=np.float32).reshape(5, 12)
        with h5py.File(path, "w") as archive:
            archive.create_dataset("dense_control/position", data=position)
            archive.create_dataset("dense_control/arm_velocity", data=velocity)
        model = get_model(
            {
                "expert_replay_dense_control_hdf5": str(path),
                "expert_replay_episode": 0,
                "expert_replay_execute_steps": 3,
            }
        )

        class Task:
            def __init__(self):
                self.chunks = []
                self.take_action_cnt = 0
                self.step_lim = 20
                self.eval_success = False

            def take_low_level_joint_action_chunk(self, chunk, arm_velocity):
                self.chunks.append((np.asarray(chunk), np.asarray(arm_velocity)))
                self.take_action_cnt += 1

        observation = {
            "endpose": {
                "left_endpose": np.zeros(7, dtype=np.float32),
                "right_endpose": np.zeros(7, dtype=np.float32),
            }
        }
        task = Task()
        replay_eval(task, model, observation)
        assert model.replay_mode == "dense_joint_control"
        assert model.cursor == 3
        np.testing.assert_array_equal(task.chunks[0][0], position[:3])
        np.testing.assert_array_equal(task.chunks[0][1], velocity[:3])


def test_dense_dataset_alignment_skips_boundary_duplicate_observations():
    last_step = np.asarray([-1, 0, 15, 15, 22, 30, 45], dtype=np.int64)
    np.testing.assert_array_equal(
        policy_observation_indices(last_step, horizon=15),
        np.asarray([0, 2, 5, 6]),
    )
    position = np.arange(20 * 14, dtype=np.float32).reshape(20, 14)
    velocity = np.arange(20 * 12, dtype=np.float32).reshape(20, 12)
    action = dense_action26(position, velocity, start=2, horizon=15)
    restored_position = np.concatenate(
        (
            action[:, :6],
            action[:, 12:13],
            action[:, 13:19],
            action[:, 25:26],
        ),
        axis=-1,
    )
    restored_velocity = np.concatenate(
        (action[:, 6:12], action[:, 19:25]), axis=-1
    )
    np.testing.assert_array_equal(restored_position, position[2:17])
    np.testing.assert_array_equal(restored_velocity, velocity[2:17])
