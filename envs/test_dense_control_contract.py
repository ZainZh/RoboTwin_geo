from __future__ import annotations

import numpy as np

from ._base_task import Base_Task


def test_low_level_joint_chunk_bypasses_planning_and_preserves_layout():
    class Robot:
        def __init__(self):
            self.arm = []
            self.gripper = []

        def set_arm_joints(self, position, velocity, arm):
            self.arm.append((np.asarray(position), np.asarray(velocity), arm))

        def set_gripper(self, value, arm, gripper_eps):
            self.gripper.append((float(value), arm, float(gripper_eps)))

    class Scene:
        def __init__(self):
            self.steps = 0

        def step(self):
            self.steps += 1

    task = Base_Task.__new__(Base_Task)
    task.robot = Robot()
    task.scene = Scene()
    task.take_action_cnt = 0
    task.step_lim = 10
    task.eval_success = False
    task.eval_video_path = None
    task.check_success = lambda: False
    task._update_render = lambda: None
    position = np.arange(2 * 14, dtype=np.float64).reshape(2, 14)
    velocity = np.arange(2 * 12, dtype=np.float64).reshape(2, 12)
    task.take_low_level_joint_action_chunk(position, velocity)
    assert task.take_action_cnt == 1
    assert task.scene.steps == 2
    np.testing.assert_array_equal(task.robot.arm[0][0], position[0, :6])
    np.testing.assert_array_equal(task.robot.arm[0][1], velocity[0, :6])
    np.testing.assert_array_equal(task.robot.arm[1][0], position[0, 7:13])
    np.testing.assert_array_equal(task.robot.arm[1][1], velocity[0, 6:12])
    assert task.robot.gripper[0] == (float(position[0, 6]), "left", 0.0)
    assert task.robot.gripper[1] == (float(position[0, 13]), "right", 0.0)


def test_low_level_settle_advances_physics_without_rewriting_commands():
    class Robot:
        def set_arm_joints(self, *_args, **_kwargs):
            raise AssertionError("settle must not issue an arm command")

        def set_gripper(self, *_args, **_kwargs):
            raise AssertionError("settle must not issue a gripper command")

    class Scene:
        def __init__(self):
            self.steps = 0

        def step(self):
            self.steps += 1

    task = Base_Task.__new__(Base_Task)
    task.robot = Robot()
    task.scene = Scene()
    task.take_action_cnt = 0
    task.step_lim = 10
    task.eval_success = False
    task.check_success = lambda: False
    task._update_render = lambda: None
    task.take_low_level_settle_steps(7)
    assert task.take_action_cnt == 1
    assert task.scene.steps == 7
