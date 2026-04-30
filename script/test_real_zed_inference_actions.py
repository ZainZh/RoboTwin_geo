import argparse
import time
import unittest

import numpy as np


class RealZedInferenceActionTest(unittest.TestCase):
    def test_action_delta_limit_clips_arm_joints_and_grippers_separately(self):
        from script.real_zed_inference.real_dp3_inference import limit_action_delta_for_execution

        last_action = np.zeros(14, dtype=np.float32)
        action = np.ones(14, dtype=np.float32)
        args = argparse.Namespace(max_executed_joint_delta=0.12, max_executed_gripper_delta=0.2)

        clipped = limit_action_delta_for_execution(action, last_action, args)

        arm_indices = [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12]
        np.testing.assert_allclose(clipped[arm_indices], np.full((12,), 0.12, dtype=np.float32))
        self.assertAlmostEqual(float(clipped[6]), 0.2)
        self.assertAlmostEqual(float(clipped[13]), 0.2)

    def test_action_delta_limit_can_be_disabled(self):
        from script.real_zed_inference.real_dp3_inference import limit_action_delta_for_execution

        last_action = np.zeros(14, dtype=np.float32)
        action = np.ones(14, dtype=np.float32)
        args = argparse.Namespace(max_executed_joint_delta=0.0, max_executed_gripper_delta=0.0)

        clipped = limit_action_delta_for_execution(action, last_action, args)

        np.testing.assert_allclose(clipped, action)

    def test_action_delta_change_limit_ramps_command_delta(self):
        from script.real_zed_inference.real_dp3_inference import prepare_action_for_execution

        last_action = np.zeros(14, dtype=np.float32)
        action = np.ones(14, dtype=np.float32)
        previous_delta = np.zeros(14, dtype=np.float32)
        args = argparse.Namespace(
            max_executed_joint_delta=0.12,
            max_executed_gripper_delta=0.2,
            max_executed_joint_delta_change=0.02,
            max_executed_gripper_delta_change=0.05,
        )

        first_command = prepare_action_for_execution(
            action,
            last_action,
            args,
            previous_command_delta=previous_delta,
        )
        second_command = prepare_action_for_execution(
            action,
            first_command,
            args,
            previous_command_delta=first_command - last_action,
        )

        arm_indices = [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12]
        np.testing.assert_allclose(first_command[arm_indices], np.full((12,), 0.02, dtype=np.float32), atol=1e-6)
        self.assertAlmostEqual(float(first_command[6]), 0.05, places=6)
        np.testing.assert_allclose(second_command[arm_indices], np.full((12,), 0.06, dtype=np.float32), atol=1e-6)
        self.assertAlmostEqual(float(second_command[6]), 0.15, places=6)

    def test_build_execution_substeps_interpolates_after_delta_limit(self):
        from script.real_zed_inference.real_dp3_inference import build_execution_substeps

        last_action = np.zeros(14, dtype=np.float32)
        target = np.ones(14, dtype=np.float32) * 0.12
        args = argparse.Namespace(execution_substeps=3)

        substeps = build_execution_substeps(last_action, target, args)

        self.assertEqual(len(substeps), 3)
        np.testing.assert_allclose(substeps[0], np.ones(14, dtype=np.float32) * 0.04, atol=1e-6)
        np.testing.assert_allclose(substeps[1], np.ones(14, dtype=np.float32) * 0.08, atol=1e-6)
        np.testing.assert_allclose(substeps[2], target, atol=1e-6)

    def test_execute_action_sends_interpolated_substeps_to_robot(self):
        from script.real_zed_inference.real_dp3_inference import execute_action

        class FakeEnv:
            def __init__(self):
                self.commands = []

            def step(self, action, flag):
                self.commands.append(np.asarray(action, dtype=np.float32).copy())
                return {"joint_positions": np.asarray(action, dtype=np.float32).copy()}

        env = FakeEnv()
        last_action = np.zeros(14, dtype=np.float32)
        action = np.ones(14, dtype=np.float32) * 0.12
        args = argparse.Namespace(
            execute=True,
            max_executed_joint_delta=0.12,
            max_executed_gripper_delta=0.12,
            max_executed_joint_delta_change=0.0,
            max_executed_gripper_delta_change=0.0,
            execution_substeps=3,
            execution_substep_sleep_sec=0.0,
            disable_action_delta_safety=False,
            max_action_delta=0.35,
            interpolate_first_action=False,
            disable_xyz_safety=True,
        )

        returned = execute_action(env=env, action=action, last_action=last_action, args=args, first_action=False)

        self.assertEqual(len(env.commands), 3)
        np.testing.assert_allclose(env.commands[0], np.ones(14, dtype=np.float32) * 0.04, atol=1e-6)
        np.testing.assert_allclose(env.commands[-1], action, atol=1e-6)
        np.testing.assert_allclose(returned, action, atol=1e-6)

    def test_async_action_controller_keeps_sending_last_target_when_buffer_empty(self):
        from script.real_zed_inference.real_dp3_inference import AsyncActionController

        class FakeEnv:
            def __init__(self):
                self.commands = []

            def step(self, action, flag):
                self.commands.append(np.asarray(action, dtype=np.float32).copy())
                return {"joint_positions": np.asarray(action, dtype=np.float32).copy()}

        env = FakeEnv()
        args = argparse.Namespace(
            execute=True,
            async_control_hz=50.0,
            async_control_max_idle_repeats=100,
            max_steps=20,
            max_executed_joint_delta=0.05,
            max_executed_gripper_delta=0.05,
            max_executed_joint_delta_change=0.01,
            max_executed_gripper_delta_change=0.01,
            execution_substeps=1,
            execution_substep_sleep_sec=0.0,
            disable_action_delta_safety=False,
            max_action_delta=0.35,
            interpolate_first_action=False,
            disable_xyz_safety=True,
            profile_timing=False,
            action_diagnostics_csv="",
        )
        controller = AsyncActionController(
            env=env,
            args=args,
            initial_action=np.zeros(14, dtype=np.float32),
        )
        controller.start()
        controller.submit_actions([np.ones(14, dtype=np.float32) * 0.2])
        time.sleep(0.09)
        controller.stop()

        self.assertGreaterEqual(len(env.commands), 3)
        unique_commands = {tuple(np.round(command, 5)) for command in env.commands}
        self.assertGreaterEqual(len(unique_commands), 2)
        np.testing.assert_allclose(env.commands[0], np.ones(14, dtype=np.float32) * 0.01, atol=1e-6)

    def test_async_action_controller_consumes_submitted_actions_in_order(self):
        from script.real_zed_inference.real_dp3_inference import AsyncActionController

        class FakeEnv:
            def __init__(self):
                self.commands = []

            def step(self, action, flag):
                self.commands.append(np.asarray(action, dtype=np.float32).copy())
                return {"joint_positions": np.asarray(action, dtype=np.float32).copy()}

        env = FakeEnv()
        args = argparse.Namespace(
            execute=True,
            async_control_hz=100.0,
            async_control_max_idle_repeats=0,
            max_steps=10,
            max_executed_joint_delta=0.5,
            max_executed_gripper_delta=0.5,
            max_executed_joint_delta_change=0.0,
            max_executed_gripper_delta_change=0.0,
            execution_substeps=1,
            execution_substep_sleep_sec=0.0,
            disable_action_delta_safety=True,
            max_action_delta=0.35,
            interpolate_first_action=False,
            disable_xyz_safety=True,
            profile_timing=False,
            action_diagnostics_csv="",
        )
        controller = AsyncActionController(
            env=env,
            args=args,
            initial_action=np.zeros(14, dtype=np.float32),
        )
        controller.start()
        controller.submit_actions([
            np.ones(14, dtype=np.float32) * 0.1,
            np.ones(14, dtype=np.float32) * 0.2,
        ])
        controller.wait_until_steps(2, timeout_sec=1.0)
        controller.stop()

        self.assertGreaterEqual(len(env.commands), 2)
        np.testing.assert_allclose(env.commands[0], np.ones(14, dtype=np.float32) * 0.1, atol=1e-6)
        np.testing.assert_allclose(env.commands[1], np.ones(14, dtype=np.float32) * 0.2, atol=1e-6)


class RealZedInferencePointcloudTest(unittest.TestCase):
    def test_camera_frame_to_output_pc_filters_workspace_before_output_transform(self):
        from script.real_zed_collection.workspace_crop_utils import WorkspaceBounds
        from script.real_zed_inference.real_dp3_inference import camera_frame_to_output_pc

        depth = np.ones((1, 2), dtype=np.float32)
        rgb = np.zeros((1, 2, 3), dtype=np.uint8)
        rgb[0, 0] = [255, 0, 0]
        rgb[0, 1] = [0, 255, 0]
        t_output_from_cam = np.eye(4, dtype=np.float32)
        t_output_from_cam[0, 3] = 10.0

        point_cloud = camera_frame_to_output_pc(
            camera_frame={"rgb": rgb, "depth_m": depth},
            camera_matrix=np.eye(3, dtype=np.float32),
            t_workspace_from_cam=np.eye(4, dtype=np.float32),
            workspace_bounds=WorkspaceBounds(
                x_min=-0.1,
                x_max=0.5,
                y_min=-0.1,
                y_max=0.5,
                z_min=0.5,
                z_max=1.5,
            ),
            t_output_from_cam=t_output_from_cam,
            min_depth_m=0.05,
            max_depth_m=3.0,
        )

        self.assertEqual(point_cloud.shape, (1, 6))
        np.testing.assert_allclose(point_cloud[0, :3], np.array([10.0, 0.0, 1.0], dtype=np.float32))
        np.testing.assert_allclose(point_cloud[0, 3:6], np.array([1.0, 0.0, 0.0], dtype=np.float32))

    def test_real_inference_parser_supports_parallel_camera_workers(self):
        from script.real_zed_inference.real_dp3_inference import build_arg_parser

        args = build_arg_parser().parse_args(["--parallel_camera_workers", "3"])

        self.assertEqual(args.parallel_camera_workers, 3)

    def test_real_inference_parser_defaults_to_smoothed_execution(self):
        from script.real_zed_inference.real_dp3_inference import build_arg_parser

        args = build_arg_parser().parse_args([])

        self.assertEqual(args.execution_substeps, 1)
        self.assertAlmostEqual(args.execution_substep_sleep_sec, 0.0)
        self.assertAlmostEqual(args.servo_j_t, 0.06)
        self.assertEqual(args.servo_j_gain, 300)
        self.assertAlmostEqual(args.max_executed_joint_delta_change, 0.0)
        self.assertAlmostEqual(args.max_executed_gripper_delta_change, 0.0)
        self.assertFalse(args.async_control)
        self.assertAlmostEqual(args.async_control_hz, 0.0)

    def test_configure_robot_servo_params_calls_robot_env(self):
        from script.real_zed_inference.real_dp3_inference import configure_robot_servo_params

        class FakeEnv:
            def __init__(self):
                self.calls = []

            def set_servo_params(self, *, servo_j_t, servo_j_gain):
                self.calls.append((servo_j_t, servo_j_gain))
                return {"servo_j_t": servo_j_t, "servo_j_gain": servo_j_gain}

        env = FakeEnv()
        args = argparse.Namespace(servo_j_t=0.08, servo_j_gain=250)

        configure_robot_servo_params(env, args)

        self.assertEqual(env.calls, [(0.08, 250)])

    def test_action_diagnostics_reports_policy_command_and_observed_deltas(self):
        from script.real_zed_inference.real_dp3_inference import build_action_diagnostic_row

        before = np.zeros(14, dtype=np.float32)
        raw_policy = np.zeros(14, dtype=np.float32)
        raw_policy[0] = 0.30
        raw_policy[6] = 0.50
        command = np.zeros(14, dtype=np.float32)
        command[0] = 0.12
        command[6] = 0.20
        observed = np.zeros(14, dtype=np.float32)
        observed[0] = 0.08
        observed[6] = 0.18

        row = build_action_diagnostic_row(
            step=7,
            raw_policy_action=raw_policy,
            command_action=command,
            observed_after_step=observed,
            action_before_step=before,
            previous_command_delta=np.zeros(14, dtype=np.float32),
            target_period_sec=0.2,
            step_elapsed_sec=0.15,
        )

        self.assertEqual(row["step"], 7)
        self.assertAlmostEqual(row["policy_arm_delta"], 0.30, places=6)
        self.assertAlmostEqual(row["command_arm_delta"], 0.12, places=6)
        self.assertAlmostEqual(row["observed_arm_delta"], 0.08, places=6)
        self.assertAlmostEqual(row["policy_gripper_delta"], 0.50, places=6)
        self.assertAlmostEqual(row["command_gripper_delta"], 0.20, places=6)
        self.assertAlmostEqual(row["observed_gripper_delta"], 0.18, places=6)
        self.assertAlmostEqual(row["command_arm_delta_change"], 0.12, places=6)
        self.assertAlmostEqual(row["command_gripper_delta_change"], 0.20, places=6)
        self.assertAlmostEqual(row["command_follow_error_arm"], 0.04, places=6)
        self.assertAlmostEqual(row["command_follow_error_gripper"], 0.02, places=6)
        self.assertAlmostEqual(row["target_period_sec"], 0.2, places=6)
        self.assertAlmostEqual(row["step_elapsed_sec"], 0.15, places=6)
        self.assertFalse(row["control_overrun"])


if __name__ == "__main__":
    unittest.main()
