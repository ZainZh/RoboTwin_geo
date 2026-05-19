import sys
import types
import unittest
from pathlib import Path

import numpy as np


XTRAINER_ROOT = Path(__file__).resolve().parents[1] / "include" / "xtrainer_clover"
if str(XTRAINER_ROOT) not in sys.path:
    sys.path.insert(0, str(XTRAINER_ROOT))

serial_module = types.ModuleType("serial")
serial_module.Serial = object
serial_tools_module = types.ModuleType("serial.tools")
serial_list_ports_module = types.ModuleType("serial.tools.list_ports")
serial_list_ports_module.comports = lambda: []
serial_tools_module.list_ports = serial_list_ports_module
serial_module.tools = serial_tools_module
sys.modules.setdefault("serial", serial_module)
sys.modules.setdefault("serial.tools", serial_tools_module)
sys.modules.setdefault("serial.tools.list_ports", serial_list_ports_module)


class XtrainerServoParamsTest(unittest.TestCase):
    def test_dobot_api_servoj_accepts_gain_parameter(self):
        from dobot_control.robots.dobot_api import DobotApiMove

        api = object.__new__(DobotApiMove)
        api.socket_dobot = 0
        sent = []
        api.sendRecvMsg = lambda string: sent.append(string) or "ok"

        api.ServoJ(1, 2, 3, 4, 5, 6, 0.06, gain=300)

        self.assertEqual(sent, ["ServoJ(1.000000,2.000000,3.000000,4.000000,5.000000,6.000000,0.060000,gain=300)"])

    def test_dobot_robot_uses_configured_servoj_t_and_gain(self):
        from dobot_control.robots.dobot import DobotRobot

        class FakeMove:
            def __init__(self):
                self.calls = []

            def ServoJ(self, *args, **kwargs):
                self.calls.append((args, kwargs))

        robot = object.__new__(DobotRobot)
        robot.robot_is_err = False
        robot.robot = FakeMove()
        robot._use_gripper = False
        robot._servo_j_t = 0.08
        robot._servo_j_gain = 250

        robot.command_joint_state(np.zeros(7, dtype=np.float32))

        args, kwargs = robot.robot.calls[0]
        self.assertAlmostEqual(args[6], 0.08)
        self.assertEqual(kwargs, {"gain": 250})

    def test_dobot_robot_set_servo_params_validates_and_stores_values(self):
        from dobot_control.robots.dobot import DobotRobot

        robot = object.__new__(DobotRobot)
        robot._servo_j_t = 0.03
        robot._servo_j_gain = 500

        result = robot.set_servo_params(servo_j_t=0.07, servo_j_gain=350)

        self.assertEqual(result, {"servo_j_t": 0.07, "servo_j_gain": 350})
        self.assertAlmostEqual(robot._servo_j_t, 0.07)
        self.assertEqual(robot._servo_j_gain, 350)

    def test_dobot_robot_rejects_invalid_servo_params(self):
        from dobot_control.robots.dobot import DobotRobot

        robot = object.__new__(DobotRobot)
        robot._servo_j_t = 0.03
        robot._servo_j_gain = 500

        with self.assertRaises(ValueError):
            robot.set_servo_params(servo_j_t=0.01, servo_j_gain=350)
        with self.assertRaises(ValueError):
            robot.set_servo_params(servo_j_t=0.07, servo_j_gain=1200)

    def test_zmq_robot_error_payload_raises_on_client_side(self):
        from dobot_control.robots.robot_node import _make_error_payload, _raise_if_remote_error

        payload = _make_error_payload("get_ik", ValueError("InverseSolution failed"))

        with self.assertRaisesRegex(RuntimeError, "Remote robot method get_ik failed.*InverseSolution failed"):
            _raise_if_remote_error(payload, "get_ik")


if __name__ == "__main__":
    unittest.main()
