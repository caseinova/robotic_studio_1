"""Exercise the supplied API with fake SDK connections, never a physical arm."""

from contextlib import ExitStack
from dataclasses import replace
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
import cv2  # Keep native OpenCV initialized outside temporary sys.modules patches.

from common.controller_config import ControllerConfig
from hardware.hardware_config import HardwareConfig
from hardware.hardware_rosbridge import DEFAULT_API_PATH, RosbridgeArm, load_backend
from hardware_fakes import Clock, configured, Camera, Gripper


class RosbridgeTests(unittest.TestCase):
    def setUp(self):
        self.clock = Clock()
        self.config = configured()
        self.config.robot.update(driver="ur3_rg2_rosbridge", api_path=str(DEFAULT_API_PATH), frequency_hz=100)
        self.receiver = Mock()
        self.receiver.isConnected.return_value = True
        self.receiver.getSafetyMode.return_value = 1
        self.receiver.getTimestamp.side_effect = lambda: self.clock() + 1000
        self.receiver.getActualQ.return_value = [0.] * 6
        self.receiver.getActualQd.return_value = [0.] * 6
        self.sender = Mock()
        self.sender.setWatchdog.return_value = self.sender.kickWatchdog.return_value = True
        self.sender.speedJ.side_effect = self.speed
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(patch.dict(sys.modules, {
            "rtde_receive": SimpleNamespace(RTDEReceiveInterface=Mock()),
            "rtde_control": SimpleNamespace(RTDEControlInterface=Mock()),
        }))
        backend = load_backend(DEFAULT_API_PATH)
        module = sys.modules[backend.__module__]
        self.receive_factory = self.stack.enter_context(patch.object(module, "RTDEReceive", return_value=self.receiver))
        self.control_factory = self.stack.enter_context(patch.object(module, "RTDEControl", return_value=self.sender))

    def speed(self, target, acceleration, period):
        self.clock.sleep(period)
        return True

    def device(self):
        return RosbridgeArm(self.config.robot, self.config.control, self.clock)

    def finish(self, device):
        device._thread.join(timeout=2)
        self.assertFalse(device._thread.is_alive())
        with self.assertRaises(RuntimeError):
            device.close()

    def test_read_only_and_timestamped_public_getters(self):
        device = self.device()
        state = device.read_joint_state()
        self.assertAlmostEqual(self.clock() - state.timestamp, .01)
        self.receiver.getActualQ.assert_called_once()
        self.receiver.getActualQd.assert_called_once()
        self.control_factory.assert_not_called()
        self.receiver.getTimestamp.side_effect = None
        self.receiver.getTimestamp.return_value = 1100.
        self.clock.sleep(.2)
        self.assertGreater(self.clock() - device.read_joint_state().timestamp, .1)
        device.close()
        self.sender.speedJ.assert_not_called()
        self.receiver.disconnect.assert_called_once()

    def test_position_hold_uses_paced_velocity_api_and_expires(self):
        self.config.control["command_timeout"] = .025
        device = self.device()
        device.command_position(np.ones(6))
        self.finish(device)
        self.assertIn("expired", str(device._error))
        self.control_factory.assert_called_once_with("127.0.0.1", 100)
        self.sender.setWatchdog.assert_called_once_with(40.)
        self.assertGreater(self.sender.speedJ.call_count, 0)
        self.assertEqual(self.sender.initPeriod.call_count, self.sender.speedJ.call_count)
        self.assertEqual(self.sender.waitPeriod.call_count, self.sender.speedJ.call_count)
        for call in self.sender.speedJ.call_args_list:
            np.testing.assert_allclose(call.args[0], [.15] * 6)
            self.assertEqual(call.args[1:], (.5, .01))
        self.sender.speedStop.assert_called_once_with(.5)
        self.sender.stopScript.assert_called_once()
        self.sender.disconnect.assert_called_once()
        self.assertEqual(self.receiver.disconnect.call_count, 2)

    def test_stale_feedback_blocks_command_delivery(self):
        device = self.device()
        device.read_joint_state()
        self.receiver.getTimestamp.side_effect = None
        self.receiver.getTimestamp.return_value = 1100.
        self.clock.sleep(.2)
        device.command_velocity(np.ones(6) * .01)
        self.finish(device)
        self.assertIn("stale", str(device._error))
        self.sender.speedJ.assert_not_called()

    def test_control_constructor_failure_closes_both_receivers(self):
        device = self.device()
        self.control_factory.side_effect = RuntimeError("control offline")
        device.command_velocity(np.zeros(6))
        self.finish(device)
        self.assertIn("control offline", str(device._error))
        self.assertEqual(self.receiver.disconnect.call_count, 2)

    def test_rejected_command_stops_and_is_reported(self):
        self.sender.speedJ.side_effect = None
        self.sender.speedJ.return_value = False
        device = self.device()
        device.command_velocity(np.ones(6) * .01)
        self.finish(device)
        self.assertIn("speedJ command failed", str(device._error))
        self.sender.stopScript.assert_called_once()

    def test_watchdog_failure_prevents_motion(self):
        self.sender.setWatchdog.return_value = False
        device = self.device()
        device.command_velocity(np.ones(6) * .01)
        self.finish(device)
        self.assertIn("watchdog", str(device._error))
        self.sender.speedJ.assert_not_called()
        self.sender.disconnect.assert_called_once()

    def test_joint_limit_and_robot_safety_failures_prevent_motion(self):
        for failure in ("joint_limit", "safety_mode"):
            with self.subTest(failure=failure):
                self.receiver.getActualQ.return_value = [7.] * 6 if failure == "joint_limit" else [0.] * 6
                self.receiver.getSafetyMode.return_value = 3 if failure == "safety_mode" else 1
                device = self.device()
                device.command_velocity(np.ones(6) * .01)
                self.finish(device)
                self.sender.speedJ.assert_not_called()

    def test_runtime_selects_api_without_enabling_control(self):
        from hardware.hardware_runtime import open_hardware

        with patch("hardware.hardware_runtime.custom_device", return_value=Camera(self.config.camera, self.clock)), \
             patch("hardware.hardware_runtime.OnRobotRG", return_value=Gripper(self.clock)):
            with open_hardware(self.config, clock=self.clock) as interface:
                self.assertIsInstance(interface.robot, RosbridgeArm)
                interface.read_state()
        self.control_factory.assert_not_called()

    def test_config_resolution_and_missing_api(self):
        config = HardwareConfig.load()
        self.assertEqual(config.robot["driver"], "ur3_rg2_rosbridge")
        self.assertEqual(config.robot["api_path"], str(DEFAULT_API_PATH))
        self.config.validate_ready(ControllerConfig(mode="pure_pbvs"), task="observe")
        bad = replace(self.config, robot={**self.config.robot, "api_path": "/missing/robot/api"})
        with self.assertRaisesRegex(ValueError, "api_path"):
            bad.validate_schema()


if __name__ == "__main__":
    unittest.main()
