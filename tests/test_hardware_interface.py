import unittest
from unittest.mock import Mock

import numpy as np

from interface import HardwareInterface
from hardware_fakes import Clock, Robot, Camera, Gripper, configured


class HardwareInterfaceTests(unittest.TestCase):
    def setUp(self):
        self.config, self.clock = configured(), Clock()
        self.robot, self.gripper = Robot(self.clock), Gripper(self.clock)
        self.camera = Camera(self.config.camera, self.clock)
        self.interface = HardwareInterface(self.config, self.robot, self.camera, self.gripper, clock=self.clock)

    def test_registered_frame_depth_units_and_calibration(self):
        frame = self.interface.capture_rgbd()
        np.testing.assert_allclose(frame.depth, 1.)
        np.testing.assert_array_equal(frame.rgb, 120)
        np.testing.assert_array_equal(frame.intrinsics, self.config.camera['intrinsics'])
        self.assertTrue(frame.valid_depth.all())
        self.assertEqual(frame.timestamp, self.clock())
        self.clock.sleep(.01)
        second = self.interface.capture_rgbd()
        self.assertFalse(np.shares_memory(frame.depth, second.depth))

    def test_encoder_states_and_commands_go_through_adapters(self):
        state = self.interface.read_state()
        state.joint_positions[:] = 5  # Caller cannot change command-validation state.
        self.interface.capture_rgbd()
        self.interface.set_joint_velocities(np.ones(6) * .1)
        self.interface.set_joint_positions(np.zeros(6))
        self.interface.close_gripper()
        self.assertEqual([kind for kind, _ in self.robot.calls], ['velocity', 'position'])
        self.assertEqual(self.gripper.calls, [0])

    def test_stale_joint_gripper_and_camera_measurements_rejected(self):
        for device, method in [(self.robot, self.interface.read_state),
                               (self.gripper, self.interface.read_state),
                               (self.camera, self.interface.capture_rgbd)]:
            device.age = .5
            with self.subTest(device=device), self.assertRaises(RuntimeError):
                method()
            device.age = 0

    def test_duplicate_camera_frame_is_not_restamped_as_fresh(self):
        self.interface.capture_rgbd()
        with self.assertRaisesRegex(RuntimeError, 'new frame'):
            self.interface.capture_rgbd()

    def test_motion_checks_speed_position_and_measurement_age(self):
        with self.assertRaises(RuntimeError):
            self.interface.set_joint_velocities(np.ones(6) * .1)
        self.interface.read_state()
        self.interface.capture_rgbd()
        with self.assertRaises(ValueError):
            self.interface.set_joint_velocities(np.ones(6))
        with self.assertRaises(ValueError):
            self.interface.set_joint_positions(np.ones(6) * 7)
        self.clock.sleep(.2)
        with self.assertRaises(RuntimeError):
            self.interface.set_joint_velocities(np.ones(6) * .1)
        # Stopping is always allowed even with stale measurements.
        self.interface.set_joint_velocities(np.zeros(6))
        self.assertEqual(len(self.robot.calls), 1)

    def test_expired_frame_blocks_motion_after_slow_optimizer(self):
        self.interface.capture_rgbd()
        self.clock.sleep(.11)
        self.interface.read_state()
        with self.assertRaisesRegex(RuntimeError, 'command frame'):
            self.interface.set_joint_velocities(np.ones(6) * .1)
        self.interface.require_camera_for_motion = False  # Existing pickup uses latched target.
        self.interface.set_joint_velocities(np.ones(6) * .1)

    def test_cleanup_attempts_all_devices_even_when_stop_fails(self):
        self.robot.stop = Mock(side_effect=RuntimeError('lost connection'))
        with self.assertRaises(RuntimeError):
            self.interface.close()
        self.assertTrue(self.robot.closed and self.gripper.closed and self.camera.closed)
        self.interface.close()  # Idempotent.


if __name__ == '__main__':
    unittest.main()
