import builtins
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import Mock, patch
from types import SimpleNamespace

import numpy as np

from controller import ApproachController, ControlResult
from controller_config import ControllerConfig
from estimator import EKFConfig
from hardware_runtime import run_hardware, open_hardware
from interface import HardwareInterface
from runtime_config import RuntimeConfig
from hardware_fakes import Clock, Robot, Camera, Gripper, configured


class HardwareRuntimeTests(unittest.TestCase):
    def setUp(self):
        self.config, self.clock = configured(), Clock()
        self.robot, self.camera = Robot(self.clock), Camera(self.config.camera, self.clock)
        self.gripper = Gripper(self.clock)
        self.interface = HardwareInterface(self.config, self.robot, self.camera, self.gripper, clock=self.clock)

    def run_loop(self, **kwargs):
        return run_hardware(self.config, ControllerConfig(mode='pure_pbvs'), EKFConfig(enabled=False),
            clock=self.clock, sleep=self.clock.sleep, interface_factory=lambda *a, **k: self.interface, **kwargs)

    def test_observe_mode_reads_devices_without_actuator_commands(self):
        self.run_loop(task='observe', steps=2)
        self.assertEqual(self.robot.calls, [])
        self.assertEqual(self.gripper.calls, [])
        self.assertTrue(self.robot.stopped and self.robot.closed and self.camera.closed and self.gripper.closed)
        self.assertAlmostEqual(self.clock.now, 100.1)

    def test_servo_passes_configured_limits_and_stops_on_reached(self):
        calls = []

        def update(controller, frame):
            controller.interface.read_state()
            calls.append(controller.limits_config.velocity_limit)
            reached = len(calls) == 2
            velocity = np.zeros(6) if reached else np.ones(6) * .05
            controller.interface.set_joint_velocities(velocity)
            return ControlResult(None, velocity, velocity, 'reached' if reached else 'tracking', 0, 0, np.nan)

        with patch.object(ApproachController, 'update', update):
            result = self.run_loop(steps=20)
        self.assertEqual(result.status, 'reached')
        self.assertEqual(calls, [.15, .15])
        self.assertTrue(self.robot.stopped)

    def test_exception_and_cycle_overrun_stop_every_device(self):
        with patch.object(ApproachController, 'update', side_effect=RuntimeError('camera failed')):
            with self.assertRaisesRegex(RuntimeError, 'camera failed'):
                self.run_loop(steps=1)
        self.assertTrue(self.robot.stopped and self.camera.closed and self.gripper.closed)
        self.setUp()
        # Observe path deliberately makes processing exceed the configured budget.
        with patch('camera_processing.TaggedCubeProcessor.process', side_effect=lambda frame: self.clock.sleep(.21)):
            with self.assertRaisesRegex(TimeoutError, 'max_cycle_time'):
                self.run_loop(task='observe', steps=1)
        self.assertTrue(self.robot.stopped and self.robot.closed)

    def test_filtered_pbvs_retains_configured_reference_marker_frame(self):
        self.config.perception['reference_marker_id'] = 2
        kinematics = self.config.build_kinematics()
        processor = self.config.build_processor()
        controller = ApproachController(self.interface, kinematics,
            controller_config=ControllerConfig(mode='pure_pbvs', tool_pose_source='tag_only'),
            estimator_config=EKFConfig(enabled=True), processor=processor)
        frame = self.interface.capture_rgbd()
        camera_from_tool = kinematics.tool_pose()
        camera_from_reference = camera_from_tool @ controller.pbvs.desired
        camera_from_cube = camera_from_reference @ np.linalg.inv(processor.cube_from_markers[2])
        observation = SimpleNamespace(timestamp=frame.timestamp, camera_from_cube=camera_from_cube,
                                      camera_from_marker=camera_from_reference)
        tool = SimpleNamespace(camera_from_marker=camera_from_tool @ kinematics.tool_from_end_effector_marker)
        with patch.object(processor, 'process', return_value=observation), \
             patch.object(controller.tool_processor, 'process_marker', return_value=tool):
            result = controller.update(frame)
        self.assertEqual(result.status, 'reached')
        self.assertTrue(result.pose_estimate.measurement_accepted)
        np.testing.assert_allclose(result.command_velocity, 0, atol=1e-12)

    def test_preflight_rejects_placeholder_before_opening_any_device(self):
        self.config.camera['driver'] = 'unconfigured'
        factory = Mock(side_effect=AssertionError('must not connect'))
        with self.assertRaises(ValueError):
            run_hardware(self.config, ControllerConfig(mode='pure_pbvs'), EKFConfig(), interface_factory=factory)
        factory.assert_not_called()

    def test_partial_connection_failure_closes_already_opened_devices(self):
        with patch('hardware_runtime.custom_device', return_value=self.camera), \
             patch('hardware_runtime.OnRobotRG', return_value=self.gripper), \
             patch('hardware_runtime.URRTDE', side_effect=RuntimeError('robot offline')):
            with self.assertRaisesRegex(RuntimeError, 'offline'):
                open_hardware(self.config, clock=self.clock)
        self.assertTrue(self.camera.closed and self.gripper.closed)

    def test_shared_launcher_selects_hardware_without_importing_simulation(self):
        import run

        settings = RuntimeConfig.load().override(backend='hardware', headless=True)
        original_import = builtins.__import__

        def guarded(name, *args, **kwargs):
            if name.split('.')[0] in ('simulate', 'mujoco', 'robot_model'):
                raise AssertionError('hardware path imported simulation')
            return original_import(name, *args, **kwargs)

        with patch('hardware_config.HardwareConfig.load', return_value=self.config), \
             patch('hardware_runtime.run_hardware', return_value='hardware') as execute, \
             patch('builtins.__import__', guarded):
            self.assertEqual(run.launch(settings), 'hardware')
        execute.assert_called_once()

    def test_cli_validation_has_no_hardware_sdk_dependency(self):
        result = subprocess.run([sys.executable, 'run.py', '--backend', 'hardware', '--validate'],
            cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True, timeout=30)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('configuration is incomplete', result.stderr)
        self.assertNotIn('Traceback', result.stderr)


if __name__ == '__main__':
    unittest.main()
