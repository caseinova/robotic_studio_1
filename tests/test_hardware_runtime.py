import builtins
import json
import tempfile
from dataclasses import asdict
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
    def test_joint_state_is_acquired_after_both_tag_detectors(self):
        self.config.control['frame_max_age'] = .2
        model = self.config.build_kinematics()
        controller = ApproachController(self.interface, model, config=self.config.build_mpc_config(),
            controller_config=ControllerConfig(mode='pure_pbvs', frame_max_age_intervals=4),
            estimator_config=EKFConfig(enabled=False), processor=self.config.build_processor())
        frame = self.interface.capture_rgbd()
        order = []

        def detect_cube(frame):
            order.append('cube')
            self.clock.sleep(.06)
            return SimpleNamespace(camera_from_marker=np.eye(4))

        def detect_tool(frame):
            order.append('tool')
            self.clock.sleep(.06)
            return None

        read = self.robot.read_joint_state
        def read_joints():
            order.append('joints')
            return read()

        with patch.object(controller.processor, 'process', side_effect=detect_cube), \
             patch.object(controller.tool_processor, 'process_marker', side_effect=detect_tool), \
             patch.object(self.robot, 'read_joint_state', side_effect=read_joints):
            result = controller.update(frame)
        self.assertEqual(order, ['cube', 'tool', 'joints'])
        self.assertEqual(result.status, 'tracking')
        self.assertEqual(result.tool_pose_source, 'fk_fallback')
        self.assertAlmostEqual(self.interface._last_state.timestamp, self.clock())
        self.assertTrue(self.robot.calls)

    def setUp(self):
        self.config, self.clock = configured(), Clock()
        self.robot, self.camera = Robot(self.clock), Camera(self.config.camera, self.clock)
        self.gripper = Gripper(self.clock)
        self.interface = HardwareInterface(self.config, self.robot, self.camera, self.gripper, clock=self.clock)

    def run_loop(self, **kwargs):
        return run_hardware(self.config, ControllerConfig(mode='pure_pbvs'), EKFConfig(enabled=False),
            clock=self.clock, sleep=self.clock.sleep, interface_factory=lambda *a, **k: self.interface, **kwargs)

    def test_observe_mode_reads_devices_without_actuator_commands(self):
        self.robot.prepare_motion = Mock()
        self.run_loop(task='observe', steps=2)
        self.robot.prepare_motion.assert_not_called()
        self.assertEqual(self.robot.calls, [])
        self.assertEqual(self.gripper.calls, [])
        self.assertTrue(self.robot.stopped and self.robot.closed and self.camera.closed and self.gripper.closed)
        self.assertAlmostEqual(self.clock.now, 100.1)

    def test_slow_window_initialization_precedes_devices_and_cycle_budget(self):
        import cv2
        events = []

        def show(*args):
            events.append('show')
            if len(events) == 1:
                self.clock.sleep(.3)

        def factory(*args, **kwargs):
            self.assertEqual(events, ['show'])
            return self.interface

        with patch.object(cv2, 'namedWindow'), patch.object(cv2, 'imshow', side_effect=show), \
             patch.object(cv2, 'waitKey', return_value=-1), \
             patch.object(cv2, 'getWindowProperty', return_value=1), patch.object(cv2, 'destroyAllWindows'):
            run_hardware(self.config, ControllerConfig(), EKFConfig(enabled=False), task='observe',
                         steps=1, camera_view=True, clock=self.clock, sleep=self.clock.sleep,
                         interface_factory=factory)
        self.assertEqual(len(events), 2)
        self.assertEqual(self.robot.calls, [])
        self.assertTrue(self.robot.closed)

    def test_window_quit_during_initialization_opens_no_devices(self):
        import cv2
        factory = Mock()
        with patch.object(cv2, 'namedWindow'), patch.object(cv2, 'imshow'), \
             patch.object(cv2, 'waitKey', return_value=ord('q')), patch.object(cv2, 'destroyAllWindows') as close:
            run_hardware(self.config, ControllerConfig(mode='pure_pbvs'), EKFConfig(), steps=1,
                         camera_view=True, interface_factory=factory)
        factory.assert_not_called()
        close.assert_called_once()

    def test_pickup_refreshes_display_without_reprocessing_latched_target(self):
        import cv2
        pickup = Mock(phase='opening', finished=False)
        pickup.update.return_value = ControlResult(
            None, np.zeros(6), np.zeros(6), 'pickup_opening', 0, 0, np.nan,
            'gripper_width=80.0 mm')
        with patch.object(type(self.config), 'validate_ready'), \
             patch('hardware_runtime.CubePickupController', return_value=pickup), \
             patch.object(ApproachController, 'update') as approach, \
             patch.object(self.interface, 'capture_rgbd', wraps=self.interface.capture_rgbd) as capture, \
             patch.object(cv2, 'namedWindow'), patch.object(cv2, 'imshow') as show, \
             patch.object(cv2, 'waitKey', return_value=-1), \
             patch.object(cv2, 'getWindowProperty', return_value=1), \
             patch.object(cv2, 'destroyAllWindows'):
            self.run_loop(task='pick', steps=3, camera_view=True)
        self.assertEqual(pickup.update.call_count, 3)
        self.assertEqual(capture.call_count, 3)
        self.assertEqual(show.call_count, 4)  # Initial paint plus each pickup frame.
        approach.assert_not_called()
        self.assertFalse(self.interface.require_camera_for_motion)

    def test_headless_pickup_does_not_capture_display_frames(self):
        pickup = Mock(phase='opening', finished=False)
        pickup.update.return_value = ControlResult(
            None, np.zeros(6), np.zeros(6), 'pickup_opening', 0, 0, np.nan)
        with patch.object(type(self.config), 'validate_ready'), \
             patch('hardware_runtime.CubePickupController', return_value=pickup), \
             patch.object(self.interface, 'capture_rgbd') as capture:
            self.run_loop(task='pick', steps=3)
        capture.assert_not_called()

    def test_servo_passes_configured_limits_and_stops_on_reached(self):
        self.robot.prepare_motion = Mock(side_effect=lambda: self.clock.sleep(2.))
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
        self.robot.prepare_motion.assert_called_once()

    def test_exception_and_cycle_overrun_stop_every_device(self):
        with patch.object(ApproachController, 'update', side_effect=RuntimeError('camera failed')):
            with self.assertRaisesRegex(RuntimeError, 'camera failed'):
                self.run_loop(steps=1)
        self.assertTrue(self.robot.stopped and self.camera.closed and self.gripper.closed)
        self.setUp()
        # Observe path deliberately makes processing exceed the configured budget.
        with patch('camera_processing.TaggedCubeProcessor.process', side_effect=lambda frame: self.clock.sleep(.21)):
            with self.assertRaisesRegex(TimeoutError, 'max_cycle_time') as failure:
                self.run_loop(task='observe', steps=1)
        self.assertIn('observe=210.0 ms', str(failure.exception))
        self.assertIn('limit=200.0 ms', str(failure.exception))
        self.assertIn('task=observe, cycle=1', str(failure.exception))
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
        self.config.robot['host'] = None
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'hardware.json'
            path.write_text(json.dumps(asdict(self.config), default=lambda value: value.tolist()))
            result = subprocess.run([sys.executable, 'run.py', '--backend', 'hardware', '--validate',
                                     '--hardware-config', str(path)],
                cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True, timeout=30)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('configuration is incomplete', result.stderr)
        self.assertNotIn('Traceback', result.stderr)

    def test_disabled_gripper_does_not_connect_and_servo_can_read_state(self):
        self.config.gripper.update(driver='disabled', host=None)
        with patch('hardware_runtime.custom_device', return_value=self.camera) as custom, \
             patch('hardware_runtime.OnRobotRG') as gripper, \
             patch('hardware_runtime.URRTDE', return_value=self.robot):
            with open_hardware(self.config, clock=self.clock) as interface:
                state = interface.read_state()
                self.assertIsNone(state.gripper_width)
                self.config.build_kinematics().synchronize(state)
                interface.capture_rgbd()
                interface.set_joint_velocities(np.zeros(6))
                with self.assertRaisesRegex(RuntimeError, 'disabled'):
                    interface.close_gripper()
            gripper.assert_not_called()
            custom.assert_called_once()
        self.assertTrue(self.robot.closed and self.robot.stopped and self.camera.closed)

    def test_disabled_gripper_rejects_pick_before_opening_devices(self):
        self.config.gripper['driver'] = 'disabled'
        factory = Mock()
        with self.assertRaisesRegex(ValueError, 'pickup'):
            run_hardware(self.config, ControllerConfig(), EKFConfig(), task='pick', interface_factory=factory)
        factory.assert_not_called()

    def test_disabled_gripper_rejects_width_dependent_geometry(self):
        self.config.gripper['driver'] = 'disabled'
        self.config.collision['links'] = [dict(opening_axis=[0, .5, 0])]
        with self.assertRaisesRegex(ValueError, 'width-dependent'):
            self.config.validate_ready(ControllerConfig(), task='servo')


if __name__ == '__main__':
    unittest.main()
