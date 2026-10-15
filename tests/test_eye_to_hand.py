"""Eye-to-hand frame invariance and measured tool Jacobian checks."""

from dataclasses import replace
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
from scipy.spatial.transform import Rotation

from camera_processing import CameraProcessor
from controller import ApproachController
from controller_config import ControllerConfig
from estimator import EKFConfig
from kinematics import MujocoKinematics
from robot_model import RobotModel
from robot_state import RobotState


class EyeToHandTests(unittest.TestCase):
    def test_tag_only_closed_loop_keeps_both_tags_visible_and_reaches_goal(self):
        from simulate import Simulation
        from noise_config import NoiseConfig

        simulation = Simulation(controller_config=ControllerConfig(
            mode="pure_pbvs", tool_pose_source="tag_only"), noise_config=NoiseConfig(enabled=False))
        records = []
        original_update = ApproachController.update

        def record_update(controller, *args, **kwargs):
            result = original_update(controller, *args, **kwargs)
            records.append((result.status, result.tool_pose_source))
            return result

        with patch.object(ApproachController, "update", record_update):
            result = simulation.run_controlled(6000)
        self.assertEqual(result.status, "reached", result.diagnostic)
        self.assertTrue(records)
        self.assertTrue(all(status in ("tracking", "reached") and source == "tag"
                            for status, source in records))

    def test_calibrated_camera_frames_produce_same_tool_command(self):
        robot = RobotModel()
        frame = robot.capture_rgbd()
        observation = CameraProcessor().process(frame)
        tool_observation = CameraProcessor(marker_id=1).process_marker(frame)
        world_from_marker = frame.world_from_camera @ observation.camera_from_marker
        world_from_tool_marker = frame.world_from_camera @ tool_observation.camera_from_marker
        state = RobotState(frame.timestamp, robot.joint_positions, np.zeros(6), 0.12)
        robot.data = None  # Controllers must use interface inputs and model FK.
        commands = []
        for camera_pose in (frame.world_from_camera, np.eye(4)):
            measured_frame = replace(frame, world_from_camera=camera_pose)
            measured_observation = replace(observation,
                camera_from_marker=np.linalg.inv(camera_pose) @ world_from_marker)
            measured_tool_observation = replace(tool_observation,
                camera_from_marker=np.linalg.inv(camera_pose) @ world_from_tool_marker)
            interface = SimpleNamespace(timestamp=frame.timestamp,
                read_state=Mock(return_value=state), set_joint_velocities=Mock())
            controller = ApproachController(interface, MujocoKinematics(robot),
                controller_config=ControllerConfig(mode="pure_pbvs", tool_pose_source="tag_only"),
                estimator_config=EKFConfig(enabled=False))
            with patch.object(controller.processor, "process", return_value=measured_observation), \
                 patch.object(controller.tool_processor, "process_marker", return_value=measured_tool_observation):
                result = controller.update(measured_frame)
                self.assertEqual(result.tool_pose_source, "tag")
                commands.append(result.command_velocity)
            interface.read_state.assert_called_once()
            interface.set_joint_velocities.assert_called_once()
        self.assertGreater(np.linalg.norm(commands[0]), 0.01)
        np.testing.assert_allclose(commands[0], commands[1], atol=1e-12)

    def test_both_tags_drive_translation_and_rotation_together(self):
        robot = RobotModel()
        frame = robot.capture_rgbd()
        state = RobotState(frame.timestamp, robot.joint_positions, np.zeros(6), 0.12)
        interface = SimpleNamespace(timestamp=frame.timestamp,
            read_state=Mock(return_value=state), set_joint_velocities=Mock())
        kinematics = MujocoKinematics(robot)
        controller = ApproachController(interface, kinematics,
            controller_config=ControllerConfig(mode="pure_pbvs", tool_pose_source="tag_only"),
            estimator_config=EKFConfig(enabled=False))
        result = controller.update(frame)
        self.assertEqual(result.status, "tracking")
        self.assertEqual(result.tool_pose_source, "tag")
        self.assertIsNotNone(result.tool_observation)
        self.assertGreater(result.rotation_error, 1.0)
        twist = kinematics.tool_jacobian(state.joint_positions) @ result.command_velocity
        self.assertGreater(np.linalg.norm(twist[:3]), 0.03)
        self.assertGreater(np.linalg.norm(twist[3:]), 0.15)

    def test_tool_tag_loss_stops_strict_mode_and_reports_fk_fallback(self):
        robot = RobotModel()
        frame = robot.capture_rgbd()
        state = RobotState(frame.timestamp, robot.joint_positions, np.zeros(6), 0.12)
        for source in ("tag_only", "tag_with_fk_fallback"):
            interface = SimpleNamespace(timestamp=frame.timestamp,
                read_state=Mock(return_value=state), set_joint_velocities=Mock())
            controller = ApproachController(interface, MujocoKinematics(robot),
                controller_config=ControllerConfig(mode="pure_pbvs", tool_pose_source=source),
                estimator_config=EKFConfig(enabled=False))
            with patch.object(controller.tool_processor, "process_marker", return_value=None):
                result = controller.update(frame)
            if source == "tag_only":
                self.assertEqual(result.status, "tool_lost_or_rejected")
                np.testing.assert_array_equal(result.command_velocity, np.zeros(6))
            else:
                self.assertEqual(result.status, "tracking")
                self.assertEqual(result.tool_pose_source, "fk_fallback")
                self.assertGreater(np.linalg.norm(result.command_velocity), 0.01)

    def test_tool_jacobian_matches_pose_derivatives_with_stationary_camera(self):
        robot = RobotModel()
        kinematics = MujocoKinematics(robot)
        q = robot.joint_positions
        jacobian = kinematics.tool_jacobian(q)
        base_rotation = kinematics.tool_pose()[:3, :3]
        numeric = np.zeros((6, 6))
        epsilon = 1e-6
        for joint in range(6):
            step = np.eye(6)[joint] * epsilon
            kinematics.set_positions(q + step)
            plus = kinematics.tool_pose()
            kinematics.set_positions(q - step)
            minus = kinematics.tool_pose()
            numeric[:3, joint] = (plus[:3, 3] - minus[:3, 3]) / (2 * epsilon)
            numeric[3:, joint] = Rotation.from_matrix(
                plus[:3, :3] @ minus[:3, :3].T).as_rotvec() / (2 * epsilon)
        np.testing.assert_allclose(jacobian, numeric, atol=1e-8)
        local = kinematics.tool_jacobian(q, local=True)
        np.testing.assert_allclose(local[:3], base_rotation.T @ numeric[:3], atol=1e-8)
        np.testing.assert_allclose(local[3:], base_rotation.T @ numeric[3:], atol=1e-8)
        np.testing.assert_array_equal(kinematics.camera_jacobian(q), np.zeros((6, 6)))
