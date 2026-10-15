"""Fixed-camera calibration scene and visible rigid end-effector marker."""

import unittest

import cv2
import mujoco
import numpy as np

from interface import MujocoInterface
from noise_config import NoiseConfig
from robot_model import RobotModel
from camera_processing import CameraProcessor
from kinematics import MujocoKinematics


class ExternalCameraTests(unittest.TestCase):
    def test_only_fixed_camera_remains_in_scene_and_bare_robot(self):
        for include_scene in (True, False):
            robot = RobotModel(include_scene=include_scene)
            self.assertEqual(robot.camera_name, "external_camera")
            self.assertEqual(robot.model.ncam, 1)
            self.assertEqual(robot.model.cam_bodyid[robot.camera_id], 0)
            self.assertEqual(mujoco.mj_name2id(
                robot.model, mujoco.mjtObj.mjOBJ_CAMERA, "wrist_camera"), -1)
        with self.assertRaisesRegex(ValueError, "external_camera"):
            RobotModel(camera_name="wrist_camera")

    def test_fixed_camera_servo_commands_tool_motion(self):
        from simulate import Simulation

        simulation = Simulation()
        initial = simulation.robot.joint_positions
        result = simulation.run_controlled(25)
        self.assertEqual(result.status, "tracking", result.diagnostic)
        self.assertGreater(np.linalg.norm(result.command_velocity), 0.01)
        self.assertGreater(np.linalg.norm(simulation.robot.joint_positions - initial), 0.001)

    def test_fixed_rgbd_camera_sees_both_distinct_markers(self):
        robot = RobotModel(camera_name="external_camera")
        frame = robot.capture_rgbd()
        parameters = cv2.aruco.DetectorParameters()
        parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_APRILTAG
        detector = cv2.aruco.ArucoDetector(
            cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50), parameters)
        corners, ids, _ = detector.detectMarkers(cv2.cvtColor(frame.rgb, cv2.COLOR_RGB2GRAY))
        self.assertTrue({0, 1}.issubset(set(ids.reshape(-1))))
        self.assertGreater(np.count_nonzero(frame.valid_depth), 10000)
        self.assertEqual(robot.model.cam_bodyid[robot.camera_id], 0)
        np.testing.assert_allclose(frame.world_from_camera[:3, 3], [0.75, 1.20, 1.05])
        # The declared 80 mm marker size must agree with its rendered geometry.
        tag = corners[list(ids.reshape(-1)).index(1)].reshape(4, 2).astype(float)
        half = robot.end_effector_marker_size / 2
        points = np.array([[-half, half, 0], [half, half, 0],
                           [half, -half, 0], [-half, -half, 0]], dtype=float)
        success, _, translation = cv2.solvePnP(points, tag, frame.intrinsics, None)
        self.assertTrue(success)
        marker_world = robot.data.site("end_effector_marker_site").xpos
        marker_camera = frame.world_from_camera[:3, :3].T @ (marker_world - frame.world_from_camera[:3, 3])
        np.testing.assert_allclose(translation.reshape(3), marker_camera, atol=0.015)

    def test_tool_tag_pose_uses_mounting_calibration_without_cube_geometry(self):
        robot = RobotModel()
        frame = robot.capture_rgbd()
        # The tool plate has no cube behind it. A deliberately unrelated cube
        # offset must not affect the marker-only pose extraction.
        marker = CameraProcessor(marker_id=1, label_offset=1.0).process_marker(frame)
        self.assertIsNotNone(marker)
        kinematics = MujocoKinematics(robot)
        camera_from_tool = marker.camera_from_marker @ np.linalg.inv(
            kinematics.tool_from_end_effector_marker)
        world_from_tool = frame.world_from_camera @ camera_from_tool
        np.testing.assert_allclose(world_from_tool[:3, 3], kinematics.tool_pose()[:3, 3], atol=0.003)
        np.testing.assert_allclose(world_from_tool[:3, :3], kinematics.tool_pose()[:3, :3], atol=0.01)

    def test_camera_stays_fixed_and_marker_is_rigidly_mounted(self):
        robot = RobotModel(camera_name="external_camera")
        frame = robot.capture_rgbd()
        tool = robot.data.site("attachment_site")
        marker = robot.data.site("end_effector_marker_site")
        offset = tool.xmat.reshape(3, 3).T @ (marker.xpos - tool.xpos)
        marker_initial = marker.xpos.copy()
        robot.data.qpos[robot._qpos_indices[0]] += 0.15
        robot.data.qpos[robot._qpos_indices[1]] += 0.1
        second = robot.capture_rgbd()
        np.testing.assert_allclose(second.world_from_camera, frame.world_from_camera, atol=1e-12)
        self.assertGreater(np.linalg.norm(marker.xpos - marker_initial), 0.01)
        np.testing.assert_allclose(tool.xmat.reshape(3, 3).T @ (marker.xpos - tool.xpos), offset, atol=1e-12)
        for name in ("end_effector_aruco", "end_effector_tag_board", "end_effector_tag_bracket"):
            geom = robot.model.geom(name)
            self.assertEqual(geom.contype, 0)
            self.assertEqual(geom.conaffinity, 0)

    def test_interface_encoder_noise_does_not_move_fixed_camera_extrinsics(self):
        robot = RobotModel(camera_name="external_camera")
        expected = robot.capture_rgbd().world_from_camera
        with MujocoInterface(robot, noise_config=NoiseConfig(enabled=True, joint_position_std=0.03)) as interface:
            first = interface.capture_rgbd()
            robot.data.time += 0.05
            second = interface.capture_rgbd()
        np.testing.assert_allclose(first.world_from_camera, expected, atol=1e-12)
        np.testing.assert_allclose(second.world_from_camera, expected, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
