"""Functional checks for the parallel gripper and its arm integration."""

import unittest

import mujoco
import numpy as np

from camera_processing import CameraProcessor
from controller import MujocoKinematics
from robot_model import RobotModel
from simulate import Simulation


class GripperTests(unittest.TestCase):
    def test_open_close_and_reset_preserve_six_joint_arm_api(self):
        simulation = Simulation()
        interface = simulation.interface
        interface.set_joint_velocities(np.zeros(6))
        arm_targets = simulation.robot.data.ctrl[simulation.robot.actuator_ids].copy()
        self.assertAlmostEqual(interface.gripper_width, 0.12)
        interface.close_gripper()
        simulation.step(500)
        self.assertLess(abs(interface.gripper_width), 0.001)
        interface.set_gripper_width(0.08)
        simulation.step(500)
        self.assertAlmostEqual(interface.gripper_width, 0.08, delta=0.001)
        interface.open_gripper()
        simulation.step(500)
        self.assertAlmostEqual(interface.gripper_width, 0.12, delta=0.001)
        np.testing.assert_array_equal(
            simulation.robot.data.ctrl[simulation.robot.actuator_ids], arm_targets
        )
        self.assertEqual(interface.joint_positions.shape, (6,))
        self.assertEqual(interface.joint_velocities.shape, (6,))
        simulation.reset()
        self.assertAlmostEqual(interface.gripper_width, 0.12)
        for invalid in (-0.01, 0.121, float("nan"), float("inf"), True, "0.1"):
            with self.assertRaises(ValueError):
                interface.set_gripper_width(invalid)

    def test_cube_contact_stops_fingers_before_full_closure(self):
        simulation = Simulation()
        robot = simulation.robot
        pose = robot.grasp_pose
        cube = robot.model.body("cube").id
        robot.model.body_pos[cube] = pose[:3, 3]
        mujoco.mju_mat2Quat(robot.model.body_quat[cube], pose[:3, :3].flatten())
        mujoco.mj_forward(robot.model, robot.data)
        simulation.interface.set_joint_velocities(np.zeros(6))
        simulation.interface.close_gripper()
        simulation.step(500)
        self.assertAlmostEqual(simulation.interface.gripper_width, 0.1, delta=0.003)
        finger_geoms = {robot.model.geom(name).id for name in (
            "gripper_left_geom", "gripper_right_geom"
        )}
        cube_geom = robot.model.geom("cube_geom").id
        contacting = {
            int(c.geom1 if c.geom2 == cube_geom else c.geom2)
            for c in robot.data.contact
            if cube_geom in (c.geom1, c.geom2)
        }
        self.assertTrue(finger_geoms.issubset(contacting))

    def test_camera_visibility_and_collision_planning_include_gripper(self):
        robot = RobotModel()
        self.assertIsNotNone(CameraProcessor().process(robot.capture_rgbd()))
        kinematics = MujocoKinematics(robot)
        floor = robot.model.geom("floor").id
        fingers = [robot.model.geom(name).id for name in (
            "gripper_left_geom", "gripper_right_geom"
        )]
        for finger in fingers:
            self.assertTrue(any(set(pair) == {finger, floor} for pair in kinematics.pairs))
        self.assertFalse(any(set(pair) == set(fingers) for pair in kinematics.pairs))
        self.assertGreater(kinematics.distances(robot.joint_positions).min(), 0.02)
        bare_robot = RobotModel(include_scene=False)
        self.assertAlmostEqual(bare_robot.gripper_width, 0.12)
        self.assertEqual(bare_robot.grasp_pose.shape, (4, 4))


if __name__ == "__main__":
    unittest.main()
