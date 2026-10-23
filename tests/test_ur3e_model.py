"""UR3e geometry and task-scene integration regressions."""

import unittest

import numpy as np
import mujoco

from robot_model import RobotModel
from kinematics import MujocoKinematics
from hardware_config import HardwareConfig


class UR3eModelTests(unittest.TestCase):
    def test_hardware_dh_matches_supplied_arm_in_common_base_frame(self):
        config = HardwareConfig.load()
        config.robot.update(joint_lower=[-6.] * 6, joint_upper=[6.] * 6)
        # Supplied MJCF uses base_link; UR's controller base is rotated by pi.
        config.calibration.update(world_from_base=np.diag([-1., -1., 1., 1.]),
                                  flange_from_tool=np.eye(4), tool_from_grasp=np.eye(4))
        hardware = config.build_kinematics()
        robot = RobotModel()
        for q in (np.zeros(6), robot.joint_positions, np.array([.2, -1., 1.2, -.7, .5, .4])):
            robot.data.qpos[robot._qpos_indices] = q
            mujoco.mj_forward(robot.model, robot.data)
            pose = hardware.forward_kinematics(q)
            np.testing.assert_allclose(pose[:3, 3], robot.data.site("attachment_site").xpos, atol=1e-7)
            np.testing.assert_allclose(pose[:3, :3],
                                       robot.data.site("attachment_site").xmat.reshape(3, 3), atol=1e-7)

    def test_supplied_link_dimensions_and_force_limits(self):
        robot = RobotModel()
        self.assertEqual(robot.model_path.parent.name, "ur3e")
        for name, position in {
            "shoulder_link": [0, 0, 0.15185],
            "forearm_link": [-0.24355, 0, 0],
            "wrist_1_link": [-0.2132, 0, 0.13105],
            "wrist_3_link": [0, 0.0921, -1.889e-11],
        }.items():
            np.testing.assert_allclose(robot.model.body(name).pos, position)
        np.testing.assert_allclose(robot.model.actuator_forcerange[robot.actuator_ids, 1],
                                   [54, 54, 28, 9, 9, 9])
        self.assertFalse(any(c.dist < -1e-5 for c in robot.data.contact))

    def test_tool_frame_and_translational_jacobian(self):
        robot = RobotModel(include_scene=False)
        kinematics = MujocoKinematics(robot)
        q = robot.joint_positions
        jacobian = kinematics.tool_jacobian(q)
        numeric = np.empty((3, 6))
        for i in range(6):
            delta = np.eye(6)[i] * 1e-6
            kinematics.set_positions(q + delta)
            plus = kinematics.tool_pose()[:3, 3].copy()
            kinematics.set_positions(q - delta)
            numeric[:, i] = (plus - kinematics.tool_pose()[:3, 3]) / 2e-6
        np.testing.assert_allclose(jacobian[:3], numeric, atol=1e-7)
        # The supplied URDF's composed wrist->tool0 rotation is identity.
        np.testing.assert_allclose(robot.data.site("attachment_site").xpos,
                                   robot.data.body("wrist_3_link").xpos, atol=1e-10)


if __name__ == "__main__":
    unittest.main()
