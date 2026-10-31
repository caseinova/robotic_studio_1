"""Offline RG2 linkage, width API, tag mounting, and contact regressions."""
import mujoco
import numpy as np
import pytest

from simulation.interface import MujocoInterface
from simulation.kinematics import MujocoKinematics
from simulation.robot_model import RobotModel
from simulation.rg2_model import MAX_WIDTH, angles_to_width, linkage_positions, width_to_angle


@pytest.mark.parametrize('width', [0, .02, .064, .08, MAX_WIDTH])
def test_linkage_width_and_independent_fk(width):
    robot = RobotModel(include_scene=False)
    ids = [robot.model.joint(n).qposadr[0] for n in robot.gripper_linkage_joint_names]
    robot.data.qpos[ids] = linkage_positions(width)
    mujoco.mj_forward(robot.model, robot.data)
    assert robot.gripper_width == pytest.approx(width, abs=1e-9)
    assert angles_to_width([width_to_angle(width)]*2) == pytest.approx(width, abs=1e-9)
    kin = MujocoKinematics(robot)
    kin.synchronize(MujocoInterface(robot).read_state())
    np.testing.assert_allclose(kin.grasp_pose(), robot.grasp_pose, atol=1e-9)


def test_open_close_with_no_obstacles():
    robot = RobotModel(include_scene=False)
    interface = MujocoInterface(robot)
    for width in (0, .064, MAX_WIDTH):
        interface.set_gripper_width(width)
        for _ in range(1000):
            mujoco.mj_step(robot.model, robot.data)
        assert robot.gripper_width == pytest.approx(width, abs=.001)
    with pytest.raises(ValueError):
        interface.set_gripper_width(MAX_WIDTH+.001)


def test_tag_on_body_face_and_realistic_size():
    robot = RobotModel(include_scene=False)
    kin = MujocoKinematics(robot)
    mount = kin.tool_from_end_effector_marker
    np.testing.assert_allclose(mount[:3, 3], [.01772389, 0, .085], atol=1e-8)
    np.testing.assert_allclose(mount[:3, 2], [1, 0, 0], atol=1e-8)
    np.testing.assert_allclose(mount[:3, 1], [0, 0, 1], atol=1e-8)
    assert robot.end_effector_marker_size == .04


def test_closing_depth_prediction_preserves_current_kinematics():
    robot = RobotModel(include_scene=False)
    kin = MujocoKinematics(robot)
    original = kin.data.qpos.copy()
    current = kin.grasp_pose()
    predicted = kin.grasp_pose_at_width(robot.cube_size)
    np.testing.assert_array_equal(kin.data.qpos, original)
    np.testing.assert_allclose(kin.grasp_pose(), current)
    assert np.linalg.norm(predicted[:3, 3] - current[:3, 3]) > .01
