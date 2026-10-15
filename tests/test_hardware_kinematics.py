"""UR3 model-free math, collision constraints, and portable controller use."""

from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
from scipy.spatial.transform import Rotation

from controller import ApproachController, CubePickupController, JointSpaceMPC, MPCConfig
from controller_config import ControllerConfig
from estimator import EKFConfig
from hardware_kinematics import (
    UR3DH, UR3Kinematics, LinkCapsule, SphereObstacle, BoxObstacle, PlaneObstacle,
    _segment_distance, _segment_box_distance,
)
from robot_state import RobotState


def pose(xyz=(0, 0, 0), rotvec=(0, 0, 0)):
    result = np.eye(4)
    result[:3, :3] = Rotation.from_rotvec(rotvec).as_matrix()
    result[:3, 3] = xyz
    return result


def backend(**kwargs):
    settings = dict(joint_limits=(-np.ones(6) * 2 * np.pi, np.ones(6) * 2 * np.pi),
                    world_from_base=np.eye(4), flange_from_tool=np.eye(4),
                    tool_from_grasp=pose((0, 0, .105)), tool_from_marker=pose((0, -.12, -.1)))
    settings.update(kwargs)
    return UR3Kinematics(**settings)


class HardwareKinematicsTests(unittest.TestCase):
    def test_known_nominal_ur3_zero_pose(self):
        k = backend()
        expected = pose((-.4569, -.19425, .06655), (np.pi / 2, 0, 0))
        np.testing.assert_allclose(k.tool_pose(), expected, atol=1e-12)
        np.testing.assert_allclose(k.forward_kinematics(np.zeros(6)), expected, atol=1e-12)

    def test_joint_offsets_base_tool_and_grasp_calibration(self):
        offsets = (.1, -.2, .3, -.1, .2, -.3)
        q = np.array([.3, -1.1, .8, -.7, .9, .2])
        base = pose((.3, -.2, .1), (.1, -.2, .4))
        tool = pose((.02, .01, .08), (.2, .1, -.1))
        grasp = pose((.01, 0, .12))
        k = backend(dh=UR3DH(offsets=offsets), world_from_base=base,
                    flange_from_tool=tool, tool_from_grasp=grasp)
        reference = backend().forward_kinematics(q + offsets)
        k.synchronize(RobotState(1., q, np.zeros(6), .10))
        np.testing.assert_allclose(k.tool_pose(), base @ reference @ tool, atol=1e-12)
        np.testing.assert_allclose(k.grasp_pose(), base @ reference @ tool @ grasp, atol=1e-12)
        before = k.tool_pose()
        k.forward_kinematics(q + .1)
        np.testing.assert_array_equal(k.tool_pose(), before)

    def test_world_and_local_jacobians_match_pose_derivatives(self):
        k = backend(world_from_base=pose((.1, -.1, .2), (.2, -.3, .4)),
                    flange_from_tool=pose((.03, -.02, .14), (.2, .4, -.3)))
        rng = np.random.default_rng(19)
        for q in rng.uniform(-2, 2, (5, 6)):
            expected = np.zeros((6, 6))
            for joint in range(6):
                delta = np.eye(6)[joint] * 1e-6
                plus, minus = k.forward_kinematics(q + delta), k.forward_kinematics(q - delta)
                expected[:3, joint] = (plus[:3, 3] - minus[:3, 3]) / 2e-6
                expected[3:, joint] = Rotation.from_matrix(
                    plus[:3, :3] @ minus[:3, :3].T).as_rotvec() / 2e-6
            np.testing.assert_allclose(k.tool_jacobian(q), expected, atol=2e-9)
            r = k.forward_kinematics(q)[:3, :3].T
            np.testing.assert_allclose(k.tool_jacobian(q, local=True),
                                       np.vstack((r @ expected[:3], r @ expected[3:])), atol=2e-9)

    def test_segment_geometry_including_parallel_degenerate_and_thin_box(self):
        z = np.zeros(3)
        self.assertAlmostEqual(_segment_distance(z, z, np.ones(3), np.ones(3)), np.sqrt(3))
        self.assertAlmostEqual(_segment_distance(z, [1, 0, 0], np.array([0, 1, 0]),
                                                np.array([1, 1, 0])), 1)
        self.assertAlmostEqual(_segment_distance(z, [1, 0, 0], np.array([.5, -1, 0]),
                                                np.array([.5, 1, 0])), 0)
        half = np.array([.001, .1, .1])
        self.assertEqual(_segment_box_distance(np.array([-10, 0, 0]), np.array([10, 0, 0]), half), 0)
        self.assertAlmostEqual(_segment_box_distance(np.array([-10, .3, 0]),
                                                    np.array([10, .3, 0]), half), .2)

    def test_distances_sphere_oriented_box_plane_and_self_collision(self):
        # Zero-length capsules are spheres; simple exact distances are known.
        links = [LinkCapsule('one', 0, (0, 0, 0), (0, 0, 0), .02),
                 LinkCapsule('two', 'tool', (0, 0, 0), (0, 0, 0), .03)]
        box = pose((.3, 0, 0), (0, 0, np.pi / 2))
        k = backend(link_capsules=links, obstacles=[SphereObstacle('sphere', (.2, 0, 0), .03),
                    BoxObstacle('box', box, (.05, .1, .2)), PlaneObstacle('floor', (0, 0, 1), -.1)])
        distance = k.distances(np.zeros(6))
        np.testing.assert_allclose(distance[:3], [.15, .18, .08], atol=1e-12)
        expected_self = np.linalg.norm(k.tool_pose()[:3, 3]) - .05
        self.assertAlmostEqual(distance[-1], expected_self)
        k.set_obstacles([SphereObstacle('overlap', (0, 0, 0), .03)])
        self.assertAlmostEqual(k.distances(np.zeros(6))[0], -.05)

    def test_collision_gradient_matches_analytic_sphere_clearance(self):
        k = backend(link_capsules=[LinkCapsule('arm', 1, (.2, 0, 0), (.2, 0, 0), .02)],
                    obstacles=[SphereObstacle('block', (.2, .07, .1519), .01)])
        q = np.zeros(6)
        distances, jacobian = k.distances(q, gradients=True)
        np.testing.assert_allclose(distances, [.04], atol=1e-12)
        np.testing.assert_allclose(jacobian, [[-.2, 0, 0, 0, 0, 0]], atol=1e-8)
        np.testing.assert_array_equal(k.positions, q)

    def test_gripper_width_and_selective_target_contact(self):
        finger = LinkCapsule('finger', 'tool', (0, 0, 0), (0, 0, .1), .01,
                             opening_axis=(0, .5, 0), contact_group='finger')
        palm = LinkCapsule('palm', 'tool', (0, 0, 0), (0, 0, 0), .02)
        k = backend(link_capsules=[finger, palm],
                    obstacles=[SphereObstacle('cube', (1, 1, 1), .05, contact_group='grasp_target'),
                               PlaneObstacle('floor', (0, 0, 1), -2)])
        k.synchronize(RobotState(1, np.zeros(6), np.zeros(6), .12))
        opened = k._capsules(k._frames)[0][0]
        k.synchronize(RobotState(1, np.zeros(6), np.zeros(6), 0))
        closed = k._capsules(k._frames)[0][0]
        np.testing.assert_allclose(opened - closed, k.tool_pose()[:3, :3] @ [0, .06, 0], atol=1e-12)
        self.assertEqual(k.pickup_collision_indices(), [1, 2, 3])
        self.assertEqual(len(k.distances(np.zeros(6))), 4)  # Normal MPC retains finger/cube pair.

    def test_invalid_parameters_and_missing_calibration_are_explicit(self):
        for kwargs in (dict(joint_limits=(np.ones(6), -np.ones(6))),
                       dict(flange_from_tool=np.zeros((4, 4))), dict(gradient_step=0),
                       dict(marker_id=50), dict(marker_size=0)):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                backend(**kwargs)
        with self.assertRaises(ValueError):
            UR3DH(a=(np.nan,) * 6)
        with self.assertRaises(ValueError):
            LinkCapsule('bad', 8, (0, 0, 0), (0, 0, 0), .1)
        k = backend(tool_from_marker=None)
        with self.assertRaisesRegex(ValueError, 'calibrated'):
            _ = k.tool_from_end_effector_marker
        with self.assertRaisesRegex(ValueError, 'link_capsules'):
            k.distances(np.zeros(6))
        with self.assertRaises(ValueError):
            k.synchronize(RobotState(1, np.zeros(6), np.zeros(6), -.1))
        with self.assertRaises(ValueError):
            k.forward_kinematics(np.full(6, np.inf))

    def test_mpc_limits_motion_toward_known_obstacle(self):
        k = backend(link_capsules=[LinkCapsule('arm', 1, (.2, 0, 0), (.2, 0, 0), .02)],
                    obstacles=[SphereObstacle('block', (.2, .07, .1519), .01)])
        mpc = JointSpaceMPC(k, MPCConfig(acceleration_limit=10))
        reference = np.array([.35, 0, 0, 0, 0, 0])
        solution = mpc.solve(np.zeros(6), np.zeros(6), reference)
        self.assertTrue(solution.feasible, solution.status)
        self.assertGreater(solution.velocity[0], 0)
        self.assertTrue(all(k.distances(q).min() >= mpc.config.clearance for q in solution.trajectory))
        self.assertLess(k.distances(reference * .4).min(), mpc.config.clearance)

    def test_pbvs_uses_hardware_tool_tag_and_cube_pose(self):
        k = backend()
        state = RobotState(1, np.array([.1, -1., 1.3, -.7, .8, .2]), np.zeros(6), .12)
        k.synchronize(state)
        interface = SimpleNamespace(timestamp=1, read_state=Mock(return_value=state),
                                    set_joint_velocities=Mock())
        controller = ApproachController(interface, k, estimator_config=EKFConfig(enabled=False),
            controller_config=ControllerConfig(mode='pure_pbvs', tool_pose_source='tag_only'))
        # Make visual tool deliberately differ from FK to verify tag use.
        measured_tool = pose((.04, 0, 0)) @ k.tool_pose()
        measured_cube_reference = measured_tool @ controller.pbvs.desired
        observation = SimpleNamespace(camera_from_marker=measured_cube_reference)
        tool_observation = SimpleNamespace(camera_from_marker=measured_tool @ k.tool_from_end_effector_marker)
        frame = SimpleNamespace(timestamp=1, world_from_camera=np.eye(4))
        with patch.object(controller.processor, 'process', return_value=observation), \
             patch.object(controller.tool_processor, 'process_marker', return_value=tool_observation):
            result = controller.update(frame)
        self.assertEqual(result.status, 'reached')
        self.assertEqual(result.tool_pose_source, 'tag')
        np.testing.assert_allclose(result.command_velocity, 0, atol=1e-12)

    def test_pickup_has_no_mujoco_model_or_site_dependency(self):
        k = backend(link_capsules=[LinkCapsule('tool', 'tool', (0, 0, 0), (0, 0, .1), .02)])
        q = np.array([0, -1.2, 1.4, -.4, .8, 0])
        state = RobotState(1, q, np.zeros(6), .12)
        k.synchronize(state)
        interface = SimpleNamespace(timestamp=1, read_state=Mock(return_value=state),
            gripper_max_width=.12, open_gripper=Mock(), set_joint_velocities=Mock())
        observation = SimpleNamespace(timestamp=1, world_from_cube=pose(k.grasp_pose()[:3, 3] - [0, 0, .1]))
        pickup = CubePickupController(interface, k)
        pickup.start(observation)
        self.assertEqual(pickup.phase, 'opening')
        self.assertEqual(pickup.update().status, 'pickup_descending')
        result = pickup.update()
        self.assertEqual(result.status, 'pickup_descending', result.diagnostic)
        self.assertGreater(np.linalg.norm(result.command_velocity), 0)

    def test_hardware_controller_imports_and_runs_without_mujoco(self):
        script = '''
import builtins
original = builtins.__import__
def blocked(name, *args, **kwargs):
    if name.split('.')[0] in ('mujoco', 'robot_model'):
        raise ImportError('simulation must not be imported')
    return original(name, *args, **kwargs)
builtins.__import__ = blocked
import numpy as np
from hardware_kinematics import UR3Kinematics
from controller import PBVS, JointSpaceMPC, CubePickupController
from interface import RobotInterface
k = UR3Kinematics(joint_limits=(-np.ones(6), np.ones(6)), world_from_base=np.eye(4),
                  flange_from_tool=np.eye(4), tool_from_grasp=np.eye(4))
assert k.tool_jacobian(np.zeros(6)).shape == (6, 6)
'''
        result = subprocess.run([sys.executable, '-c', script], cwd=Path(__file__).resolve().parents[1],
                                capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == '__main__':
    unittest.main()
