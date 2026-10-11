"""PBVS direction, constrained MPC, and actual closed-loop approach checks."""

from dataclasses import replace
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch
from contextlib import ExitStack

import mujoco
import numpy as np

from controller import ApproachController, JointSpaceMPC, MujocoKinematics, PBVS
from interface import MujocoInterface
from robot_model import RobotModel
from simulate import Simulation


class ControllerTests(unittest.TestCase):
    def test_pbvs_direction_and_zero_error(self):
        desired = np.eye(4)
        desired[:3, :3] = np.diag([1, -1, -1])
        desired[2, 3] = 0.2
        pbvs = PBVS(desired)
        reference, error = pbvs.reference(desired, np.eye(6))
        np.testing.assert_allclose(reference, 0, atol=1e-12)
        observed = desired.copy()
        observed[2, 3] = 0.4
        reference, error = pbvs.reference(observed, np.eye(6))
        self.assertGreater(reference[2], 0)  # camera must move forward toward tag
        np.testing.assert_allclose(error[:3], [0, 0, 0.2], atol=1e-12)

    def test_full_vision_and_physics_loop_reaches_tool_target(self):
        simulation = Simulation()
        result = simulation.run_controlled(6000)
        self.assertEqual(result.status, "reached")
        self.assertLess(result.position_error, 0.003)
        self.assertLess(result.rotation_error, 0.025)
        self.assertGreaterEqual(result.minimum_clearance, 0.02)
        mujoco.mj_forward(simulation.robot.model, simulation.robot.data)
        # Tool center aligns with cube center, 18 cm above the tagged face.
        np.testing.assert_allclose(simulation.robot.data.site("attachment_site").xpos,
                                   [-0.074, 0.492, 0.28015], atol=0.004)
        self.assertFalse(any(simulation.robot.data.contact[i].dist < -1e-5
                             for i in range(simulation.robot.data.ncon)))
        simulation.reset()
        np.testing.assert_allclose(simulation.robot.model.actuator_gainprm[simulation.robot.actuator_ids, 0],
                                   [2000, 2000, 2000, 500, 500, 500])

    def test_obstacle_constraints_override_pbvs_reference(self):
        robot = self._obstacle_robot()
        controller = ApproachController(robot, MujocoInterface(robot))
        result = controller.update(robot.capture_rgbd())
        self.assertEqual(result.status, "tracking")
        q = robot.joint_positions
        config = controller.mpc.config
        unsafe_prediction = q + config.dt * config.horizon * result.reference_velocity
        self.assertLess(np.min(controller.kinematics.distances(unsafe_prediction)), config.clearance)
        self.assertGreaterEqual(result.minimum_clearance, config.clearance)
        self.assertFalse(np.allclose(result.command_velocity, result.reference_velocity))
        for q_predicted in q + np.cumsum(controller.mpc.previous_plan, axis=0) * config.dt:
            self.assertGreaterEqual(np.min(controller.kinematics.distances(q_predicted)), config.clearance)

    def test_distance_gradient_matches_finite_difference(self):
        robot = self._obstacle_robot()
        kinematics = MujocoKinematics(robot)
        pair_index = kinematics.pairs.index((29, robot.model.geom("test_obstacle_geom").id))
        q = robot.joint_positions
        _, gradients = kinematics.distances(q, gradients=True)
        numeric = np.zeros(6)
        for i in range(6):
            perturbation = np.eye(6)[i] * 1e-5
            numeric[i] = (kinematics.distances(q + perturbation)[pair_index]
                          - kinematics.distances(q - perturbation)[pair_index]) / 2e-5
        np.testing.assert_allclose(gradients[pair_index], numeric, atol=1e-4)

    def test_actual_motion_preserves_clearance_with_blocking_obstacle(self):
        robot = self._obstacle_robot()
        simulation = Simulation(robot)
        controller = ApproachController(robot, simulation.interface)
        minimum = np.inf
        with mujoco.Renderer(robot.model, height=480, width=640) as renderer:
            for _ in range(100):
                result = controller.update(robot.capture_rgbd(renderer=renderer))
                self.assertEqual(result.status, "tracking")
                simulation.step(25)
                minimum = min(minimum, float(np.min(controller.kinematics.distances(robot.joint_positions))))
        self.assertGreaterEqual(minimum, controller.mpc.config.clearance)
        self.assertFalse(any(robot.data.contact[i].dist < -1e-5 for i in range(robot.data.ncon)))

    def test_joint_position_velocity_and_acceleration_constraints(self):
        robot = RobotModel()
        kinematics = MujocoKinematics(robot)
        # Pan-axis upper bound close enough to be active within the horizon.
        upper = kinematics.limits[1].copy()
        upper[0] = robot.joint_positions[0] + 0.01
        kinematics.limits = kinematics.limits[0], upper
        mpc = JointSpaceMPC(kinematics)
        solution = mpc.solve(robot.joint_positions, np.zeros(6), [0.3, 0, 0, 0, 0, 0])
        self.assertTrue(solution.feasible, solution.status)
        self.assertTrue(np.all(solution.trajectory <= upper + 1e-6))
        self.assertLessEqual(np.max(np.abs(mpc.previous_plan)), mpc.config.velocity_limit + 1e-6)
        changes = np.diff(np.vstack((np.zeros(6), mpc.previous_plan)), axis=0)
        self.assertLessEqual(np.max(np.abs(changes)), mpc.config.acceleration_limit * mpc.config.dt + 1e-6)

    def test_target_loss_and_stale_frames_stop(self):
        robot = RobotModel()
        interface = MujocoInterface(robot)
        controller = ApproachController(robot, interface)
        frame = robot.capture_rgbd()
        controller.update(frame)
        for unusable in (replace(frame, rgb=np.full_like(frame.rgb, 255)),
                         replace(frame, timestamp=-1.0)):
            result = controller.update(unusable)
            self.assertEqual(result.status, "target_lost_or_stale")
            np.testing.assert_array_equal(result.command_velocity, np.zeros(6))
            np.testing.assert_array_equal(robot.data.ctrl[robot.actuator_ids], np.zeros(6))

    def test_solver_failure_stops(self):
        robot = RobotModel()
        mpc = JointSpaceMPC(MujocoKinematics(robot))
        with patch("controller.minimize", return_value=SimpleNamespace(success=False, message="test failure")):
            solution = mpc.solve(robot.joint_positions, np.zeros(6), np.ones(6) * 0.1)
        self.assertFalse(solution.feasible)
        np.testing.assert_array_equal(solution.velocity, np.zeros(6))

    def test_initial_clearance_violation_stops(self):
        robot = self._obstacle_robot(offset=0.045)
        controller = ApproachController(robot, MujocoInterface(robot))
        result = controller.update(robot.capture_rgbd())
        self.assertEqual(result.status, "clearance_violation")
        np.testing.assert_array_equal(result.command_velocity, np.zeros(6))

    def test_velocity_servo_and_position_mode_restoration(self):
        simulation = Simulation()
        simulation.interface.set_joint_velocities([0.1, 0, 0, 0, 0, 0])
        simulation.step(100)
        np.testing.assert_allclose(simulation.interface.joint_velocities,
                                   [0.1, 0, 0, 0, 0, 0], atol=0.01)
        current = simulation.interface.joint_positions
        simulation.interface.set_joint_positions(current)
        self.assertFalse(simulation.interface._velocity_mode)
        np.testing.assert_allclose(simulation.robot.data.qfrc_applied, 0)

    def test_viewer_loop_advances_control_intervals_and_moves_robot(self):
        # Render and solve normally; mock only the user-facing window calls.
        simulation = Simulation()
        initial = simulation.robot.joint_positions
        viewer = MagicMock()
        viewer.is_running.side_effect = [True] * 105 + [False]
        context = MagicMock()
        context.__enter__.return_value = viewer
        with ExitStack() as stack:
            stack.enter_context(patch("mujoco.viewer.launch_passive", return_value=context))
            for name in ("namedWindow", "imshow", "destroyWindow"):
                stack.enter_context(patch(f"cv2.{name}"))
            stack.enter_context(patch("cv2.waitKey", return_value=-1))
            stack.enter_context(patch("cv2.getWindowProperty", return_value=1))
            stack.enter_context(patch("simulate.time.sleep"))
            simulation.launch_viewer(servo=True)
        self.assertAlmostEqual(simulation.robot.data.time, 5.25, places=6)
        self.assertGreater(np.linalg.norm(simulation.robot.joint_positions - initial), 0.3)
        mujoco.mj_forward(simulation.robot.model, simulation.robot.data)
        np.testing.assert_allclose(simulation.robot.data.site("attachment_site").xpos,
                                   [-0.074, 0.492, 0.28015], atol=0.004)
        np.testing.assert_array_equal(simulation.robot.data.ctrl[simulation.robot.actuator_ids], np.zeros(6))

    @staticmethod
    def _obstacle_robot(offset=0.065):
        robot = RobotModel()
        center = robot.data.geom_xpos[29] + [0, 0, -offset]
        spec = mujoco.MjSpec.from_file(str(robot.model_path))
        body = spec.worldbody.add_body(name="test_obstacle", pos=center)
        body.add_geom(name="test_obstacle_geom", type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[0.02, 0, 0])
        # Existing joint/site/actuator indices are unchanged by an extra fixed body.
        robot.model = spec.compile()
        robot.data = mujoco.MjData(robot.model)
        robot.reset()
        return robot


if __name__ == "__main__":
    unittest.main()
