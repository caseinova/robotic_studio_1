"""Pickup sequencing, interface portability, and real contact-driven lifting."""

from dataclasses import replace
from contextlib import ExitStack
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, Mock, patch

import mujoco
import numpy as np

from controller import ApproachController, ControlResult, CubePickupController
from controller_config import ControllerConfig
from kinematics import MujocoKinematics
from robot_model import RobotModel
from robot_state import RobotState
from simulate import Simulation
from simulation_config import SimulationConfig
from noise_config import NoiseConfig


class PickupTests(unittest.TestCase):
    def test_dwell_allows_xy_motion_inside_radius_and_resets_outside(self):
        original, interface, observation, state = self._recorded_pickup()
        controller = CubePickupController(interface, original.kinematics,
            pickup_dwell_seconds=1., pickup_dwell_radius_m=.01)
        controller.start(observation)
        self.assertEqual(controller.update().status, 'pickup_dwelling')
        grasp = original.kinematics.grasp_pose().copy()
        with patch.object(original.kinematics, 'grasp_pose', return_value=grasp):
            for t, dx, expected in ((10., .002, 'dwelling'), (10.6, -.004, 'dwelling'),
                                     (10.7, .011, 'dwelling'), (10.8, .003, 'dwelling'),
                                     (11.5, -.003, 'dwelling'), (11.81, .005, 'descending')):
                interface.timestamp = t
                interface.read_state.return_value = replace(state, timestamp=t)
                grasp[0, 3] = observation.world_from_cube[0, 3] + dx
                controller.update()
                self.assertEqual(controller.phase, expected)
        for call in interface.set_joint_velocities.call_args_list:
            np.testing.assert_array_equal(call.args[0], np.zeros(6))
        interface.open_gripper.assert_called_once()
        interface.close_gripper.assert_not_called()

    def test_dwell_stale_state_fails_instead_of_descending(self):
        original, interface, observation, state = self._recorded_pickup()
        controller = CubePickupController(interface, original.kinematics, pickup_dwell_seconds=1.)
        controller.start(observation)
        controller.update()
        interface.timestamp = 12.
        self.assertEqual(controller.update().diagnostic, 'stale_or_invalid_robot_state')

    def test_grasp_height_offset_changes_only_world_z_target(self):
        for offset in (0., .005, -.005):
            controller, interface, observation, state = self._recorded_pickup()
            controller = CubePickupController(interface, controller.kinematics,
                                              grasp_height_offset_m=offset)
            grasp = controller.kinematics.grasp_pose().copy()
            entry = controller.kinematics.tool_pose().copy()
            controller.start(observation)
            expected = entry.copy()
            expected[2, 3] += observation.world_from_cube[2, 3] + offset - grasp[2, 3]
            np.testing.assert_allclose(controller._descent_pose, expected)
            np.testing.assert_allclose(controller._entry_pose, entry)
            self.assertEqual(controller.phase, 'opening')

    def test_offset_above_entry_rejected_before_opening(self):
        controller, interface, observation, state = self._recorded_pickup()
        controller = CubePickupController(interface, controller.kinematics, grasp_height_offset_m=.2)
        controller.start(observation)
        self.assertEqual(controller.phase, 'failed')
        interface.open_gripper.assert_not_called()

    def test_configured_frame_budget_keeps_state_budget_unchanged(self):
        for ratio, frame_age, state_age, accepted in (
                (2., .15, 0., False), (4., .15, 0., True),
                (4., .20, 0., True), (4., .201, 0., False), (4., .15, .11, False)):
            with self.subTest(ratio=ratio, frame_age=frame_age, state_age=state_age):
                interface = Mock(timestamp=1., gripper_max_width=.11)
                interface.read_state.return_value = RobotState(1.-state_age, np.zeros(6), np.zeros(6), .08)
                model = Mock()
                model.pickup_collision_indices.return_value = []
                pose = np.eye(4)
                pose[2, 3] = .3
                model.tool_pose.return_value = pose
                model.grasp_pose.return_value = pose
                controller = CubePickupController(interface, model, cube_size=.064,
                    frame_max_age_intervals=ratio)
                controller.start(SimpleNamespace(timestamp=1.-frame_age, world_from_cube=np.eye(4)))
                self.assertEqual(controller.phase, 'opening' if accepted else 'failed')
                if not accepted:
                    interface.open_gripper.assert_not_called()
                    self.assertEqual(controller.diagnostic, 'stale_pickup_target_or_state')

    def test_physical_pickup_after_both_approach_modes(self):
        for mode, position in (
            ("pure_pbvs", [-0.079, 0.30, 0.05]),
            ("pbvs_mpc", [-0.074, 0.32, 0.05]),
        ):
            with self.subTest(mode=mode):
                simulation = Simulation(
                    pickup=True, config=SimulationConfig(cube_position=position),
                    controller_config=ControllerConfig(mode=mode),
                )
                records = []
                original_update = CubePickupController.update

                def record_update(controller):
                    result = original_update(controller)
                    records.append((
                        controller.phase,
                        simulation.robot.data.site("attachment_site").xpos.copy(),
                        controller._entry_pose.copy(),
                    ))
                    return result

                with patch.object(CubePickupController, "update", record_update), \
                     patch.object(simulation.interface, "set_gripper_width",
                                  wraps=simulation.interface.set_gripper_width) as grip_commands:
                    result = simulation.run_controlled(15000)
                self.assertEqual(result.status, "pickup_complete", result.diagnostic)
                self.assertEqual([call.args[0] for call in grip_commands.call_args_list], [0.12, 0.0])
                self.assertEqual(set(phase for phase, _, _ in records), {
                    "descending", "closing", "lifting", "complete"
                })
                drift = max(np.linalg.norm(tool[:2] - entry[:2, 3]) for _, tool, entry in records)
                self.assertLess(drift, 0.003)  # Measured straight-line tracking, including payload.
                cube_height = simulation.robot.data.body("cube").xpos[2]
                self.assertGreater(cube_height, position[2] + 0.09)
                self.assertAlmostEqual(simulation.interface.gripper_width, 0.10, delta=0.003)
                np.testing.assert_allclose(records[-1][1], records[0][1], atol=0.002)
                # Keep advancing physics: the cube must stay gripped and raised.
                simulation.step(1000)
                self.assertGreater(simulation.robot.data.body("cube").xpos[2], position[2] + 0.08)

    def _recorded_pickup(self, robot=None):
        robot = RobotModel() if robot is None else robot
        positions = robot.joint_positions
        kinematics = MujocoKinematics(robot)
        robot.data = None  # Prediction must not depend on a live simulator.
        state = RobotState(10.0, positions, np.zeros(6), 0.12)
        interface = SimpleNamespace(
            timestamp=10.0, gripper_max_width=0.12, read_state=Mock(return_value=state),
            capture_rgbd=Mock(side_effect=AssertionError("pickup does not require fresh vision")),
            open_gripper=Mock(), close_gripper=Mock(),
            set_joint_positions=Mock(), set_joint_velocities=Mock(),
        )
        grasp_position = kinematics.data.site("grasp_site").xpos.copy()
        pose = np.eye(4)
        pose[:3, 3] = grasp_position - [0, 0, 0.1]
        observation = SimpleNamespace(timestamp=10.0, world_from_cube=pose)
        return CubePickupController(interface, kinematics), interface, observation, state

    def test_pickup_uses_interface_and_descends_without_camera(self):
        controller, interface, observation, state = self._recorded_pickup()
        entry = controller.kinematics.tool_pose().copy()
        controller.start(observation)
        interface.open_gripper.assert_called_once()
        self.assertEqual(controller.update().status, "pickup_descending")
        result = controller.update()
        self.assertEqual(result.status, "pickup_descending", result.diagnostic)
        interface.capture_rgbd.assert_not_called()
        interface.close_gripper.assert_not_called()
        predicted = controller.kinematics.tool_pose()
        self.assertLess(predicted[2, 3], entry[2, 3])
        np.testing.assert_allclose(predicted[:2, 3], entry[:2, 3], atol=0.0001)
        interface.timestamp = 11.0
        result = controller.update()
        self.assertEqual(result.status, "pickup_failed")
        self.assertEqual(result.diagnostic, "stale_or_invalid_robot_state")
        np.testing.assert_array_equal(interface.set_joint_velocities.call_args.args[0], np.zeros(6))

    def test_bad_handoff_and_small_gripper_stop_before_motion(self):
        for problem, expected in (
            ("stale", "stale_pickup_target_or_state"),
            ("offset", "tool_not_centered_over_cube"),
            ("small", "gripper_opening_too_small"),
        ):
            controller, interface, observation, state = self._recorded_pickup()
            if problem == "stale":
                observation.timestamp = 9.0
            elif problem == "offset":
                observation.world_from_cube[0, 3] += 0.05
            else:
                interface.gripper_max_width = 0.085
            controller.start(observation)
            self.assertEqual(controller.phase, "failed")
            if problem == "offset":
                self.assertTrue(controller.diagnostic.startswith(expected + ":"))
                self.assertIn("dx=-50.0 mm", controller.diagnostic)
                self.assertIn("distance=50.0 mm, limit=10.0 mm", controller.diagnostic)
                self.assertIn("grasp_xy_mm=", controller.diagnostic)
                self.assertIn("cube_xy_mm=", controller.diagnostic)
            else:
                self.assertEqual(controller.diagnostic, expected)
            interface.open_gripper.assert_not_called()
            interface.close_gripper.assert_not_called()
            np.testing.assert_array_equal(interface.set_joint_velocities.call_args.args[0], np.zeros(6))

    def test_empty_grasp_never_starts_lifting(self):
        controller, interface, observation, state = self._recorded_pickup()
        controller.start(observation)
        # Reproduce the measured result of fingers closing on empty space.
        controller.phase = "closing"
        controller._phase_started = 10.0
        interface.timestamp = 11.0
        interface.read_state.return_value = replace(state, timestamp=11.0, gripper_width=0.0)
        result = controller.update()
        self.assertEqual(result.status, "pickup_failed")
        self.assertEqual(result.diagnostic, "empty_grasp")
        np.testing.assert_array_equal(interface.set_joint_velocities.call_args.args[0], np.zeros(6))

    def test_opening_reports_width_and_timeout_without_retry_or_descent(self):
        controller, interface, observation, state = self._recorded_pickup()
        interface.read_state.return_value = replace(state, gripper_width=.08)
        controller.start(observation)
        result = controller.update()
        self.assertEqual(result.status, "pickup_opening")
        self.assertIn("gripper_width=80.0 mm", result.diagnostic)
        self.assertIn("requested=120.0 mm", result.diagnostic)
        interface.timestamp = 31.
        interface.read_state.return_value = replace(state, timestamp=31., gripper_width=.08)
        result = controller.update()
        self.assertEqual(result.status, "pickup_failed")
        self.assertIn("phase_timeout: opening", result.diagnostic)
        self.assertIn("required>=119.0 mm", result.diagnostic)
        interface.open_gripper.assert_called_once()
        interface.close_gripper.assert_not_called()
        for call in interface.set_joint_velocities.call_args_list:
            np.testing.assert_array_equal(call.args[0], np.zeros(6))

    def test_opening_progress_clears_on_descent(self):
        controller, interface, observation, state = self._recorded_pickup()
        controller.start(observation)
        interface.read_state.return_value = replace(state, gripper_width=.08)
        self.assertTrue(controller.update().diagnostic)
        interface.read_state.return_value = state
        result = controller.update()
        self.assertEqual(result.status, "pickup_descending")
        self.assertEqual(result.diagnostic, "")

    def test_closing_accepts_noisy_settled_width_but_not_moving_fingers(self):
        for moving in (False, True):
            with self.subTest(moving=moving):
                controller, interface, observation, state = self._recorded_pickup()
                controller.start(observation)
                controller._change_phase("closing", 10.0)
                for i in range(21):
                    now = 10.0 + 0.05 * i
                    width = (0.114 - 0.005 * (now - 10.0) if moving
                             else 0.10 + (0.001 if i % 2 else -0.001))
                    interface.timestamp = now
                    interface.read_state.return_value = replace(
                        state, timestamp=now, gripper_width=width,
                    )
                    controller.update()
                    if i < 10:
                        self.assertEqual(controller.phase, "closing")
                    if controller.phase == "lifting":
                        break
                self.assertEqual(controller.phase, "closing" if moving else "lifting")

    def test_repeated_width_read_does_not_count_as_settling(self):
        controller, interface, observation, state = self._recorded_pickup()
        controller.start(observation)
        controller._change_phase("closing", 10.0)
        interface.timestamp = 11.0
        interface.read_state.return_value = replace(state, timestamp=11.0, gripper_width=0.10)
        for _ in range(20):
            controller.update()
        self.assertEqual(controller.phase, "closing")

    def test_irregular_samples_allow_settled_grasp_but_not_moving_fingers(self):
        for moving in (False, True):
            controller, interface, observation, state = self._recorded_pickup()
            controller.start(observation)
            controller._change_phase('closing', 10.)
            for i in range(18):
                now = 10. + i*.063
                width = .105 - .005*(now-10.) if moving else .1
                interface.timestamp = now
                interface.read_state.return_value = replace(state, timestamp=now, gripper_width=width)
                result = controller.update()
                if controller.phase == 'lifting':
                    break
            self.assertEqual(controller.phase, 'closing' if moving else 'lifting')
            if moving:
                self.assertIn('settled=False', result.diagnostic)

    def test_settling_rejects_gap_in_feedback(self):
        controller, _, _, _ = self._recorded_pickup()
        for t in (10., 10.06, 10.12, 10.18, 10.24, 10.6):
            _, settled = controller._gripper_settling(t, .1)
        self.assertFalse(settled)

    def test_closing_timeout_reports_measured_width(self):
        controller, interface, observation, state = self._recorded_pickup()
        controller.start(observation)
        controller._change_phase('closing', 10.)
        interface.timestamp = 31.
        interface.read_state.return_value = replace(state, timestamp=31., gripper_width=.12)
        result = controller.update()
        self.assertEqual(result.status, 'pickup_failed')
        self.assertIn('phase_timeout: closing', result.diagnostic)
        self.assertIn('gripper_width=120.0 mm', result.diagnostic)
        self.assertIn('settled=False', result.diagnostic)

    def test_physical_pickup_with_noisy_measurements(self):
        simulation = Simulation(
            pickup=True, config=SimulationConfig(cube_position=[-0.079, 0.32, 0.05]),
            controller_config=ControllerConfig(mode="pure_pbvs"),
            noise_config=NoiseConfig(
                enabled=True, seed=42, joint_position_std=0.003,
                joint_velocity_std=0.01, gripper_width_std=0.001,
                depth_std=0.002, depth_std_per_meter_squared=0.005,
                depth_dropout_probability=0.01,
            ),
        )
        result = simulation.run_controlled(20000)
        self.assertEqual(result.status, "pickup_complete", result.diagnostic)
        self.assertGreater(simulation.robot.data.body("cube").xpos[2], 0.14)
        simulation.step(1000)
        self.assertGreater(simulation.robot.data.body("cube").xpos[2], 0.13)

    def test_other_obstacle_contact_is_not_allowed(self):
        robot = RobotModel()
        finger = robot.model.geom("gripper_left_geom").id
        spec = mujoco.MjSpec.from_file(str(robot.model_path))
        obstacle = spec.worldbody.add_body(name="pickup_blocker", pos=robot.data.geom_xpos[finger])
        obstacle.add_geom(type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[0.02, 0, 0])
        robot.model = spec.compile()
        robot.data = mujoco.MjData(robot.model)
        robot.reset()
        controller, interface, observation, state = self._recorded_pickup(robot)
        controller.start(observation)
        controller.update()  # Confirm full opening before descending.
        result = controller.update()
        self.assertEqual(result.status, "pickup_failed")
        self.assertTrue(result.diagnostic.startswith("pickup_path_blocked:"))
        self.assertIn("clearance=", result.diagnostic)
        self.assertIn("required=15.0 mm", result.diagnostic)
        self.assertIn("at=", result.diagnostic)
        np.testing.assert_array_equal(interface.set_joint_velocities.call_args.args[0], np.zeros(6))

    def test_viewer_hands_off_to_pickup_once(self):
        simulation = Simulation(pickup=True)
        approach = ApproachController(simulation.interface, MujocoKinematics(simulation.robot))
        with simulation.interface:
            observation = approach.processor.process(simulation.interface.capture_rgbd())
        reached = ControlResult(observation, np.zeros(6), np.zeros(6), "reached", 0, 0, np.nan)
        approach.update = Mock(return_value=reached)
        pickup = SimpleNamespace(phase="idle", start=Mock(), update=Mock())
        pickup.start.side_effect = lambda _: setattr(pickup, "phase", "lifting")
        pickup.update.return_value = ControlResult(
            None, np.zeros(6), np.zeros(6), "pickup_lifting", 0, 0, np.nan
        )
        viewer = MagicMock()
        viewer.is_running.side_effect = [True, True, False]
        context = MagicMock()
        context.__enter__.return_value = viewer
        with ExitStack() as stack:
            stack.enter_context(patch("controller.ApproachController", return_value=approach))
            stack.enter_context(patch("controller.CubePickupController", return_value=pickup))
            stack.enter_context(patch("mujoco.viewer.launch_passive", return_value=context))
            for name in ("namedWindow", "imshow", "destroyWindow"):
                stack.enter_context(patch(f"cv2.{name}"))
            stack.enter_context(patch("cv2.waitKey", return_value=-1))
            stack.enter_context(patch("cv2.getWindowProperty", return_value=1))
            stack.enter_context(patch("simulate.time.sleep"))
            simulation.launch_viewer()
        approach.update.assert_called_once()
        pickup.start.assert_called_once_with(observation)
        self.assertEqual(pickup.update.call_count, 2)


if __name__ == "__main__":
    unittest.main()
