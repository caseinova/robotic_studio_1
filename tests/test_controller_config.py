"""Controller mode selection and the direct PBVS execution path."""

from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import mujoco

from controller import ApproachController
from kinematics import MujocoKinematics
from controller_config import ControllerConfig
from simulate import Simulation
from simulation_config import SimulationConfig


class ControllerConfigTests(unittest.TestCase):
    def test_load_modes_and_reject_unknown_settings(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "controller.json"
            for mode in ("pure_pbvs", "pbvs_mpc"):
                path.write_text(json.dumps({"mode": mode, "tool_clearance": 0.2}), encoding="utf-8")
                loaded = ControllerConfig.load(path)
                self.assertEqual(loaded.mode, mode)
                self.assertEqual(loaded.tool_clearance, 0.2)
            path.write_text('{"mode": "invalid"}', encoding="utf-8")
            with self.assertRaises(ValueError):
                ControllerConfig.load(path)
            path.write_text('{"mod": "pure_pbvs"}', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "unknown"):
                ControllerConfig.load(path)

    def test_invalid_target_distance(self):
        for distance in (0.159, -1, True, "0.2", None, float("nan"), float("inf")):
            with self.subTest(distance=distance), self.assertRaises(ValueError):
                ControllerConfig(tool_clearance=distance)
        self.assertEqual(ControllerConfig(tool_clearance=0.16).tool_clearance, 0.16)

    def test_configured_final_position_and_explicit_override(self):
        simulation = Simulation()
        settings = ControllerConfig(tool_clearance=0.2)
        for override, distance in ((None, 0.2), (0.22, 0.22)):
            controller = ApproachController(
                simulation.interface, MujocoKinematics(simulation.robot),
                controller_config=settings, tool_clearance=override,
            )
            # Desired marker-from-tool = marker-from-camera * camera-from-tool.
            desired = np.linalg.inv(controller.pbvs.desired) @ np.linalg.inv(
                controller.kinematics.tool_from_camera
            )
            np.testing.assert_allclose(desired[:3, 3], [0, 0, distance], atol=1e-12)
            np.testing.assert_allclose(desired[:3, :3], np.diag([1, -1, -1]), atol=1e-12)

    def test_nondefault_distance_reached_by_both_modes(self):
        for mode in ("pure_pbvs", "pbvs_mpc"):
            with self.subTest(mode=mode):
                settings = ControllerConfig(mode=mode, tool_clearance=0.2)
                simulation = Simulation(controller_config=settings)
                result = simulation.run_controlled(6000)
                self.assertEqual(result.status, "reached")
                mujoco.mj_forward(simulation.robot.model, simulation.robot.data)
                np.testing.assert_allclose(
                    simulation.robot.data.site("attachment_site").xpos,
                    [-0.074, 0.492, 0.30015], atol=0.004,
                )

    def test_pure_pbvs_bypasses_mpc_and_collision_queries(self):
        config = ControllerConfig(mode="pure_pbvs")
        simulation = Simulation(controller_config=config)
        controller = ApproachController(simulation.interface, MujocoKinematics(simulation.robot), controller_config=config)
        self.assertIsNone(controller.mpc)
        frame = simulation.robot.capture_rgbd()
        with patch("controller.JointSpaceMPC.solve", side_effect=AssertionError("MPC must not run")), \
             patch.object(controller.kinematics, "distances", side_effect=AssertionError("no collision queries")):
            result = controller.update(frame)
        self.assertEqual(result.status, "tracking")
        np.testing.assert_allclose(result.command_velocity, result.reference_velocity)
        self.assertTrue(np.isnan(result.minimum_clearance))
        lost = controller.update(replace(frame, rgb=np.full_like(frame.rgb, 255)))
        np.testing.assert_array_equal(lost.command_velocity, np.zeros(6))

    def test_pure_pbvs_closed_loop_reaches_goal(self):
        simulation = Simulation(controller_config=ControllerConfig(mode="pure_pbvs"))
        with patch("controller.JointSpaceMPC.solve", side_effect=AssertionError("MPC must not run")):
            result = simulation.run_controlled(6000)
        self.assertEqual(result.status, "reached")
        self.assertLess(result.position_error, 0.003)
        np.testing.assert_array_equal(simulation.robot.data.ctrl[simulation.robot.actuator_ids], np.zeros(6))

    def test_shadowed_tag_at_offset_cube_placement_does_not_stop_approach(self):
        # This placement previously froze at t=1.95 s: a shadow pulled a
        # subpixel corner off the tag and depth-refined pose was rejected.
        settings = ControllerConfig(mode="pure_pbvs")
        simulation = Simulation(
            config=SimulationConfig(cube_position=[-0.079, 0.432, 0.05]),
            controller_config=settings,
        )
        controller = ApproachController(simulation.interface, MujocoKinematics(simulation.robot), controller_config=settings)
        with mujoco.Renderer(simulation.robot.model, height=480, width=640) as renderer:
            for _ in range(200):
                simulation.step(25)
                result = controller.update(simulation.robot.capture_rgbd(renderer=renderer))
                self.assertIsNotNone(result.observation, result.diagnostic)
                if result.status == "reached":
                    break
        self.assertEqual(result.status, "reached")
        self.assertLess(result.position_error, 0.003)


if __name__ == "__main__":
    unittest.main()
