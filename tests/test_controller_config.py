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
from controller_config import ControllerConfig
from simulate import Simulation
from simulation_config import SimulationConfig


class ControllerConfigTests(unittest.TestCase):
    def test_load_modes_and_reject_unknown_settings(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "controller.json"
            for mode in ("pure_pbvs", "pbvs_mpc"):
                path.write_text(json.dumps({"mode": mode}), encoding="utf-8")
                self.assertEqual(ControllerConfig.load(path).mode, mode)
            path.write_text('{"mode": "invalid"}', encoding="utf-8")
            with self.assertRaises(ValueError):
                ControllerConfig.load(path)
            path.write_text('{"mod": "pure_pbvs"}', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "unknown"):
                ControllerConfig.load(path)

    def test_pure_pbvs_bypasses_mpc_and_collision_queries(self):
        config = ControllerConfig(mode="pure_pbvs")
        simulation = Simulation(controller_config=config)
        controller = ApproachController(simulation.robot, simulation.interface, controller_config=config)
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
        controller = ApproachController(simulation.robot, simulation.interface, controller_config=settings)
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
