"""Experiment placement is applied before sensing and controller creation."""

from pathlib import Path
import json
import tempfile
import unittest

import mujoco
import numpy as np

from common.camera_processing import CameraProcessor
from common.controller import ApproachController
from simulation.kinematics import MujocoKinematics
from simulation.simulation_config import SimulationConfig
from simulation.simulate import Simulation


class SimulationConfigTests(unittest.TestCase):
    def test_custom_placement_visible_to_perception_and_controller(self):
        config = SimulationConfig(cube_position=[-0.054, 0.482, 0.05])
        simulation = Simulation(config=config)
        observation = CameraProcessor().process(simulation.robot.capture_rgbd())
        self.assertIsNotNone(observation)
        np.testing.assert_allclose(observation.world_from_cube[:3, 3], config.cube_position, atol=0.001)
        controller = ApproachController(simulation.interface, MujocoKinematics(simulation.robot))
        cube_id = simulation.robot.model.body("cube").id
        np.testing.assert_allclose(controller.kinematics.data.xpos[cube_id], config.cube_position)
        simulation.step(20)
        simulation.reset()
        mujoco.mj_forward(simulation.robot.model, simulation.robot.data)
        np.testing.assert_allclose(simulation.robot.data.xpos[cube_id], config.cube_position)

    def test_load_and_quaternion_normalization(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "experiment.json"
            path.write_text(json.dumps({"cube_position": [0, 0.5, 0.05],
                                        "cube_quaternion": [2, 0, 0, 0]}), encoding="utf-8")
            config = SimulationConfig.load(path)
        self.assertEqual(config.cube_position, (0, 0.5, 0.05))
        self.assertEqual(config.cube_quaternion, (1, 0, 0, 0))

    def test_invalid_experiment_settings(self):
        for settings in ({"cube_position": [0, 1]}, {"cube_position": [0, float("nan"), 0]},
                         {"cube_quaternion": [0, 0, 0, 0]}):
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                SimulationConfig(**settings)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "experiment.json"
            path.write_text('{"cube_postion": [0, 0, 0]}', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "unknown"):
                SimulationConfig.load(path)


if __name__ == "__main__":
    unittest.main()
