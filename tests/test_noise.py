"""Noise statistics, repeatability, and preservation of clean robot physics."""

from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from simulation.kinematics import MujocoKinematics
from simulation.noise import MeasurementNoise
from simulation.noise_config import NoiseConfig
from common.robot_state import RGBDFrame, RobotState
from simulation.simulate import Simulation


def quiet_config(**settings):
    config = NoiseConfig(
        enabled=True, joint_position_std=0, joint_velocity_std=0,
        gripper_width_std=0, rgb_pixel_std=0, depth_std=0,
        depth_std_per_meter_squared=0, depth_dropout_probability=0,
    )
    return replace(config, **settings)


def clean_state(timestamp=0.0):
    return RobotState(timestamp, np.zeros(6), np.zeros(6), 0.06)


def clean_frame(height=10, width=12):
    return RGBDFrame(
        np.full((height, width, 3), 128, dtype=np.uint8),
        np.ones((height, width), dtype=np.float32), np.eye(3), np.eye(4), 0.0, (0.01, 10.0),
    )


class NoiseTests(unittest.TestCase):
    def test_config_load_and_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "noise.json"
            path.write_text(json.dumps({
                "enabled": True, "seed": 7, "joint_position_std": [0.001] * 6,
                "depth_dropout_probability": 0.1,
            }), encoding="utf-8")
            config = NoiseConfig.load(path)
            self.assertEqual(config.joint_position_std, (0.001,) * 6)
            self.assertEqual(config.seed, 7)
            path.write_text('{"joint_postion_std": 0.01}', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "unknown"):
                NoiseConfig.load(path)
        for settings in (
            {"enabled": 1}, {"seed": True}, {"seed": -1},
            {"joint_position_std": -0.1}, {"joint_position_std": [0.1] * 5},
            {"joint_velocity_bias": [0, 0, 0, 0, 0, float("nan")]},
            {"gripper_width_std": True}, {"rgb_pixel_std": float("inf")},
            {"depth_bias": "0.1"}, {"depth_dropout_probability": 1.1},
        ):
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                NoiseConfig(**settings)

    def test_seed_cache_reset_and_independent_camera_streams(self):
        first = MeasurementNoise(NoiseConfig(enabled=True))
        second = MeasurementNoise(NoiseConfig(enabled=True))
        original = first.apply_state(clean_state(), max_gripper_width=0.12)
        sample = first.apply_state(clean_state(), max_gripper_width=0.12)
        np.testing.assert_array_equal(original.joint_positions, sample.joint_positions)
        sample.joint_positions[:] = 99  # Consumers cannot modify the cached sample.
        np.testing.assert_array_equal(
            original.joint_positions,
            first.apply_state(clean_state(), max_gripper_width=0.12).joint_positions,
        )
        for timestamp in range(8):
            first.apply_frame(clean_frame())  # Camera reads must not affect encoder draws.
            a = first.apply_state(clean_state(timestamp), max_gripper_width=0.12)
            b = second.apply_state(clean_state(timestamp), max_gripper_width=0.12)
            np.testing.assert_array_equal(a.joint_positions, b.joint_positions)
            np.testing.assert_array_equal(a.joint_velocities, b.joint_velocities)
            self.assertEqual(a.gripper_width, b.gripper_width)
        first.reset()
        np.testing.assert_array_equal(
            original.joint_positions,
            first.apply_state(clean_state(), max_gripper_width=0.12).joint_positions,
        )
        # Image/depth draws also replay exactly with a fixed seed.
        first.reset()
        second.reset()
        a, b = first.apply_frame(clean_frame()), second.apply_frame(clean_frame())
        np.testing.assert_array_equal(a.rgb, b.rgb)
        np.testing.assert_array_equal(a.depth, b.depth)

    def test_biases_and_gripper_limits(self):
        config = quiet_config(
            joint_position_bias=[0.001 * i for i in range(6)],
            joint_velocity_bias=-0.01, gripper_width_bias=0.2,
        )
        state = clean_state()
        result = MeasurementNoise(config).apply_state(state, max_gripper_width=0.12)
        np.testing.assert_allclose(result.joint_positions, config.joint_position_bias)
        np.testing.assert_allclose(result.joint_velocities, -0.01)
        self.assertEqual(result.gripper_width, 0.12)
        self.assertEqual(state.gripper_width, 0.06)
        np.testing.assert_array_equal(state.joint_positions, np.zeros(6))
        noise = MeasurementNoise(replace(config, gripper_width_bias=-0.2))
        self.assertEqual(noise.apply_state(state, max_gripper_width=0.12).gripper_width, 0.0)

    def test_depth_dropouts_preserve_invalid_and_background_pixels(self):
        frame = clean_frame()
        frame.depth[0, :5] = [0, np.nan, 10, np.inf, -1]
        source = frame.depth.copy()
        valid = frame.valid_depth.copy()
        output = MeasurementNoise(quiet_config(depth_dropout_probability=1)).apply_frame(frame)
        np.testing.assert_array_equal(output.depth[valid], np.zeros(np.count_nonzero(valid)))
        np.testing.assert_equal(output.depth[~valid], source[~valid])
        np.testing.assert_equal(frame.depth, source)
        self.assertFalse(np.any(output.valid_depth))
        output = MeasurementNoise(quiet_config(depth_bias=-2)).apply_frame(frame)
        self.assertFalse(np.any(output.valid_depth))  # Out-of-range readings become holes.

    def test_gaussian_statistics_and_depth_distance_scaling(self):
        noise = MeasurementNoise(quiet_config(joint_position_std=0.01, joint_velocity_std=0.04))
        samples = [noise.apply_state(clean_state(i), max_gripper_width=0.12) for i in range(5000)]
        positions = np.array([sample.joint_positions for sample in samples])
        velocities = np.array([sample.joint_velocities for sample in samples])
        np.testing.assert_allclose(positions.std(axis=0), 0.01, rtol=0.05)
        np.testing.assert_allclose(velocities.std(axis=0), 0.04, rtol=0.05)
        frame = clean_frame(100, 100)
        frame.depth[50:] = 2.0
        output = MeasurementNoise(quiet_config(depth_std_per_meter_squared=0.001,
                                              depth_bias=0.01)).apply_frame(frame)
        error = output.depth - frame.depth
        self.assertAlmostEqual(float(error[:50].std()), 0.001, delta=0.00008)
        self.assertAlmostEqual(float(error[50:].std()), 0.004, delta=0.00032)
        self.assertAlmostEqual(float(error.mean()), 0.01, delta=0.0001)

    def test_disabled_noise_and_rgb_clipping_keep_owned_arrays(self):
        frame = clean_frame()
        output = MeasurementNoise(NoiseConfig()).apply_frame(frame)
        np.testing.assert_array_equal(output.rgb, frame.rgb)
        np.testing.assert_array_equal(output.depth, frame.depth)
        output.rgb[:] = 0
        output.depth[:] = 0
        output.intrinsics[:] = 0
        self.assertTrue(np.all(frame.rgb == 128))
        self.assertTrue(np.all(frame.depth == 1))
        np.testing.assert_array_equal(frame.intrinsics, np.eye(3))
        output = MeasurementNoise(quiet_config(rgb_pixel_std=1000)).apply_frame(frame)
        self.assertEqual(output.rgb.dtype, np.uint8)
        self.assertTrue(np.any(output.rgb == 0))
        self.assertTrue(np.any(output.rgb == 255))

    def test_interface_noise_preserves_physics_and_uses_measured_camera_pose(self):
        config = NoiseConfig(enabled=True, joint_position_std=0.002)
        simulation = Simulation(noise_config=config)
        robot = simulation.robot
        qpos, qvel, ctrl = robot.data.qpos.copy(), robot.data.qvel.copy(), robot.data.ctrl.copy()
        state = simulation.interface.read_state()
        self.assertFalse(np.array_equal(state.joint_positions, robot.joint_positions))
        with simulation.interface:
            frame = simulation.interface.capture_rgbd()
        prediction = MujocoKinematics(robot)
        clean_pose = prediction.camera_pose()
        prediction.synchronize(state)
        np.testing.assert_allclose(frame.world_from_camera, prediction.camera_pose(), atol=1e-12)
        # Encoder noise affects tool FK, but not the fixed camera mount.
        np.testing.assert_allclose(frame.world_from_camera, clean_pose, atol=1e-12)
        np.testing.assert_array_equal(robot.data.qpos, qpos)
        np.testing.assert_array_equal(robot.data.qvel, qvel)
        np.testing.assert_array_equal(robot.data.ctrl, ctrl)
        self.assertEqual(frame.timestamp, simulation.interface.timestamp)
        simulation.step(25)
        simulation.reset()
        replay = simulation.interface.read_state()
        np.testing.assert_array_equal(replay.joint_positions, state.joint_positions)
        np.testing.assert_array_equal(robot.data.ctrl, ctrl)  # Home commands are not noisy feedback.


if __name__ == "__main__":
    unittest.main()
