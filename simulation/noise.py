"""Seeded measurement corruption, separate from robot physics and commands."""

from dataclasses import replace

import numpy as np

from simulation.noise_config import NoiseConfig
from common.robot_state import RGBDFrame, RobotState


class MeasurementNoise:
    """Add configurable sensor errors without changing the source data.

    Separate random streams keep camera draws from changing encoder noise.
    Repeated reads of the same clean state return the same noisy sample, with
    independent arrays. Each camera acquisition draws a new image sample.
    reset() restarts the sequence when a fixed seed is configured.
    """

    def __init__(self, config: NoiseConfig | None = None):
        self.config = config or NoiseConfig()
        self.reset()

    def reset(self) -> None:
        streams = np.random.SeedSequence(self.config.seed).spawn(6)
        (self._position_rng, self._velocity_rng, self._gripper_rng,
         self._rgb_rng, self._depth_rng, self._dropout_rng) = (
            np.random.default_rng(stream) for stream in streams
        )
        self._state_key = None
        self._state_sample = None

    def apply_state(self, state: RobotState, *, max_gripper_width: float) -> RobotState:
        """Corrupt feedback only; biases stay fixed throughout the experiment."""
        if not self.config.enabled:
            return replace(state, joint_positions=state.joint_positions.copy(),
                           joint_velocities=state.joint_velocities.copy())
        key = (state.timestamp, state.joint_positions.tobytes(),
               state.joint_velocities.tobytes(), state.gripper_width, max_gripper_width)
        if key != self._state_key:
            c = self.config
            positions = (state.joint_positions + np.asarray(c.joint_position_bias)
                         + self._position_rng.normal(0, c.joint_position_std, size=6))
            velocities = (state.joint_velocities + np.asarray(c.joint_velocity_bias)
                          + self._velocity_rng.normal(0, c.joint_velocity_std, size=6))
            width = state.gripper_width + c.gripper_width_bias + self._gripper_rng.normal(0, c.gripper_width_std)
            self._state_sample = replace(
                state, joint_positions=positions, joint_velocities=velocities,
                gripper_width=float(np.clip(width, 0, max_gripper_width)),
            )
            self._state_key = key
        return replace(self._state_sample,
                       joint_positions=self._state_sample.joint_positions.copy(),
                       joint_velocities=self._state_sample.joint_velocities.copy())

    def apply_frame(self, frame: RGBDFrame) -> RGBDFrame:
        """Corrupt RGB and valid metric depth; preserve invalid/background depth."""
        c = self.config
        rgb, depth = frame.rgb.copy(), frame.depth.copy()
        if c.enabled:
            if c.rgb_pixel_std > 0:
                pixels = rgb.astype(float) + self._rgb_rng.normal(0, c.rgb_pixel_std, size=rgb.shape)
                rgb = np.clip(np.rint(pixels), 0, 255).astype(np.uint8)
            valid = frame.valid_depth
            values = depth[valid].astype(float)
            sigma = c.depth_std + c.depth_std_per_meter_squared * values ** 2
            values += c.depth_bias + self._depth_rng.normal(0, sigma)
            near, far = frame.depth_range
            invalid = (~np.isfinite(values) | (values <= near)
                       | (values >= far * (1 - 1e-5)))
            if c.depth_dropout_probability > 0:
                invalid |= self._dropout_rng.random(len(values)) < c.depth_dropout_probability
            values[invalid] = 0.0
            depth[valid] = values
        return replace(frame, rgb=rgb, depth=depth,
                       intrinsics=frame.intrinsics.copy(),
                       world_from_camera=frame.world_from_camera.copy())
