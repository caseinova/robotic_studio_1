"""Simulation measurement noise settings; standard deviations use sensor units."""

from dataclasses import dataclass, fields
import json
import math
from pathlib import Path


DEFAULT_NOISE_CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "noise_config.json"


@dataclass(frozen=True)
class NoiseConfig:
    """Independent Gaussian measurement errors plus fixed bias and depth holes.

    Joint settings accept one number for all joints or six numbers in arm order.
    Position units are radians, velocity radians/second, gripper/depth meters,
    and RGB standard deviation intensity levels on the 0..255 scale.
    Depth sigma(z) = depth_std + depth_std_per_meter_squared * z**2.
    These defaults are experiment settings, not a calibrated sensor model.
    Python callers default to noise OFF; the CLI loads noise_config.json.
    """

    enabled: bool = False
    seed: int | None = 42
    joint_position_std: float | tuple[float, ...] = 0.0003
    joint_velocity_std: float | tuple[float, ...] = 0.001
    joint_position_bias: float | tuple[float, ...] = 0.0
    joint_velocity_bias: float | tuple[float, ...] = 0.0
    gripper_width_std: float = 0.0001
    gripper_width_bias: float = 0.0
    rgb_pixel_std: float = 1.0
    depth_std: float = 0.0002
    depth_std_per_meter_squared: float = 0.0005
    depth_bias: float = 0.0
    depth_dropout_probability: float = 0.001

    def __post_init__(self):
        if not isinstance(self.enabled, bool):
            raise ValueError("enabled must be true or false")
        if self.seed is not None and (isinstance(self.seed, bool)
                                     or not isinstance(self.seed, int) or self.seed < 0):
            raise ValueError("seed must be a non-negative integer or null")
        for name in ("joint_position_std", "joint_velocity_std",
                     "joint_position_bias", "joint_velocity_bias"):
            value = getattr(self, name)
            if isinstance(value, (tuple, list)):
                if len(value) != 6:
                    raise ValueError(f"{name} must have six values")
                for item in value:
                    self._check_number(name, item, nonnegative=name.endswith("std"))
                object.__setattr__(self, name, tuple(float(item) for item in value))
            else:
                self._check_number(name, value, nonnegative=name.endswith("std"))
        for name in ("gripper_width_std", "rgb_pixel_std", "depth_std",
                     "depth_std_per_meter_squared", "depth_dropout_probability"):
            self._check_number(name, getattr(self, name), nonnegative=True)
        for name in ("gripper_width_bias", "depth_bias"):
            self._check_number(name, getattr(self, name))
        if self.depth_dropout_probability > 1:
            raise ValueError("depth_dropout_probability must be between 0 and 1")

    @staticmethod
    def _check_number(name, value, *, nonnegative=False):
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(value) or (nonnegative and value < 0)):
            qualifier = "non-negative " if nonnegative else ""
            raise ValueError(f"{name} must contain finite {qualifier}numbers")

    @classmethod
    def load(cls, path: str | Path = DEFAULT_NOISE_CONFIG_PATH) -> "NoiseConfig":
        with Path(path).open(encoding="utf-8") as source:
            values = json.load(source)
        if not isinstance(values, dict):
            raise ValueError("noise config must be a JSON object")
        unknown = set(values) - {field.name for field in fields(cls)}
        if unknown:
            raise ValueError("unknown noise config settings: " + ", ".join(sorted(unknown)))
        return cls(**values)
