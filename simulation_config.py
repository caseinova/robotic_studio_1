"""Experiment settings consumed only by simulate.py, never by hardware APIs."""

from dataclasses import dataclass, fields
import json
import math
from pathlib import Path


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "config" / "simulation_config.json"


@dataclass(frozen=True)
class SimulationConfig:
    """Cube center in world meters and orientation (w,x,y,z)."""

    cube_position: tuple[float, float, float] = (-0.074, 0.32, 0.032)
    cube_quaternion: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)

    def __post_init__(self):
        for name, length in (("cube_position", 3), ("cube_quaternion", 4)):
            values = getattr(self, name)
            if (not isinstance(values, (list, tuple)) or len(values) != length
                    or any(isinstance(v, bool) or not isinstance(v, (int, float))
                           or not math.isfinite(v) for v in values)):
                raise ValueError(f"{name} must contain {length} finite numbers")
            object.__setattr__(self, name, tuple(float(v) for v in values))
        norm = math.sqrt(sum(v * v for v in self.cube_quaternion))
        if norm < 1e-12 or not math.isfinite(norm):
            raise ValueError("cube_quaternion must be a nonzero finite quaternion")
        object.__setattr__(self, "cube_quaternion", tuple(v / norm for v in self.cube_quaternion))

    @classmethod
    def load(cls, path: str | Path = DEFAULT_CONFIG_PATH) -> "SimulationConfig":
        with Path(path).open(encoding="utf-8") as source:
            values = json.load(source)
        if not isinstance(values, dict):
            raise ValueError("simulation config must be a JSON object")
        unknown = set(values) - {field.name for field in fields(cls)}
        if unknown:
            raise ValueError("unknown simulation config settings: " + ", ".join(sorted(unknown)))
        return cls(**values)
