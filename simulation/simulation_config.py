"""Experiment settings consumed only by simulate.py, never by hardware APIs."""

from dataclasses import dataclass, fields, field
import json
import math
from pathlib import Path
from simulation.viewer import viewer_settings


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "simulation_config.json"


@dataclass(frozen=True)
class SimulationConfig:
    """Cube center in world meters and orientation (w,x,y,z)."""

    cube_position: tuple[float, float, float] = (-0.074, 0.32, 0.032)
    cube_quaternion: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    # Horizontal destination tags on the floor, keyed by sequence destination name.
    destination_positions: dict = field(default_factory=lambda: {'drop_zone_b': (.12, .32, 0.)})
    viewer: dict = field(default_factory=dict)
    pickup_collision_clearance_m: float = 0.015
    approach_collision_clearance_m: float = 0.015
    placement_collision_clearance_m: float = 0.015
    mpc_clearance_buffer_m: float = 0.003

    def __post_init__(self):
        for name in ('pickup_collision_clearance_m', 'approach_collision_clearance_m',
                     'placement_collision_clearance_m', 'mpc_clearance_buffer_m'):
            value = getattr(self, name)
            if (isinstance(value, bool) or not isinstance(value, (int, float))
                    or not math.isfinite(value) or value < 0
                    or (value == 0 and name != 'mpc_clearance_buffer_m')):
                raise ValueError(f'{name} must be finite and positive (buffer may be zero)')
        object.__setattr__(self, 'viewer', viewer_settings(self.viewer))
        if not isinstance(self.destination_positions, dict):
            raise ValueError('destination_positions must be a name-to-XYZ object')
        positions = {}
        for name, xyz in self.destination_positions.items():
            if (not isinstance(name, str) or not name or not isinstance(xyz, (list, tuple))
                    or len(xyz) != 3 or any(isinstance(v, bool) or not isinstance(v, (int, float))
                                           or not math.isfinite(v) for v in xyz)
                    or abs(xyz[2]) > 1e-9):
                raise ValueError('simulation destination positions must be finite XYZ on the floor (z=0)')
            positions[name] = tuple(float(v) for v in xyz)
        object.__setattr__(self, 'destination_positions', positions)
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

    def build_mpc_config(self):
        from common.controller import MPCConfig
        return MPCConfig(clearance=self.approach_collision_clearance_m,
                         clearance_buffer=self.mpc_clearance_buffer_m)

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
