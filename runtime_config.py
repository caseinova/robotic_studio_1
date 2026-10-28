"""Backend selection and paths shared by simulation and hardware launchers."""

from dataclasses import dataclass, replace
from pathlib import Path
import json

from hardware_config import object_settings

DEFAULT_RUNTIME_CONFIG_PATH = Path(__file__).parent / "config" / "runtime_config.json"


@dataclass(frozen=True)
class RuntimeConfig:
    backend: str
    task: str
    hardware_config: Path
    simulation_config: Path
    controller_config: Path
    estimator_config: Path
    noise_config: Path
    headless: bool
    camera: bool
    steps: int

    def __post_init__(self):
        if self.backend not in ("simulation", "hardware"):
            raise ValueError("backend must be simulation or hardware")
        if self.task not in ("observe", "servo", "pick"):
            raise ValueError("task must be observe, servo, or pick")
        if type(self.steps) is not int or self.steps < 0:
            raise ValueError("steps must be a non-negative integer")
        if type(self.headless) is not bool or type(self.camera) is not bool:
            raise ValueError("headless/camera settings must be boolean")

    @classmethod
    def load(cls, path=DEFAULT_RUNTIME_CONFIG_PATH):
        path = Path(path).resolve()
        with path.open(encoding="utf-8") as source:
            values = json.load(source)
        object_settings(values, cls.__dataclass_fields__, "runtime")
        if set(values) != set(cls.__dataclass_fields__):
            raise ValueError("runtime config must specify all fields")
        for name in ("hardware_config", "simulation_config", "controller_config", "estimator_config", "noise_config"):
            if not isinstance(values[name], str) or not values[name]:
                raise ValueError(f"{name} must be a file path")
            values[name] = (path.parent / values[name]).resolve()
        return cls(**values)

    def override(self, **values):
        return replace(self, **{key: value for key, value in values.items() if value is not None})
