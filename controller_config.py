"""Controller selection, separate from simulation scene experiment settings."""

from dataclasses import dataclass
import json
from pathlib import Path


DEFAULT_CONTROLLER_CONFIG_PATH = Path(__file__).resolve().with_name("controller_config.json")
CONTROLLER_MODES = ("pure_pbvs", "pbvs_mpc")


@dataclass(frozen=True)
class ControllerConfig:
    mode: str = "pbvs_mpc"

    def __post_init__(self):
        if self.mode not in CONTROLLER_MODES:
            raise ValueError("controller mode must be pure_pbvs or pbvs_mpc")

    @classmethod
    def load(cls, path: str | Path = DEFAULT_CONTROLLER_CONFIG_PATH) -> "ControllerConfig":
        with Path(path).open(encoding="utf-8") as source:
            values = json.load(source)
        if not isinstance(values, dict):
            raise ValueError("controller config must be a JSON object")
        unknown = set(values) - {"mode"}
        if unknown:
            raise ValueError("unknown controller config settings: " + ", ".join(sorted(unknown)))
        return cls(**values)
