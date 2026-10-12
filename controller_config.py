"""Controller mode and final approach target, shared by every robot backend."""

from dataclasses import dataclass
import json
import math
from pathlib import Path


DEFAULT_CONTROLLER_CONFIG_PATH = Path(__file__).resolve().with_name("controller_config.json")
CONTROLLER_MODES = ("pure_pbvs", "pbvs_mpc")


@dataclass(frozen=True)
class ControllerConfig:
    mode: str = "pbvs_mpc"
    # Final tool flange distance from the tagged face, along its outward normal.
    # The flange is centered over the tag; units are meters.
    tool_clearance: float = 0.18

    def __post_init__(self):
        if self.mode not in CONTROLLER_MODES:
            raise ValueError("controller mode must be pure_pbvs or pbvs_mpc")
        if (isinstance(self.tool_clearance, bool)
                or not isinstance(self.tool_clearance, (int, float))
                or not math.isfinite(self.tool_clearance) or self.tool_clearance < 0.16):
            raise ValueError("tool_clearance must be a finite number of at least 0.16 meters")

    @classmethod
    def load(cls, path: str | Path = DEFAULT_CONTROLLER_CONFIG_PATH) -> "ControllerConfig":
        with Path(path).open(encoding="utf-8") as source:
            values = json.load(source)
        if not isinstance(values, dict):
            raise ValueError("controller config must be a JSON object")
        unknown = set(values) - {"mode", "tool_clearance"}
        if unknown:
            raise ValueError("unknown controller config settings: " + ", ".join(sorted(unknown)))
        return cls(**values)
