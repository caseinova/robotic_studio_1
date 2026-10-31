"""Controller mode and final approach target, shared by every robot backend."""

from dataclasses import dataclass
import json
import math
from pathlib import Path


DEFAULT_CONTROLLER_CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "controller_config.json"
CONTROLLER_MODES = ("pure_pbvs", "pbvs_mpc")


@dataclass(frozen=True)
class ControllerConfig:
    mode: str = "pbvs_mpc"
    # Final tool flange distance from the tagged face, along its outward normal.
    # The flange is centered over the tag; units are meters.
    tool_clearance: float = 0.18
    # Desired yaw about the cube tag's outward normal, in degrees.
    tool_yaw_deg: float = 90.0
    # Use both tags when visible; encoders provide a fallback for tool-tag loss.
    tool_pose_source: str = "tag_with_fk_fallback"
    # Vision freshness budget in control intervals; hardware checks also apply.
    frame_max_age_intervals: float = 2.0
    # Pickup grasp-center height relative to cube center, along latched cube +Z (meters).
    grasp_height_offset_m: float = 0.0
    pickup_dwell_seconds: float = 0.0
    pickup_dwell_radius_m: float = 0.01
    approach_linear_speed_m_s: float = 0.06
    approach_angular_speed_rad_s: float = 0.3
    pickup_linear_speed_m_s: float = 0.03
    pickup_angular_speed_rad_s: float = 0.3
    pickup_reapproach_attempts: int = 3

    def __post_init__(self):
        if type(self.pickup_reapproach_attempts) is not int or self.pickup_reapproach_attempts < 0:
            raise ValueError("pickup_reapproach_attempts must be a nonnegative integer")
        for name in ("approach_linear_speed_m_s", "approach_angular_speed_rad_s",
                     "pickup_linear_speed_m_s", "pickup_angular_speed_rad_s"):
            value = getattr(self, name)
            if (isinstance(value, bool) or not isinstance(value, (int, float))
                    or not math.isfinite(value) or value <= 0):
                raise ValueError(f"{name} must be finite and positive")
        for name, value, positive in (("pickup_dwell_seconds", self.pickup_dwell_seconds, False),
                                      ("pickup_dwell_radius_m", self.pickup_dwell_radius_m, True)):
            if (isinstance(value, bool) or not isinstance(value, (int, float))
                    or not math.isfinite(value) or value < 0 or (positive and value == 0)):
                raise ValueError(f"{name} must be finite and {'positive' if positive else 'nonnegative'}")
        if (isinstance(self.grasp_height_offset_m, bool)
                or not isinstance(self.grasp_height_offset_m, (int, float))
                or not math.isfinite(self.grasp_height_offset_m)):
            raise ValueError("grasp_height_offset_m must be a finite number")
        if (isinstance(self.frame_max_age_intervals, bool)
                or not isinstance(self.frame_max_age_intervals, (int, float))
                or not math.isfinite(self.frame_max_age_intervals) or self.frame_max_age_intervals <= 0):
            raise ValueError("frame_max_age_intervals must be finite and positive")
        if self.mode not in CONTROLLER_MODES:
            raise ValueError("controller mode must be pure_pbvs or pbvs_mpc")
        if (isinstance(self.tool_clearance, bool)
                or not isinstance(self.tool_clearance, (int, float))
                or not math.isfinite(self.tool_clearance) or self.tool_clearance < 0.16):
            raise ValueError("tool_clearance must be a finite number of at least 0.16 meters")
        if (isinstance(self.tool_yaw_deg, bool)
                or not isinstance(self.tool_yaw_deg, (int, float))
                or not math.isfinite(self.tool_yaw_deg)):
            raise ValueError("tool_yaw_deg must be a finite number")
        if self.tool_pose_source not in ("tag_only", "tag_with_fk_fallback", "fk"):
            raise ValueError("tool_pose_source must be tag_only, tag_with_fk_fallback, or fk")

    @classmethod
    def load(cls, path: str | Path = DEFAULT_CONTROLLER_CONFIG_PATH) -> "ControllerConfig":
        with Path(path).open(encoding="utf-8") as source:
            values = json.load(source)
        if not isinstance(values, dict):
            raise ValueError("controller config must be a JSON object")
        unknown = set(values) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError("unknown controller config settings: " + ", ".join(sorted(unknown)))
        return cls(**values)
