"""Named destination tags and bounded pick/place session settings."""

from dataclasses import dataclass, field
import json
from pathlib import Path

import numpy as np

from common.geometry import _pose, _positive, PlaneObstacle

DEFAULT_SEQUENCE_CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "sequence_config.json"


@dataclass(frozen=True)
class Destination:
    marker_id: int
    marker_size_m: float
    # Desired cube CENTER pose relative to the destination marker, not the tool.
    marker_from_cube: list
    # Named collision-map table; placement height comes from the detected tag.
    collision_support_plane: str
    approach_height_m: float = .10
    retreat_height_m: float = .10
    max_drop_height_m: float = .003

    def __post_init__(self):
        if type(self.marker_id) is not int or not 0 <= self.marker_id < 50:
            raise ValueError("destination marker_id must be in DICT_4X4_50")
        for name in ("marker_size_m", "approach_height_m", "retreat_height_m", "max_drop_height_m"):
            _positive(getattr(self, name), name)
        object.__setattr__(self, "marker_from_cube", _pose(self.marker_from_cube, "marker_from_cube"))
        if not isinstance(self.collision_support_plane, str) or not self.collision_support_plane:
            raise ValueError("destination requires a named collision_support_plane")


@dataclass(frozen=True)
class SequenceConfig:
    destinations: dict = field(default_factory=dict)
    idle_timeout_s: float = 10.
    action_timeout_s: float = 60.
    phase_timeout_s: float = 20.
    transfer_timeout_s: float | None = None
    cartesian_speed_m_s: float = .02
    angular_speed_rad_s: float = .2

    def __post_init__(self):
        if self.transfer_timeout_s is not None:
            _positive(self.transfer_timeout_s, "transfer_timeout_s")
        for name in ("idle_timeout_s", "action_timeout_s", "phase_timeout_s",
                     "cartesian_speed_m_s", "angular_speed_rad_s"):
            _positive(getattr(self, name), name)
        if not isinstance(self.destinations, dict):
            raise ValueError("destinations must be a name-to-settings object")
        destinations = {}
        for name, values in self.destinations.items():
            if not isinstance(name, str) or not name:
                raise ValueError("destination names must be nonempty strings")
            destinations[name] = values if isinstance(values, Destination) else Destination(**values)
        ids = [d.marker_id for d in destinations.values()]
        if len(ids) != len(set(ids)):
            raise ValueError("destination marker IDs must be unique")
        object.__setattr__(self, "destinations", destinations)

    @classmethod
    def load(cls, path=DEFAULT_SEQUENCE_CONFIG_PATH):
        return cls(**json.loads(Path(path).read_text()))

    def validate_destination(self, name, hardware, kinematics):
        if name not in self.destinations:
            raise ValueError(f"unknown destination: {name}")
        destination = self.destinations[name]
        reserved = {int(i) for i in hardware.perception["cube_from_markers"]}
        reserved.add(hardware.perception["tool_marker_id"])
        if destination.marker_id in reserved:
            raise ValueError("destination ID overlaps a cube/tool marker")
        fingers = [link for link in kinematics.links if link.frame == "tool" and link.contact_group == "finger"]
        body = [link for link in kinematics.links if link.frame == "tool" and link.contact_group != "finger"]
        if len(fingers) < 2 or not body:
            raise ValueError("placement requires measured tool-body and two finger collision capsules")
        support = next((o for o in kinematics.obstacles if o.name == destination.collision_support_plane), None)
        if not isinstance(support, PlaneObstacle) or not np.allclose(support.normal, [0, 0, 1]):
            raise ValueError("placement requires a configured horizontal support plane (+world Z)")
        if set(support.exclude_links) & {link.name for link in (*fingers, *body)}:
            raise ValueError("support plane must not exclude tool/finger collision geometry")
        return destination
