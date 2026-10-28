"""Validated hardware installation settings; loading never opens a device."""

from dataclasses import dataclass
from pathlib import Path
import copy
import json

import numpy as np

from hardware_kinematics import (
    UR3DH, UR3Kinematics, LinkCapsule, SphereObstacle, BoxObstacle, PlaneObstacle, _pose, _vector,
)

DEFAULT_HARDWARE_CONFIG_PATH = Path(__file__).with_name("hardware_config.json")


def object_settings(value, allowed, name):
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an object")
    unknown = set(value) - set(allowed)
    if unknown:
        raise ValueError(f"unknown {name} settings: {', '.join(sorted(unknown))}")
    return value


def positive(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return value


@dataclass(frozen=True)
class HardwareConfig:
    robot: dict
    camera: dict
    gripper: dict
    calibration: dict
    perception: dict
    collision: dict
    control: dict
    mpc: dict

    @classmethod
    def load(cls, path=DEFAULT_HARDWARE_CONFIG_PATH):
        with Path(path).open(encoding="utf-8") as source:
            values = json.load(source)
        object_settings(values, cls.__dataclass_fields__, "hardware")
        missing = set(cls.__dataclass_fields__) - set(values)
        if missing:
            raise ValueError("missing hardware sections: " + ", ".join(sorted(missing)))
        result = cls(**copy.deepcopy(values))
        if result.robot.get("api_path") is not None:
            if not isinstance(result.robot["api_path"], str) or not result.robot["api_path"].strip():
                raise ValueError("robot.api_path must be a repository path")
            api_path = Path(result.robot["api_path"])
            result.robot["api_path"] = str((Path(path).resolve().parent / api_path).resolve())
        result.validate_schema()
        return result

    def validate_schema(self):
        sections = {
            "robot": "driver factory options host frequency_hz dh joint_lower joint_upper api_path",
            "camera": "driver factory options width height intrinsics distortion depth_scale depth_range",
            "gripper": "driver factory options model host port unit_id force_n max_width_m timeout_s use_device_fingertip_offset",
            "calibration": "world_from_base world_from_camera flange_from_tool tool_from_grasp tool_from_marker",
            "perception": "cube_size marker_size tool_marker_id tool_marker_size reference_marker_id cube_from_markers",
            "collision": "links obstacles excluded_self_pairs",
            "control": "dt velocity_limit acceleration_limit command_timeout state_max_age frame_max_age gripper_max_age max_cycle_time max_run_seconds",
            "mpc": "horizon clearance clearance_buffer tracking_weight smoothing_weight max_iterations",
        }
        for name, allowed in sections.items():
            object_settings(getattr(self, name), allowed.split(), name)
        if self.camera.get("factory") == "hardware_realsense:create_camera":
            from hardware_realsense import validate_settings

            validate_settings(self.camera)
        for name in ("robot", "camera", "gripper"):
            settings = getattr(self, name)
            driver = settings.get("driver")
            supported = {"robot": ("ur_rtde", "ur3_rg2_rosbridge", "custom"), "camera": ("unconfigured", "custom"),
                         "gripper": ("onrobot_rg", "custom", "disabled")}[name]
            if driver not in supported:
                raise ValueError(f"unsupported {name}.driver: {driver!r}")
            if not isinstance(settings.get("options", {}), dict):
                raise ValueError(f"{name}.options must be an object")
            if driver != "custom" and settings.get("options"):
                raise ValueError(f"{name}.options are only used by custom adapters")
        for name in sections["control"].split():
            positive(self.control.get(name), f"control.{name}")
        if self.control["command_timeout"] <= self.control["dt"]:
            raise ValueError("command_timeout must exceed the control interval")
        if self.control["max_cycle_time"] > self.control["command_timeout"]:
            raise ValueError("max_cycle_time must not exceed command_timeout")
        # The existing controller also enforces freshness within two intervals.
        if max(self.control["state_max_age"], self.control["frame_max_age"]) > 2 * self.control["dt"]:
            raise ValueError("state/frame max age must not exceed twice control.dt")
        if self.robot["driver"] in ("ur_rtde", "ur3_rg2_rosbridge"):
            frequency = positive(self.robot.get("frequency_hz"), "robot.frequency_hz")
            if frequency > 125:
                raise ValueError("configured UR worker frequency must not exceed 125 Hz")
            if frequency < 1 / self.control["dt"]:
                raise ValueError("robot worker frequency must be at least the controller update rate")
        if self.robot["driver"] == "ur3_rg2_rosbridge":
            from hardware_rosbridge import DEFAULT_API_PATH

            if not isinstance(self.robot.get("dh"), dict) or set(self.robot["dh"]) != {"a", "d", "alpha", "offsets"}:
                raise ValueError("UR3e API requires explicit robot.dh arrays; CB-series defaults are not substituted")
            path = self.robot.get("api_path", str(DEFAULT_API_PATH))
            if not isinstance(path, str) or not path.strip():
                raise ValueError("robot.api_path must be a repository path")
            if not (Path(path) / "robot" / "real_ur_backend.py").is_file():
                raise ValueError("robot.api_path must contain robot/real_ur_backend.py")

    def validate_ready(self, controller_config, *, task="servo"):
        """Reject placeholders and inconsistent calibration before connecting."""
        self.validate_schema()
        missing = []
        if self.gripper["driver"] == "disabled":
            if task not in ("servo", "observe"):
                raise ValueError("disabled gripper is only supported for servo/observe, not pickup")
            if any(np.any(link.get("opening_axis", [0, 0, 0])) for link in self.collision.get("links", [])):
                raise ValueError("disabled gripper cannot use width-dependent collision geometry")
        for name in ("robot", "camera", "gripper"):
            settings = getattr(self, name)
            if name == "gripper" and settings["driver"] == "disabled":
                continue
            if settings["driver"] == "unconfigured":
                missing.append(f"{name}.driver/factory")
            elif settings["driver"] == "custom":
                factory = settings.get("factory")
                parts = factory.split(":") if isinstance(factory, str) else []
                if (len(parts) != 2 or not all(p.isidentifier() for p in parts[0].split("."))
                        or not parts[1].isidentifier()):
                    missing.append(f"{name}.factory (module:factory)")
            elif not isinstance(settings.get("host"), str) or not settings["host"].strip():
                missing.append(f"{name}.host")
        for key in ("world_from_base", "world_from_camera", "flange_from_tool", "tool_from_grasp"):
            if self.calibration.get(key) is None:
                missing.append(f"calibration.{key}")
        if controller_config.tool_pose_source != "fk" and self.calibration.get("tool_from_marker") is None:
            missing.append("calibration.tool_from_marker")
        for key in ("joint_lower", "joint_upper"):
            if self.robot.get(key) is None:
                missing.append(f"robot.{key}")
        if self.camera.get("intrinsics") is None:
            missing.append("camera.intrinsics")
        if not self.perception.get("cube_from_markers"):
            missing.append("perception.cube_from_markers")
        if missing:
            raise ValueError("hardware configuration is incomplete:\n  " + "\n  ".join(missing))
        for key, value in self.calibration.items():
            if value is not None:
                _pose(value, f"calibration.{key}")
        for name in ("width", "height"):
            if type(self.camera.get(name)) is not int or self.camera[name] <= 0:
                raise ValueError(f"camera.{name} must be a positive integer")
        k = np.asarray(self.camera["intrinsics"], dtype=float)
        if (k.shape != (3, 3) or not np.all(np.isfinite(k)) or k[0, 0] <= 0 or k[1, 1] <= 0
                or not np.allclose(k[2], [0, 0, 1]) or not np.allclose([k[0, 1], k[1, 0]], 0)):
            raise ValueError("camera.intrinsics must be a finite pinhole K with zero skew")
        d = np.asarray(self.camera.get("distortion"), dtype=float)
        if d.shape not in ((4,), (5,), (8,), (12,), (14,)) or not np.all(np.isfinite(d)):
            raise ValueError("camera.distortion must contain 4, 5, 8, 12, or 14 OpenCV coefficients")
        positive(self.camera.get("depth_scale"), "camera.depth_scale")
        depth_range = _vector(self.camera.get("depth_range"), 2, "camera.depth_range")
        if not 0 < depth_range[0] < depth_range[1]:
            raise ValueError("camera.depth_range must be positive near/far meters")
        positive(self.gripper.get("max_width_m"), "gripper.max_width_m")
        if self.gripper["driver"] == "onrobot_rg":
            model = self.gripper.get("model")
            if model not in ("RG2", "RG6"):
                raise ValueError("onrobot_rg supports RG2/RG6; select a custom adapter for other models")
            if not 0 < positive(self.gripper.get("force_n"), "gripper.force_n") <= (40 if model == "RG2" else 120):
                raise ValueError("gripper force exceeds the selected model's range")
            if self.gripper["max_width_m"] > (.11 if model == "RG2" else .16):
                raise ValueError("gripper max_width_m exceeds the selected model's nominal stroke")
            for key, maximum in (("port", 65535), ("unit_id", 247)):
                if type(self.gripper.get(key)) is not int or not 1 <= self.gripper[key] <= maximum:
                    raise ValueError(f"gripper.{key} is out of range")
            positive(self.gripper.get("timeout_s"), "gripper.timeout_s")
            if type(self.gripper.get("use_device_fingertip_offset")) is not bool:
                raise ValueError("use_device_fingertip_offset must be boolean")
        kinematics = self.build_kinematics()
        if task == "pick" or (task != "observe" and controller_config.mode == "pbvs_mpc"):
            kinematics.distances(np.zeros(6))  # Reject absent collision geometry.
        processor = self.build_processor()  # Validate tag transforms and IDs without a camera.
        if self.perception["tool_marker_id"] in processor.cube_from_markers:
            raise ValueError("tool and cube marker IDs must be distinct")
        self.build_mpc_config()

    def build_kinematics(self):
        links = [LinkCapsule(**item) for item in self.collision.get("links", [])]
        shapes = {"sphere": SphereObstacle, "box": BoxObstacle, "plane": PlaneObstacle}
        obstacles = []
        for item in self.collision.get("obstacles", []):
            settings = dict(item)
            kind = settings.pop("type", None)
            if kind not in shapes:
                raise ValueError(f"unsupported obstacle type: {kind!r}")
            obstacles.append(shapes[kind](**settings))
        return UR3Kinematics(joint_limits=(self.robot["joint_lower"], self.robot["joint_upper"]),
            **{k: v for k, v in self.calibration.items() if k != "world_from_camera"},
            dh=UR3DH(**self.robot.get("dh", {})), link_capsules=links, obstacles=obstacles,
            excluded_self_pairs=self.collision.get("excluded_self_pairs", []),
            marker_id=self.perception["tool_marker_id"], marker_size=self.perception["tool_marker_size"])

    def build_processor(self):
        from camera_processing import TaggedCubeProcessor

        return TaggedCubeProcessor(cube_size=self.perception["cube_size"], marker_size=self.perception["marker_size"],
            cube_from_markers=self.perception["cube_from_markers"],
            reference_marker_id=self.perception["reference_marker_id"])

    def build_mpc_config(self):
        from controller import MPCConfig

        return MPCConfig(**self.mpc, dt=self.control["dt"], velocity_limit=self.control["velocity_limit"],
                         acceleration_limit=self.control["acceleration_limit"])
