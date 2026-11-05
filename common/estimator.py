"""Relative pose error-state EKF; no simulator truth is read.

Nominal state: cube position, rotation, linear/angular velocity in world axes.
Error order: [dp, dtheta, dv, dw], with R_true = Exp(dtheta) @ R. Tracking in
world coordinates separates measured camera ego-motion from cube motion.
"""

from dataclasses import dataclass, replace
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

DEFAULT_ESTIMATOR_CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "estimator_config.json"

TUNING_FIELDS = (
    "position_measurement_std", "orientation_measurement_std",
    "linear_acceleration_noise", "angular_acceleration_noise",
    "initial_linear_velocity_std", "initial_angular_velocity_std",
    "innovation_gate", "max_prediction_age",
)
TOOL_FUSION_FIELDS = ('fk_position_noise', 'fk_orientation_noise', 'max_position_std',
                      'max_orientation_std', 'max_measurement_skew',
                      'max_position_correction', 'max_rotation_correction')


@dataclass(frozen=True)
class EKFConfig:
    enabled: bool = True
    # Per-tracker overrides; None preserves the legacy shared enabled switch.
    cube_enabled: bool | None = None
    tool_enabled: bool | None = None
    position_measurement_std: float = 0.004  # m, including camera extrinsic uncertainty
    orientation_measurement_std: float = 0.03  # rad
    linear_acceleration_noise: float = 0.05  # continuous density, m/s^(3/2)
    angular_acceleration_noise: float = 0.2  # rad/s^(3/2)
    initial_linear_velocity_std: float = 0.1  # m/s
    initial_angular_velocity_std: float = 0.5  # rad/s
    innovation_gate: float = 25.0  # squared Mahalanobis distance
    max_prediction_age: float = 0.3  # seconds since last accepted measurement
    prediction_model: str = 'constant_velocity'
    fk_position_noise: float = 0.002
    fk_orientation_noise: float = 0.02
    max_position_std: float = 0.2
    max_orientation_std: float = 0.3
    max_measurement_skew: float = 0.05
    max_position_correction: float = 0.003
    max_rotation_correction: float = 0.02
    # Target-specific tuning overrides legacy flat fields, which remain defaults.
    cube: dict | None = None
    tool: dict | None = None

    def __post_init__(self):
        if not isinstance(self.enabled, bool):
            raise ValueError("EKF enabled must be boolean")
        for name in ("cube_enabled", "tool_enabled"):
            if getattr(self, name) is not None and not isinstance(getattr(self, name), bool):
                raise ValueError(f"EKF {name} must be boolean or null")
        if self.prediction_model not in ('constant_velocity', 'fk_increment'):
            raise ValueError('invalid prediction_model')
        for name in TUNING_FIELDS + TOOL_FUSION_FIELDS:
            value = getattr(self, name)
            if (isinstance(value, bool) or not isinstance(value, (int, float))
                    or not np.isfinite(value) or value <= 0):
                raise ValueError(f"{name} must be a finite positive number")
        for target in ("cube", "tool"):
            tuning = getattr(self, target)
            if tuning is None:
                continue
            if not isinstance(tuning, dict):
                raise ValueError(f"EKF {target} tuning must be a JSON object")
            allowed = TUNING_FIELDS + (TOOL_FUSION_FIELDS + ('prediction_model',) if target == 'tool' else ())
            unknown = set(tuning) - set(allowed)
            if unknown:
                raise ValueError(f"unknown {target} EKF settings: " + ", ".join(sorted(map(str, unknown))))
            for name, value in tuning.items():
                if name == 'prediction_model':
                    if value not in ('constant_velocity', 'fk_increment'):
                        raise ValueError('invalid tool.prediction_model')
                    continue
                if (isinstance(value, bool) or not isinstance(value, (int, float))
                        or not np.isfinite(value) or value <= 0):
                    raise ValueError(f"{target}.{name} must be a finite positive number")
            object.__setattr__(self, target, dict(tuning))

    def for_target(self, target):
        if target not in ("cube", "tool"):
            raise ValueError("EKF target must be cube or tool")
        override = getattr(self, target + "_enabled")
        tuning = dict(getattr(self, target) or {})
        if target == 'cube':
            tuning['prediction_model'] = 'constant_velocity'
        return replace(self, enabled=self.enabled if override is None else override,
                       cube_enabled=None, tool_enabled=None, cube=None, tool=None,
                       **tuning)

    @classmethod
    def load(cls, path=DEFAULT_ESTIMATOR_CONFIG_PATH):
        with Path(path).open(encoding="utf-8") as source:
            values = json.load(source)
        if not isinstance(values, dict):
            raise ValueError("estimator config must be a JSON object")
        unknown = set(values) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError("unknown estimator settings: " + ", ".join(sorted(unknown)))
        return cls(**values)


@dataclass(frozen=True)
class RelativePoseEstimate:
    """Owned arrays. A_from_B maps B into A; distances are meters.

    cube_from_camera is the camera pose relative to the cube; camera_from_cube
    is its inverse for PBVS. Velocities/covariance are in world axes, with
    covariance ordered [dp, dtheta, dv, dw]. Camera extrinsics are supplied,
    not independently estimated.
    """
    timestamp: float
    measurement_timestamp: float
    cube_from_camera: np.ndarray
    camera_from_cube: np.ndarray
    world_from_cube: np.ndarray
    linear_velocity: np.ndarray
    angular_velocity: np.ndarray
    covariance: np.ndarray
    measurement_accepted: bool
    status: str


def _skew(vector):
    x, y, z = vector
    return np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])


def _left_jacobian(vector):
    """SO(3) left Jacobian with stable small-angle expansion."""
    angle = np.linalg.norm(vector)
    cross = _skew(vector)
    if angle < 1e-5:
        return np.eye(3) + 0.5 * cross + cross @ cross / 6
    return (np.eye(3) + (1 - np.cos(angle)) / angle**2 * cross
            + (angle - np.sin(angle)) / angle**3 * cross @ cross)


def _pose(value, name):
    pose = np.asarray(value, dtype=float)
    if (pose.shape != (4, 4) or not np.all(np.isfinite(pose))
            or not np.allclose(pose[3], [0, 0, 0, 1], atol=1e-8)
            or not np.allclose(pose[:3, :3].T @ pose[:3, :3], np.eye(3), atol=1e-6)
            or not np.isclose(np.linalg.det(pose[:3, :3]), 1, atol=1e-6)):
        raise ValueError(f"{name} must be a finite rigid 4x4 transform")
    return pose


def _inverse(pose):
    result = np.eye(4)
    result[:3, :3] = pose[:3, :3].T
    result[:3, 3] = -result[:3, :3] @ pose[:3, 3]
    return result


class RelativePoseEKF:
    """Constant-velocity cube model, corrected by camera pose measurements.

    update() takes camera_from_cube and measured world_from_camera at the same
    timestamp. predict() bridges brief tag loss; returns None before initial
    detection or after max_prediction_age. Reacquisition after expiry starts
    a new track. Timestamps must be monotonic; reset() after a clock restart.
    Duplicate camera timestamps never apply a measurement twice.
    """

    def __init__(self, config: EKFConfig | None = None):
        self.config = config or EKFConfig()
        self.reset()

    def reset(self):
        self.timestamp = None
        self.measurement_timestamp = None
        self._last_camera_timestamp = None
        self._position = np.zeros(3)
        self._rotation = np.eye(3)
        self._velocity = np.zeros(3)
        self._angular_velocity = np.zeros(3)
        self._covariance = np.zeros((12, 12))
        self.last_status = "uninitialized"
        self.innovation_distance = None

    def step(self, frame, observation=None) -> RelativePoseEstimate | None:
        """Consume an interface RGBDFrame and CameraProcessor observation."""
        if observation is None:
            return self.predict(frame.timestamp, frame.world_from_camera)
        if abs(observation.timestamp - frame.timestamp) > 1e-9:
            raise ValueError("observation and frame timestamps must match")
        return self.update(frame.timestamp, observation.camera_from_cube, frame.world_from_camera)

    def update(self, timestamp, camera_from_cube, world_from_camera) -> RelativePoseEstimate:
        camera = _pose(world_from_camera, "world_from_camera")
        measured = camera @ _pose(camera_from_cube, "camera_from_cube")
        self._check_time(timestamp)
        if self._last_camera_timestamp == timestamp:
            self.last_status = "duplicate_measurement"
            return self._estimate(camera, False)
        expired = (self.measurement_timestamp is not None
                   and timestamp - self.measurement_timestamp > self.config.max_prediction_age)
        if self.timestamp is None or expired:
            self._initialize(timestamp, measured)
            self.last_status = "reinitialized" if expired else "initialized"
            return self._estimate(camera, True)
        self._propagate(timestamp)
        self._last_camera_timestamp = float(timestamp)
        residual = np.concatenate((
            measured[:3, 3] - self._position,
            Rotation.from_matrix(measured[:3, :3] @ self._rotation.T).as_rotvec(),
        ))
        h = np.zeros((6, 12))
        h[:, :6] = np.eye(6)
        measurement_covariance = self._measurement_covariance()
        innovation_covariance = h @ self._covariance @ h.T + measurement_covariance
        self.innovation_distance = float(residual @ np.linalg.solve(innovation_covariance, residual))
        if self.innovation_distance > self.config.innovation_gate:
            self.last_status = "measurement_rejected"
            return self._estimate(camera, False)
        gain = np.linalg.solve(innovation_covariance, h @ self._covariance).T
        correction = gain @ residual
        self._position += correction[:3]
        self._rotation = Rotation.from_rotvec(correction[3:6]).as_matrix() @ self._rotation
        self._velocity += correction[6:9]
        self._angular_velocity += correction[9:12]
        # Joseph covariance update, then reset orientation error coordinates.
        remaining = np.eye(12) - gain @ h
        covariance = (remaining @ self._covariance @ remaining.T
                      + gain @ measurement_covariance @ gain.T)
        reset = np.eye(12)
        reset[3:6, 3:6] = _left_jacobian(correction[3:6])
        self._covariance = reset @ covariance @ reset.T
        self._covariance = (self._covariance + self._covariance.T) / 2
        self.measurement_timestamp = float(timestamp)
        self.last_status = "updated"
        return self._estimate(camera, True)

    def predict(self, timestamp, world_from_camera) -> RelativePoseEstimate | None:
        camera = _pose(world_from_camera, "world_from_camera")
        self._check_time(timestamp)
        if self.timestamp is None:
            return None
        if timestamp - self.measurement_timestamp > self.config.max_prediction_age:
            self.timestamp = float(timestamp)
            self.last_status = "expired"
            return None
        self._propagate(timestamp)
        self.last_status = "predicted"
        return self._estimate(camera, False)

    def _check_time(self, timestamp):
        if isinstance(timestamp, bool) or not np.isfinite(timestamp):
            raise ValueError("timestamp must be finite")
        if self.timestamp is not None and timestamp < self.timestamp:
            raise ValueError("timestamps must be monotonic; reset after a clock restart")

    def _measurement_covariance(self):
        c = self.config
        return np.diag([c.position_measurement_std**2] * 3
                       + [c.orientation_measurement_std**2] * 3)

    def _initialize(self, timestamp, measured):
        self._position = measured[:3, 3].copy()
        self._rotation = measured[:3, :3].copy()
        self._velocity[:] = 0
        self._angular_velocity[:] = 0
        c = self.config
        self._covariance = np.diag(
            [c.position_measurement_std**2] * 3 + [c.orientation_measurement_std**2] * 3
            + [c.initial_linear_velocity_std**2] * 3 + [c.initial_angular_velocity_std**2] * 3
        )
        self.timestamp = self.measurement_timestamp = self._last_camera_timestamp = float(timestamp)
        self.innovation_distance = None

    def _propagate(self, timestamp):
        dt = timestamp - self.timestamp
        if dt == 0:
            return
        # Keep angular linearization local during gaps in camera detections.
        steps = max(1, int(np.ceil(dt / 0.05)))
        step = dt / steps
        for _ in range(steps):
            angle = self._angular_velocity * step
            rotation_step = Rotation.from_rotvec(angle).as_matrix()
            transition = np.eye(12)
            transition[:3, 6:9] = np.eye(3) * step
            transition[3:6, 3:6] = rotation_step
            transition[3:6, 9:12] = _left_jacobian(angle) * step
            process_covariance = np.zeros((12, 12))
            for pose_start, velocity_start, noise in (
                (0, 6, self.config.linear_acceleration_noise),
                (3, 9, self.config.angular_acceleration_noise),
            ):
                a, b = slice(pose_start, pose_start + 3), slice(velocity_start, velocity_start + 3)
                process_covariance[a, a] = np.eye(3) * noise**2 * step**3 / 3
                process_covariance[a, b] = process_covariance[b, a] = np.eye(3) * noise**2 * step**2 / 2
                process_covariance[b, b] = np.eye(3) * noise**2 * step
            self._position += self._velocity * step
            self._rotation = rotation_step @ self._rotation
            self._covariance = transition @ self._covariance @ transition.T + process_covariance
        self._covariance = (self._covariance + self._covariance.T) / 2
        self.timestamp = float(timestamp)

    def _estimate(self, camera, accepted):
        if self.timestamp - self.measurement_timestamp > self.config.max_prediction_age:
            self.last_status = "expired"
            return None
        world_from_cube = np.eye(4)
        world_from_cube[:3, :3] = self._rotation
        world_from_cube[:3, 3] = self._position
        camera_from_cube = _inverse(camera) @ world_from_cube
        return RelativePoseEstimate(
            self.timestamp, self.measurement_timestamp,
            _inverse(camera_from_cube), camera_from_cube, world_from_cube,
            self._velocity.copy(), self._angular_velocity.copy(),
            self._covariance.copy(), accepted, self.last_status,
        )
