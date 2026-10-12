"""Backend-independent measurements exchanged through RobotInterface."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RobotState:
    """One measured robot snapshot, using the interface's monotonic clock.

    Joint order: shoulder pan, shoulder lift, elbow, wrist 1, wrist 2, wrist 3.
    Positions are radians, velocities radians/second, and inner gripper width
    meters. Adapters return independent arrays so commands cannot alter them.
    """

    timestamp: float
    joint_positions: np.ndarray
    joint_velocities: np.ndarray
    gripper_width: float


@dataclass(frozen=True)
class RGBDFrame:
    """Registered, rectified RGB and metric depth from one acquisition.

    rgb: uint8 (H, W, 3), RGB channel order.
    depth: float32 (H, W), optical-axis Z in meters, not ray distance.
    intrinsics: 3x3 pinhole K; integer (column, row) coordinates are pixel centers.
    world_from_camera: 4x4 optical-to-world transform, calibrated from measured
        robot state. Optical axes are +X right, +Y down, +Z forward.
    timestamp: acquisition time in seconds on the same clock as RobotState.
    depth_range: near/far valid distances in meters; background is at far.

    A hardware adapter must register depth to RGB and rectify lens distortion.
    Arrays belong to the returned frame and remain valid after another capture.
    """

    rgb: np.ndarray
    depth: np.ndarray
    intrinsics: np.ndarray
    world_from_camera: np.ndarray
    timestamp: float
    depth_range: tuple[float, float]

    @property
    def valid_depth(self) -> np.ndarray:
        """Mask out background and clipped or nonfinite depth pixels."""
        near, far = self.depth_range
        return (
            np.isfinite(self.depth)
            & (self.depth > near)
            & (self.depth < far * (1 - 1e-5))
        )
