"""Robot measurements, camera acquisition, and commands for every backend."""

from abc import ABC, abstractmethod

import numpy as np

from common.robot_state import RGBDFrame, RobotState



class RobotInterface(ABC):
    """API used by upper-level controllers regardless of the robot backend.

    Joint order: shoulder pan, shoulder lift, elbow, wrist 1, wrist 2, wrist 3.
    Angles use radians and velocities use radians/second. Hardware adapters
    should implement this contract using their chosen robot transport.
    State and camera timestamps must share a monotonic clock in seconds.
    MuJoCo uses simulation time; hardware can use a monotonic host clock with
    camera/device timestamps mapped onto it. Camera poses use the model's
    world frame and calibrated camera extrinsics. Geometry calculations live
    behind kinematics.Kinematics (UR3Kinematics for hardware); this class owns
    sensor/command communication, not a second robot-state prediction.
    """

    @abstractmethod
    def read_state(self) -> RobotState:
        """Read one coherent snapshot of arm and gripper measurements."""

    @abstractmethod
    def capture_rgbd(self) -> RGBDFrame:
        """Acquire aligned RGB/depth, calibration, pose, and acquisition time."""

    @property
    def joint_positions(self) -> np.ndarray:
        return self.read_state().joint_positions

    @property
    def joint_velocities(self) -> np.ndarray:
        return self.read_state().joint_velocities

    @property
    @abstractmethod
    def timestamp(self) -> float:
        """Current monotonic time on the state/camera timestamp clock."""

    @abstractmethod
    def set_joint_positions(self, positions) -> None:
        """Command six joint position targets in the documented order."""

    @abstractmethod
    def set_joint_velocities(self, velocities) -> None:
        """Command six joint velocities in radians/second."""

    @property
    def gripper_width(self) -> float:
        """Read inner finger opening in meters."""
        return self.read_state().gripper_width

    @abstractmethod
    def set_gripper_width(self, width: float) -> None:
        """Command a symmetric inner finger opening in meters."""

    @property
    @abstractmethod
    def gripper_max_width(self) -> float:
        """Largest supported inner opening, in meters."""

    def open_gripper(self) -> None:
        """Fully open the backend's gripper."""
        self.set_gripper_width(self.gripper_max_width)

    def close_gripper(self) -> None:
        """Close the fingers; object contact can prevent full closure."""
        self.set_gripper_width(0.0)

    @abstractmethod
    def close(self) -> None:
        """Release camera/transport resources; do not advance robot motion."""

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()


