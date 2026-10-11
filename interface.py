"""Portable joint control API for simulation and future hardware backends."""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from robot_model import RobotModel


class RobotInterface(ABC):
    """API used by upper-level controllers regardless of the robot backend.

    Joint order: shoulder pan, shoulder lift, elbow, wrist 1, wrist 2, wrist 3.
    Angles use radians and velocities use radians/second. Hardware adapters
    should implement this contract using their chosen robot transport.
    """

    @property
    @abstractmethod
    def joint_positions(self) -> np.ndarray:
        """Read six measured joint angles as an independent array."""

    @property
    @abstractmethod
    def joint_velocities(self) -> np.ndarray:
        """Read six measured joint velocities as an independent array."""

    @abstractmethod
    def set_joint_positions(self, positions) -> None:
        """Command six joint position targets in the documented order."""


class MujocoInterface(RobotInterface):
    """Connect the controller API to an existing MuJoCo robot model."""

    def __init__(self, robot: "RobotModel"):
        self.robot = robot

    @property
    def joint_positions(self) -> np.ndarray:
        return self.robot.joint_positions

    @property
    def joint_velocities(self) -> np.ndarray:
        return self.robot.joint_velocities

    def set_joint_positions(self, positions) -> None:
        """Set actuator targets; simulate.py advances the physics."""
        targets = np.asarray(positions, dtype=float)
        if targets.shape != (6,) or not np.all(np.isfinite(targets)):
            raise ValueError("positions must contain six finite joint angles")
        limits = self.robot.model.actuator_ctrlrange[self.robot.actuator_ids]
        if np.any(targets < limits[:, 0]) or np.any(targets > limits[:, 1]):
            raise ValueError("joint position targets exceed actuator limits")
        self.robot.data.ctrl[self.robot.actuator_ids] = targets
