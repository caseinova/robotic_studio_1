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

    @abstractmethod
    def set_joint_velocities(self, velocities) -> None:
        """Command six joint velocities in radians/second."""


class MujocoInterface(RobotInterface):
    """Connect the controller API to an existing MuJoCo robot model."""

    def __init__(self, robot: "RobotModel"):
        self.robot = robot
        ids = robot.actuator_ids
        self._position_gain = robot.model.actuator_gainprm[ids].copy()
        self._position_bias = robot.model.actuator_biasprm[ids].copy()
        self._velocity_mode = False

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
        ids = self.robot.actuator_ids
        self.robot.model.actuator_gainprm[ids] = self._position_gain
        self.robot.model.actuator_biasprm[ids] = self._position_bias
        self._velocity_mode = False
        self.robot.data.qfrc_applied[self.robot._qvel_indices] = 0
        self.robot.data.ctrl[self.robot.actuator_ids] = targets

    def set_joint_velocities(self, velocities) -> None:
        """Switch the bundled affine actuators to damped velocity servos.

        The simulation calls prepare_step() for bias-force compensation.
        set_joint_positions() restores the original position actuators.
        """
        targets = np.asarray(velocities, dtype=float)
        if targets.shape != (6,) or not np.all(np.isfinite(targets)):
            raise ValueError("velocities must contain six finite values")
        if np.any(np.abs(targets) > 0.6):
            raise ValueError("simulation velocity commands are limited to 0.6 rad/s")
        ids = self.robot.actuator_ids
        gains = -self._position_bias[:, 2]
        self.robot.model.actuator_gainprm[ids, 0] = gains
        self.robot.model.actuator_biasprm[ids, :3] = np.column_stack(
            (np.zeros(6), np.zeros(6), -gains)
        )
        self.robot.data.ctrl[ids] = targets
        self._velocity_mode = True

    def prepare_step(self) -> None:
        """Update simulation feedforward before a physics step."""
        if self._velocity_mode:
            import mujoco

            mujoco.mj_forward(self.robot.model, self.robot.data)
            indices = self.robot._qvel_indices
            self.robot.data.qfrc_applied[indices] = self.robot.data.qfrc_bias[indices]
