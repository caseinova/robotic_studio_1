"""Prediction-only kinematics and collision geometry, independent of live I/O."""

from typing import Protocol

import numpy as np

from common.robot_state import RobotState



def inverse_pose(transform):
    result = np.eye(4)
    result[:3, :3] = transform[:3, :3].T
    result[:3, 3] = -result[:3, :3] @ transform[:3, 3]
    return result


class Kinematics(Protocol):
    """Calculations used by controllers; inputs come from RobotInterface.

    Poses map local coordinates into world coordinates. Jacobians have linear
    rows followed by angular rows, with columns in interface joint order.
    Collision-pair ordering must remain stable over an MPC solve.
    """

    limits: tuple[np.ndarray, np.ndarray]
    end_effector_marker_id: int
    end_effector_marker_size: float
    tool_from_end_effector_marker: np.ndarray

    def synchronize(self, state: RobotState) -> None: ...
    def tool_pose(self) -> np.ndarray: ...
    def grasp_pose(self) -> np.ndarray: ...
    def tool_jacobian(self, positions, *, local=False) -> np.ndarray: ...
    def distances(self, positions, *, gradients=False): ...
    def pickup_collision_indices(self) -> list[int]: ...
    def collision_pair_name(self, index: int) -> str: ...


