"""Load the bundled UR5e model and its MuJoCo state.

Control access lives in interface.py; simulation execution lives in simulate.py.
"""

from pathlib import Path

import mujoco
import numpy as np


MODEL_DIRECTORY = (
    Path(__file__).resolve().parent
    / "universal_robots_ur5e"
    / "universal_robots_ur5e"
)


class RobotModel:
    """UR5e model definition and state.

    ``model`` and ``data`` are public MuJoCo objects for controllers,
    estimators, rendering, and other simulation integrations.
    """

    joint_names = (
        "shoulder_pan_joint",
        "shoulder_lift_joint",
        "elbow_joint",
        "wrist_1_joint",
        "wrist_2_joint",
        "wrist_3_joint",
    )
    actuator_names = (
        "shoulder_pan", "shoulder_lift", "elbow", "wrist_1", "wrist_2", "wrist_3"
    )

    def __init__(self, *, include_scene: bool = True):
        self.model_path = MODEL_DIRECTORY / (
            "scene.xml" if include_scene else "ur5e.xml"
        )
        self.model = mujoco.MjModel.from_xml_path(str(self.model_path))
        self.data = mujoco.MjData(self.model)
        self.joint_ids = np.array([
            self.model.joint(name).id for name in self.joint_names
        ])
        self.actuator_ids = np.array([
            self.model.actuator(name).id for name in self.actuator_names
        ])
        self._qpos_indices = self.model.jnt_qposadr[self.joint_ids]
        self._qvel_indices = self.model.jnt_dofadr[self.joint_ids]
        self.reset()

    @property
    def joint_positions(self) -> np.ndarray:
        """Return a copy of the six joint angles in radians."""
        return self.data.qpos[self._qpos_indices].copy()

    @property
    def joint_velocities(self) -> np.ndarray:
        """Return a copy of the six joint velocities in radians/second."""
        return self.data.qvel[self._qvel_indices].copy()

    def reset(self) -> None:
        """Reset time, state, and actuator targets to the model's home pose."""
        mujoco.mj_resetDataKeyframe(
            self.model, self.data, self.model.key("home").id
        )
        mujoco.mj_forward(self.model, self.data)

