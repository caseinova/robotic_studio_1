"""Load and simulate the bundled UR5e using MuJoCo.

Example::

    robot = RobotModel()
    robot.set_joint_positions(robot.joint_positions)
    robot.step(100)

Run ``python robot_model.py`` to open the viewer, or pass ``--headless``
to simulate without a window. Use the project's virtual environment.
"""

from pathlib import Path
import argparse
import time

import mujoco
import numpy as np


MODEL_DIRECTORY = (
    Path(__file__).resolve().parent
    / "universal_robots_ur5e"
    / "universal_robots_ur5e"
)


class RobotModel:
    """UR5e simulation with position targets in radians.

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

    def set_joint_positions(self, positions) -> None:
        """Set six position actuator targets without teleporting the joints.

        Targets must be finite and within the bundled actuator limits.
        Call ``step`` to advance the robot toward them.
        """
        targets = np.asarray(positions, dtype=float)
        if targets.shape != (6,) or not np.all(np.isfinite(targets)):
            raise ValueError("positions must contain six finite joint angles")
        limits = self.model.actuator_ctrlrange[self.actuator_ids]
        if np.any(targets < limits[:, 0]) or np.any(targets > limits[:, 1]):
            raise ValueError("joint position targets exceed actuator limits")
        self.data.ctrl[self.actuator_ids] = targets

    def step(self, steps: int = 1) -> None:
        """Advance by ``steps`` fixed MuJoCo timesteps."""
        if isinstance(steps, bool) or not isinstance(steps, int) or steps < 0:
            raise ValueError("steps must be a non-negative integer")
        for _ in range(steps):
            mujoco.mj_step(self.model, self.data)

    def launch_viewer(self) -> None:
        """Simulate at approximately real time until the viewer is closed."""
        import mujoco.viewer

        with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
            while viewer.is_running():
                start = time.perf_counter()
                self.step()
                viewer.sync()
                remaining = self.model.opt.timestep - (time.perf_counter() - start)
                if remaining > 0:
                    time.sleep(remaining)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--headless", action="store_true", help="run without a viewer")
    parser.add_argument("--steps", type=int, default=1000, help="headless simulation steps")
    args = parser.parse_args()
    if args.steps < 0:
        parser.error("--steps must be non-negative")
    robot = RobotModel()
    if args.headless:
        robot.step(args.steps)
        print(f"Simulated {robot.data.time:.3f} seconds")
        print("Joint positions (radians):", robot.joint_positions)
    else:
        robot.launch_viewer()


if __name__ == "__main__":
    main()
