"""MuJoCo measurements and actuation."""
from dataclasses import replace
from typing import TYPE_CHECKING
import numpy as np
from common.robot_state import RGBDFrame, RobotState
from simulation.noise import MeasurementNoise
from simulation.noise_config import NoiseConfig
from common.interface import RobotInterface
if TYPE_CHECKING:
    from simulation.robot_model import RobotModel

class MujocoInterface(RobotInterface):
    """Connect the controller API to an existing MuJoCo robot model."""

    def __init__(self, robot: "RobotModel", *, noise_config: NoiseConfig | None = None):
        self.robot = robot
        ids = robot.actuator_ids
        self._position_gain = robot.model.actuator_gainprm[ids].copy()
        self._position_bias = robot.model.actuator_biasprm[ids].copy()
        self._velocity_mode = False
        self._renderer = None
        self.noise = MeasurementNoise(noise_config)
        self._camera_kinematics = None

    def read_state(self) -> RobotState:
        """Called on the physics thread, between simulation steps."""
        clean_state = RobotState(
            timestamp=float(self.robot.data.time),
            joint_positions=self.robot.joint_positions,
            joint_velocities=self.robot.joint_velocities,
            gripper_width=self.robot.gripper_width,
        )
        return self.noise.apply_state(clean_state, max_gripper_width=self.gripper_max_width)

    def reset_noise(self) -> None:
        """Start a new experiment with the configured seed and no cached state."""
        self.noise.reset()

    @property
    def timestamp(self) -> float:
        return float(self.robot.data.time)

    @property
    def gripper_max_width(self) -> float:
        return self.robot.gripper_max_width

    def capture_rgbd(self) -> RGBDFrame:
        """Keep the rendering backend behind the same API as a real camera."""
        if self._renderer is None:
            import mujoco

            self._renderer = mujoco.Renderer(self.robot.model, height=480, width=640)
        frame = self.robot.capture_rgbd(renderer=self._renderer)
        if self.noise.config.enabled:
            # Images are rendered from true physics. Their world pose is the
            # encoder-based estimate a real camera adapter would supply.
            from simulation.kinematics import MujocoKinematics

            if self._camera_kinematics is None:
                self._camera_kinematics = MujocoKinematics(self.robot)
            self._camera_kinematics.synchronize(self.read_state())
            frame = replace(frame, world_from_camera=self._camera_kinematics.camera_pose())
        return self.noise.apply_frame(frame)

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None

    def set_gripper_width(self, width: float) -> None:
        """Drive both finger slides, without changing the arm's control mode."""
        if (isinstance(width, bool) or not isinstance(width, (int, float))
                or not np.isfinite(width) or not 0 <= width <= self.robot.gripper_max_width):
            raise ValueError(f"gripper width must be finite and between 0 and {self.robot.gripper_max_width} meters")
        from simulation.rg2_model import width_to_angle
        self.robot.data.ctrl[self.robot.gripper_actuator_ids] = width_to_angle(width)

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
