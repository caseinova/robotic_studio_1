"""Robot measurements, camera acquisition, and commands for every backend."""

from abc import ABC, abstractmethod
from dataclasses import replace
from typing import TYPE_CHECKING
import time

import numpy as np

from robot_state import RGBDFrame, RobotState
from noise import MeasurementNoise
from noise_config import NoiseConfig

if TYPE_CHECKING:
    from robot_model import RobotModel


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


class HardwareInterface(RobotInterface):
    """Portable measured-state and command API over configured device adapters.

    All adapter samples use the supplied monotonic clock. Camera adapters
    provide registered raw RGB-D; this class scales depth and rectifies both
    images using installation calibration. No synthetic noise is injected.
    """

    def __exit__(self, exc_type, exc_value, traceback):
        try:
            self.close()
        except Exception as cleanup_error:
            if exc_value is None:
                raise
            raise RuntimeError(f"{exc_type.__name__}: {exc_value}; additionally: {cleanup_error}") from exc_value
        return False

    def __init__(self, config, robot, camera, gripper, *, clock=time.monotonic):
        self.config, self.robot, self.camera, self.gripper = config, robot, camera, gripper
        self.clock = clock
        self._closed = False
        self._last_state = self._last_frame = None
        self.require_camera_for_motion = True
        import cv2

        self._intrinsics = np.asarray(config.camera["intrinsics"], dtype=float).copy()
        self._world_from_camera = np.asarray(config.calibration["world_from_camera"], dtype=float).copy()
        size = (config.camera["width"], config.camera["height"])
        self._maps = cv2.initUndistortRectifyMap(self._intrinsics,
            np.asarray(config.camera["distortion"], dtype=float), None, self._intrinsics, size, cv2.CV_32FC1)

    @property
    def timestamp(self):
        return self.clock()

    def _fresh(self, timestamp, max_age, name):
        age = self.timestamp - timestamp
        if not np.isfinite(timestamp) or not -1e-6 <= age <= max_age:
            raise RuntimeError(f"stale or invalid {name} timestamp (age {age:.3f} s)")

    def _open(self):
        if self._closed:
            raise RuntimeError("hardware interface is closed")

    def read_state(self):
        self._open()
        width = self.gripper.read_width() if self.gripper is not None else None
        # Gripper transport may block; sample arm feedback afterward so that
        # this latency does not consume the command's robot-state age budget.
        joints = self.robot.read_joint_state()
        self._fresh(joints.timestamp, self.config.control["state_max_age"], "robot state")
        if width is not None:
            self._fresh(width.timestamp, self.config.control["gripper_max_age"], "gripper state")
        q, dq = np.array(joints.positions, dtype=float, copy=True), np.array(joints.velocities, dtype=float, copy=True)
        if q.shape != (6,) or dq.shape != (6,) or not np.all(np.isfinite([q, dq])):
            raise RuntimeError("joint state must contain six finite positions and velocities")
        if width is not None and (not np.isfinite(width.width) or not -.0002 <= width.width <= self.gripper_max_width + .0002):
            raise RuntimeError("measured gripper width is outside the configured physical opening")
        self._last_state = RobotState(float(joints.timestamp), q, dq,
                                      None if width is None else float(np.clip(width.width, 0, self.gripper_max_width)))
        return replace(self._last_state, joint_positions=q.copy(), joint_velocities=dq.copy())

    def capture_rgbd(self):
        self._open()
        import cv2

        sample = self.camera.capture()
        self._fresh(sample.timestamp, self.config.control["frame_max_age"], "RGB-D frame")
        if self._last_frame is not None and sample.timestamp <= self._last_frame.timestamp:
            raise RuntimeError("camera must return a new frame with increasing acquisition time")
        shape = (self.config.camera["height"], self.config.camera["width"])
        if sample.rgb.dtype != np.uint8 or sample.rgb.shape != (*shape, 3) or sample.depth.shape != shape:
            raise RuntimeError("camera must return registered RGB uint8 and depth at configured resolution")
        rgb = cv2.remap(sample.rgb, *self._maps, interpolation=cv2.INTER_LINEAR)
        depth = cv2.remap(np.asarray(sample.depth, dtype=np.float32) * self.config.camera["depth_scale"],
                          *self._maps, interpolation=cv2.INTER_NEAREST,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        self._last_frame = RGBDFrame(rgb.copy(), depth.copy(), self._intrinsics.copy(),
            self._world_from_camera.copy(), float(sample.timestamp), tuple(self.config.camera["depth_range"]))
        return self._last_frame

    @property
    def gripper_max_width(self):
        return self.config.gripper["max_width_m"]

    def _motion_inputs(self):
        self._open()
        if self._last_state is None:
            raise RuntimeError("read measured state before commanding hardware")
        self._fresh(self._last_state.timestamp, self.config.control["state_max_age"], "command state")
        if self.require_camera_for_motion:
            if self._last_frame is None:
                raise RuntimeError("acquire RGB-D before visual control")
            self._fresh(self._last_frame.timestamp, self.config.control["frame_max_age"], "command frame")

    def prepare_motion(self):
        self._open()
        prepare = getattr(self.robot, "prepare_motion", None)
        if callable(prepare):
            prepare()
        self._last_state = self._last_frame = None

    def set_joint_velocities(self, velocities):
        self._open()
        target = np.asarray(velocities, dtype=float)
        if target.shape != (6,) or not np.all(np.isfinite(target)):
            raise ValueError("velocities must contain six finite values")
        if np.any(np.abs(target) > self.config.control["velocity_limit"] + 1e-10):
            raise ValueError("hardware joint velocity limit exceeded")
        if np.any(target):
            self._motion_inputs()
            predicted = self._last_state.joint_positions + self.config.control["dt"] * target
            if (np.any(predicted < self.config.robot["joint_lower"])
                    or np.any(predicted > self.config.robot["joint_upper"])):
                raise ValueError("joint velocity command would exceed position limits")
        self.robot.command_velocity(target.copy())

    def set_joint_positions(self, positions):
        self._motion_inputs()
        target = np.asarray(positions, dtype=float)
        if (target.shape != (6,) or not np.all(np.isfinite(target))
                or np.any(target < self.config.robot["joint_lower"])
                or np.any(target > self.config.robot["joint_upper"])):
            raise ValueError("joint position target exceeds hardware limits")
        self.robot.command_position(target.copy())

    def set_gripper_width(self, width):
        if self.gripper is None:
            raise RuntimeError("gripper is disabled; gripper commands are unavailable")
        self._motion_inputs()
        if isinstance(width, bool) or not np.isfinite(width) or not 0 <= width <= self.gripper_max_width:
            raise ValueError("gripper command is outside configured physical opening")
        self.gripper.command_width(float(width))

    def close(self):
        if self._closed:
            return
        errors = []
        try:
            actions = [self.robot.stop, self.robot.close]
            if self.gripper is not None:
                actions.append(self.gripper.close)
            actions.append(self.camera.close)
            for action in actions:
                try:
                    action()
                except Exception as error:
                    errors.append(error)
        finally:
            self._closed = True
        if errors:
            raise RuntimeError("hardware cleanup failed: " + "; ".join(dict.fromkeys(map(str, errors)))) from errors[0]


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
            from kinematics import MujocoKinematics

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
            raise ValueError("gripper width must be finite and between 0 and 0.12 meters")
        self.robot.data.ctrl[self.robot.gripper_actuator_ids] = width / 2

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
