"""Measured hardware I/O and command validation."""
from dataclasses import replace
import time
import numpy as np
from common.robot_state import RGBDFrame, RobotState
from common.interface import RobotInterface

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


