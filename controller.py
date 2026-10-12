"""PBVS joint-velocity control, optionally tracked by constrained joint MPC.

Prediction: q[k+1] = q[k] + dt*u[k]. The horizon cost tracks the PBVS
velocity reference and penalizes velocity changes. Hard constraints enforce
joint position, velocity and acceleration limits plus signed geom clearance.
Collision constraints are linearized about the previous plan, then verified
on the nonlinear geometry at endpoints and intermediate samples. This local
kinematic MPC is not a global path planner or a dynamic hardware safety system.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING

import cv2
import numpy as np
from scipy.optimize import minimize

from camera_processing import CameraProcessor, CubeObservation
from controller_config import ControllerConfig
from robot_state import RGBDFrame
from kinematics import MujocoKinematics, inverse_pose

if TYPE_CHECKING:
    from interface import RobotInterface


@dataclass(frozen=True)
class MPCConfig:
    dt: float = 0.05
    horizon: int = 8
    velocity_limit: float = 0.35  # rad/s
    acceleration_limit: float = 1.5  # rad/s^2
    clearance: float = 0.02  # meters
    clearance_buffer: float = 0.003  # extra prediction margin
    tracking_weight: float = 1.0
    smoothing_weight: float = 0.15
    max_iterations: int = 60

    def __post_init__(self):
        for name in ("dt", "velocity_limit", "acceleration_limit", "clearance",
                     "tracking_weight", "smoothing_weight"):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if self.velocity_limit > 0.6:
            raise ValueError("velocity_limit exceeds the simulation interface limit")
        if not np.isfinite(self.clearance_buffer) or self.clearance_buffer < 0:
            raise ValueError("clearance_buffer must be finite and non-negative")
        if not isinstance(self.horizon, int) or self.horizon < 2:
            raise ValueError("horizon must be an integer of at least two steps")
        if not isinstance(self.max_iterations, int) or self.max_iterations < 1:
            raise ValueError("max_iterations must be a positive integer")


class PBVS:
    """Pose error -> optical camera twist -> damped joint-velocity reference."""

    def __init__(self, desired_camera_from_marker, *, gain=1.2, damping=0.025):
        self.desired = np.asarray(desired_camera_from_marker, dtype=float)
        if self.desired.shape != (4, 4) or not np.all(np.isfinite(self.desired)):
            raise ValueError("desired camera pose must be finite and 4x4")
        if not np.isfinite(gain) or gain <= 0 or not np.isfinite(damping) or damping <= 0:
            raise ValueError("PBVS gain and damping must be finite and positive")
        self.gain = gain
        self.damping = damping

    def reference(self, camera_from_marker, camera_jacobian):
        # Desired camera pose expressed in the current optical camera frame.
        error_pose = camera_from_marker @ inverse_pose(self.desired)
        rotation_error, _ = cv2.Rodrigues(error_pose[:3, :3])
        error = np.concatenate((error_pose[:3, 3], rotation_error.reshape(3)))
        twist = self.gain * error
        for part, maximum in ((slice(0, 3), 0.06), (slice(3, 6), 0.3)):
            norm = np.linalg.norm(twist[part])
            if norm > maximum:
                twist[part] *= maximum / norm
        jacobian = np.asarray(camera_jacobian)
        reference = jacobian.T @ np.linalg.solve(
            jacobian @ jacobian.T + self.damping ** 2 * np.eye(6), twist
        )
        return reference, error


@dataclass(frozen=True)
class MPCSolution:
    velocity: np.ndarray
    trajectory: np.ndarray
    feasible: bool
    status: str
    minimum_clearance: float


class JointSpaceMPC:
    """Receding-horizon velocity tracking with hard clearance inequalities."""

    def __init__(self, kinematics: MujocoKinematics, config: MPCConfig | None = None):
        self.kinematics = kinematics
        self.config = config or MPCConfig()
        n = self.config.horizon
        self.integration = np.kron(np.tril(np.ones((n, n))) * self.config.dt, np.eye(6))
        difference = np.eye(n) - np.eye(n, k=-1)
        self.difference = np.kron(difference, np.eye(6))
        self.previous_plan = np.zeros((n, 6))

    def reset(self):
        self.previous_plan[:] = 0

    def solve(self, positions, velocities, reference) -> MPCSolution:
        c, k = self.config, self.kinematics
        q, dq, ref = (np.asarray(value, dtype=float) for value in (positions, velocities, reference))
        if any(value.shape != (6,) or not np.all(np.isfinite(value)) for value in (q, dq, ref)):
            raise ValueError("MPC positions, velocities and reference must be six finite values")
        current_distance = float(np.min(k.distances(q), initial=np.inf))
        if current_distance < c.clearance:
            return self._stop(q, "clearance_violation", current_distance)
        ref = ref / max(1, np.max(np.abs(ref)) / c.velocity_limit)
        desired = np.tile(ref, c.horizon)
        previous_velocity = np.zeros(c.horizon * 6)
        previous_velocity[:6] = dq
        shifted = np.vstack((self.previous_plan[1:], self.previous_plan[-1]))
        initial = shifted.reshape(-1)

        def objective(u):
            tracking = u - desired
            change = self.difference @ u - previous_velocity
            return c.tracking_weight * (tracking @ tracking) + c.smoothing_weight * (change @ change)

        def gradient(u):
            return (2 * c.tracking_weight * (u - desired)
                    + 2 * c.smoothing_weight * self.difference.T @ (self.difference @ u - previous_velocity))

        qbase = np.tile(q, c.horizon)
        lower, upper = (np.tile(limit, c.horizon) for limit in k.limits)
        base_matrix = np.vstack((self.integration, -self.integration,
                                 self.difference, -self.difference))
        base_rhs = np.concatenate((lower - qbase, qbase - upper,
                                   previous_velocity - c.acceleration_limit * c.dt,
                                   -previous_velocity - c.acceleration_limit * c.dt))
        for _ in range(3):
            matrices, rhs = [base_matrix], [base_rhs]
            # Constrain both midpoint and endpoint of every prediction interval.
            for step in range(c.horizon):
                for fraction in (0.5, 1.0):
                    integrator = self.integration[step * 6:(step + 1) * 6].copy()
                    integrator[:, step * 6:(step + 1) * 6] *= fraction
                    nominal = q + integrator @ initial
                    distances, derivatives = k.distances(nominal, gradients=True)
                    matrices.append(derivatives @ integrator)
                    rhs.append(c.clearance + c.clearance_buffer - distances + derivatives @ (nominal - q))
            matrix, bound = np.vstack(matrices), np.concatenate(rhs)
            result = minimize(
                objective, initial, jac=gradient, method="SLSQP",
                bounds=[(-c.velocity_limit, c.velocity_limit)] * len(initial),
                constraints={"type": "ineq", "fun": lambda u: matrix @ u - bound,
                             "jac": lambda u: matrix},
                options={"maxiter": c.max_iterations, "ftol": 1e-8},
            )
            if not result.success or not np.all(np.isfinite(result.x)):
                return self._stop(q, "optimizer_failed: " + result.message, current_distance)
            plan = result.x.reshape(c.horizon, 6)
            trajectory = q + (self.integration @ result.x).reshape(c.horizon, 6)
            minimum = current_distance
            start = q
            for end in trajectory:
                for fraction in (0.25, 0.5, 0.75, 1.0):
                    minimum = min(minimum, float(np.min(k.distances(start + fraction * (end - start)), initial=np.inf)))
                start = end
            if minimum >= c.clearance and np.min(matrix @ result.x - bound) >= -1e-6:
                self.previous_plan = plan
                return MPCSolution(plan[0].copy(), trajectory, True, "tracking", minimum)
            initial = result.x
        return self._stop(q, "nonlinear_clearance_check_failed", current_distance)

    def _stop(self, positions, status, minimum):
        self.reset()
        return MPCSolution(np.zeros(6), np.tile(positions, (self.config.horizon, 1)), False, status, minimum)


@dataclass(frozen=True)
class ControlResult:
    observation: CubeObservation | None
    reference_velocity: np.ndarray
    command_velocity: np.ndarray
    status: str
    position_error: float
    rotation_error: float
    minimum_clearance: float
    diagnostic: str = ""


class ApproachController:
    """Detect the cube and move the tool above its tagged face, without grasping."""

    def __init__(self, interface: "RobotInterface", kinematics: MujocoKinematics, *,
                 tool_clearance: float = 0.18, config: MPCConfig | None = None,
                 controller_config: ControllerConfig | None = None):
        if not np.isfinite(tool_clearance) or tool_clearance < 0.16:
            raise ValueError("tool_clearance must be at least 0.16 m to keep the offset camera's tag in view")
        self.interface = interface
        self.processor = CameraProcessor()
        self.kinematics = kinematics
        self.settings = controller_config or ControllerConfig()
        self.limits_config = config or MPCConfig()
        self.control_dt = self.limits_config.dt
        self.mpc = (JointSpaceMPC(self.kinematics, self.limits_config)
                    if self.settings.mode == "pbvs_mpc" else None)
        marker_from_tool = np.eye(4)
        marker_from_tool[:3, :3] = np.diag([1, -1, -1])
        marker_from_tool[:3, 3] = [0, 0, tool_clearance]
        marker_from_camera = marker_from_tool @ self.kinematics.tool_from_camera
        self.pbvs = PBVS(inverse_pose(marker_from_camera))

    def update(self, frame: RGBDFrame | None = None) -> ControlResult:
        """Acquire through the interface and command one control interval.

        A caller displaying the camera can pass an already acquired interface
        frame to avoid capturing twice. State is read once for this update.
        """
        if frame is None:
            frame = self.interface.capture_rgbd()
        state = self.interface.read_state()
        self.kinematics.synchronize(state)
        observation = self.processor.process(frame)
        now = self.interface.timestamp
        stale_frame = abs(now - frame.timestamp) > 2 * self.control_dt
        stale_state = abs(now - state.timestamp) > 2 * self.control_dt
        if observation is None or stale_frame or stale_state:
            self.interface.set_joint_velocities(np.zeros(6))
            if self.mpc is not None:
                self.mpc.reset()
            return ControlResult(observation, np.zeros(6), np.zeros(6), "target_lost_or_stale",
                                 np.inf, np.inf, np.inf,
                                 diagnostic=("stale_rgbd_frame" if stale_frame
                                             else "stale_robot_state" if stale_state
                                             else self.processor.last_status))
        positions = state.joint_positions
        minimum = (float(np.min(self.kinematics.distances(positions), initial=np.inf))
                   if self.mpc is not None else np.nan)
        if self.mpc is not None and minimum < self.mpc.config.clearance:
            self.interface.set_joint_velocities(np.zeros(6))
            self.mpc.reset()
            return ControlResult(observation, np.zeros(6), np.zeros(6), "clearance_violation",
                                 np.inf, np.inf, minimum)
        jacobian = self.kinematics.camera_jacobian(positions)
        reference, error = self.pbvs.reference(observation.camera_from_marker, jacobian)
        position_error, rotation_error = np.linalg.norm(error[:3]), np.linalg.norm(error[3:])
        if position_error < 0.003 and rotation_error < 0.025:
            self.interface.set_joint_velocities(np.zeros(6))
            if self.mpc is not None:
                self.mpc.reset()
            return ControlResult(observation, reference, np.zeros(6), "reached",
                                 position_error, rotation_error, minimum)
        if self.mpc is None:
            # Direct PBVS: basic speed/joint bounds, with no collision optimizer.
            limit = self.limits_config.velocity_limit
            command = reference / max(1, np.max(np.abs(reference)) / limit)
            lower, upper = self.kinematics.limits
            if np.any(positions < lower) or np.any(positions > upper):
                command = np.zeros(6)
                status = "joint_limit_violation"
            else:
                command = np.clip(command, np.maximum(-limit, (lower - positions) / self.control_dt),
                                  np.minimum(limit, (upper - positions) / self.control_dt))
                status = "tracking"
            self.interface.set_joint_velocities(command)
            return ControlResult(observation, reference, command, status,
                                 position_error, rotation_error, minimum)
        solution = self.mpc.solve(positions, state.joint_velocities, reference)
        self.interface.set_joint_velocities(solution.velocity)
        return ControlResult(observation, reference, solution.velocity, solution.status,
                             position_error, rotation_error, solution.minimum_clearance)
