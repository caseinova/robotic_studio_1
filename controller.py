"""PBVS joint-velocity control, optionally tracked by constrained joint MPC.

Prediction: q[k+1] = q[k] + dt*u[k]. The horizon cost tracks the PBVS
velocity reference and penalizes velocity changes. Hard constraints enforce
joint position, velocity and acceleration limits plus signed geom clearance.
Collision constraints are linearized about the previous plan, then verified
on the nonlinear geometry at endpoints and intermediate samples. This local
kinematic MPC is not a global path planner or a dynamic hardware safety system.
"""

from dataclasses import dataclass, replace
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
                 tool_clearance: float | None = None, config: MPCConfig | None = None,
                 controller_config: ControllerConfig | None = None):
        self.interface = interface
        self.processor = CameraProcessor()
        self.kinematics = kinematics
        self.settings = controller_config or ControllerConfig()
        if tool_clearance is not None:
            self.settings = replace(self.settings, tool_clearance=tool_clearance)
        self.limits_config = config or MPCConfig()
        self.control_dt = self.limits_config.dt
        self.mpc = (JointSpaceMPC(self.kinematics, self.limits_config)
                    if self.settings.mode == "pbvs_mpc" else None)
        marker_from_tool = np.eye(4)
        marker_from_tool[:3, :3] = np.diag([1, -1, -1])
        marker_from_tool[:3, 3] = [0, 0, self.settings.tool_clearance]
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


class CubePickupController:
    """Open, descend vertically, grip, then return vertically to the entry pose.

    Start only after ApproachController reaches its target. The detected cube
    pose is latched at handoff: the tag need not remain visible inside the
    gripper. All measurements and commands go through RobotInterface. The
    supplied kinematics is a prediction workspace, never live simulator data.

    This is a Cartesian velocity sequence, independent of PBVS/MPC. Known
    obstacles and other robot links are checked before each motion command.
    Finger contact with the selected cube is intentionally allowed. Gripper
    width settling indicates a possible grasp; it is not a force measurement.
    """

    def __init__(self, interface: "RobotInterface", kinematics: MujocoKinematics, *,
                 cube_size: float = 0.10, speed: float = 0.03,
                 control_dt: float = 0.05, phase_timeout: float = 20.0):
        for name, value in (("cube_size", cube_size), ("speed", speed),
                            ("control_dt", control_dt), ("phase_timeout", phase_timeout)):
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        self.interface = interface
        self.kinematics = kinematics
        self.cube_size = cube_size
        self.speed = speed
        self.control_dt = control_dt
        self.phase_timeout = phase_timeout
        self.phase = "idle"
        self.diagnostic = ""
        self._phase_started = 0.0
        self._last_width = None
        self._width_stable_since = 0.0
        self._entry_pose = None
        self._descent_pose = None
        self._pose_integral = np.zeros(6)
        self._hold_positions = None

        # Existing prediction pairs remain untouched. Only the two finger/cube
        # pairs may make contact during pickup; the palm must stay clear.
        model = kinematics.model
        cube_body = model.body("cube").id
        cube_geoms = set(np.flatnonzero(model.geom_bodyid == cube_body))
        fingers = {model.geom(name).id for name in (
            "gripper_left_geom", "gripper_right_geom"
        )}
        self._collision_indices = [
            index for index, (a, b) in enumerate(kinematics.pairs)
            if not ((a in fingers and b in cube_geoms)
                    or (b in fingers and a in cube_geoms))
        ]

    @property
    def finished(self) -> bool:
        return self.phase in ("complete", "failed")

    def start(self, observation: CubeObservation) -> None:
        """Latch the final approach pose and the observed cube center."""
        if self.phase != "idle":
            raise RuntimeError("pickup has already started; construct a new controller to retry")
        state = self.interface.read_state()
        now = self.interface.timestamp
        if (observation is None or not np.isfinite(now)
                or not np.isfinite(observation.timestamp) or not np.isfinite(state.timestamp)
                or abs(now - observation.timestamp) > 2 * self.control_dt
                or abs(now - state.timestamp) > 2 * self.control_dt):
            self._fail("stale_pickup_target_or_state")
            return
        if (not np.all(np.isfinite(observation.world_from_cube))
                or not np.all(np.isfinite(state.joint_positions))
                or not np.isfinite(state.gripper_width)):
            self._fail("invalid_pickup_target_or_state")
            return
        self.kinematics.synchronize(state)
        self._entry_pose = self.kinematics.tool_pose().copy()
        self._descent_pose = self._entry_pose.copy()
        grasp_position = self.kinematics.data.site("grasp_site").xpos
        # Align the grasp site's height with the cube center. Keep flange X/Y
        # and orientation fixed, giving a vertical path in the world frame.
        cube_center = observation.world_from_cube[:3, 3]
        self._descent_pose[2, 3] += cube_center[2] - grasp_position[2]
        if np.linalg.norm(grasp_position[:2] - cube_center[:2]) > 0.01:
            self._fail("tool_not_centered_over_cube")
            return
        if self._descent_pose[2, 3] >= self._entry_pose[2, 3]:
            self._fail("cube_not_below_gripper")
            return
        if self.interface.gripper_max_width < self.cube_size + 0.01:
            self._fail("gripper_opening_too_small")
            return
        self.interface.set_joint_velocities(np.zeros(6))
        self.interface.open_gripper()
        self._change_phase("opening", now)

    def update(self) -> ControlResult:
        """Advance one phase without blocking, sleeping, or acquiring images."""
        state = self.interface.read_state()
        now = self.interface.timestamp
        if self.phase == "idle":
            self.interface.set_joint_velocities(np.zeros(6))
            return self._result()
        if self.finished:
            if self.phase == "failed":
                self.interface.set_joint_velocities(np.zeros(6))
            return self._result()
        if (not np.isfinite(now) or not np.isfinite(state.timestamp)
                or abs(now - state.timestamp) > 2 * self.control_dt
                or not np.all(np.isfinite(state.joint_positions))
                or not np.all(np.isfinite(state.joint_velocities))
                or not np.isfinite(state.gripper_width)):
            self._fail("stale_or_invalid_robot_state")
            return self._result()
        if now - self._phase_started > self.phase_timeout:
            self._fail("phase_timeout: " + self.phase)
            return self._result()
        self.kinematics.synchronize(state)

        if self.phase == "opening":
            self.interface.set_joint_velocities(np.zeros(6))
            if state.gripper_width >= self.interface.gripper_max_width - 0.001:
                self._change_phase("descending", now)
            return self._result()

        if self.phase == "closing":
            self.interface.set_joint_velocities(np.zeros(6))
            if abs(state.gripper_width - self._last_width) > 0.0005:
                self._width_stable_since = now
            self._last_width = state.gripper_width
            elapsed = now - self._phase_started
            if elapsed >= 0.5:
                if state.gripper_width < self.cube_size * 0.5:
                    self._fail("empty_grasp")
                elif (state.gripper_width <= self.cube_size * 1.15
                      and now - self._width_stable_since >= 0.3):
                    self._change_phase("lifting", now)
            return self._result()

        if self.phase == "lifting" and state.gripper_width < self.cube_size * 0.5:
            self._fail("grasp_lost")
            return self._result()

        target = self._descent_pose if self.phase == "descending" else self._entry_pose
        current = self.kinematics.tool_pose()
        error = target[:3, 3] - current[:3, 3]
        rotation_error = cv2.Rodrigues(target[:3, :3] @ current[:3, :3].T)[0].reshape(3)
        if np.linalg.norm(error) < 0.0015 and np.linalg.norm(rotation_error) < 0.02:
            self.interface.set_joint_velocities(np.zeros(6))
            if self.phase == "descending":
                self.interface.close_gripper()
                self._last_width = state.gripper_width
                self._width_stable_since = now
                self._change_phase("closing", now)
            else:
                # Hold a fixed joint position after lifting, including the
                # payload load, instead of leaving zero-velocity servos free
                # to drift under the cube's external contact forces.
                self._hold_positions = state.joint_positions.copy()
                self.interface.set_joint_positions(self._hold_positions)
                self._change_phase("complete", now)
            return self._result(position_error=np.linalg.norm(error))

        # Convert the optical-camera Jacobian into a world-frame flange
        # Jacobian. The offset correction moves the linear velocity reference
        # from the camera center to the tool flange.
        camera_jacobian = self.kinematics.camera_jacobian(state.joint_positions)
        camera_pose = self.kinematics.camera_pose()
        angular = camera_pose[:3, :3] @ camera_jacobian[3:]
        offset = camera_pose[:3, 3] - current[:3, 3]
        linear = camera_pose[:3, :3] @ camera_jacobian[:3] + np.cross(offset, angular.T).T
        jacobian = np.vstack((linear, angular))
        # Integral feedback counters the extra load from the held cube. X/Y
        # feedback corrects drift off the vertical line rather than planning
        # horizontal travel.
        pose_error = np.concatenate((error, rotation_error))
        # Integrate translation only near the target to avoid winding up
        # while the vertical speed is saturated over a long descent/lift.
        integration_error = pose_error.copy()
        if np.linalg.norm(error) > self.speed / 8:
            integration_error[2] = 0
        self._pose_integral = np.clip(
            self._pose_integral + self.control_dt * integration_error, -0.02, 0.02
        )
        twist = 8.0 * pose_error + 4.0 * self._pose_integral
        linear_velocity = twist[:3]
        # Limit vertical travel independently, so a large height error cannot
        # dilute the horizontal feedback that keeps the tool on its line.
        linear_velocity[:2] *= min(1.0, self.speed / max(np.linalg.norm(linear_velocity[:2]), 1e-12))
        linear_velocity[2] = np.clip(linear_velocity[2], -self.speed, self.speed)
        twist[3:] *= min(1.0, 0.3 / max(np.linalg.norm(twist[3:]), 1e-12))
        command = jacobian.T @ np.linalg.solve(jacobian @ jacobian.T + 0.01 ** 2 * np.eye(6), twist)
        command /= max(1.0, np.max(np.abs(command)) / 0.35)
        lower, upper = self.kinematics.limits
        prediction = state.joint_positions + self.control_dt * command
        if np.any(prediction < lower) or np.any(prediction > upper):
            self._fail("joint_limit_violation")
            return self._result()
        minimum = min(self._clearance(state.joint_positions), self._clearance(prediction))
        if minimum < 0.02:
            self._fail("pickup_path_blocked")
            return self._result(minimum_clearance=minimum)
        self.interface.set_joint_velocities(command)
        return self._result(command, np.linalg.norm(error), minimum)

    def _clearance(self, positions) -> float:
        distances = self.kinematics.distances(positions)
        return float(np.min(distances[self._collision_indices], initial=np.inf))

    def _change_phase(self, phase, now) -> None:
        self.phase = phase
        self._phase_started = now
        self._pose_integral[:] = 0

    def _fail(self, reason) -> None:
        self.phase = "failed"
        self.diagnostic = reason
        self.interface.set_joint_velocities(np.zeros(6))

    def _result(self, command=None, position_error=0.0, minimum_clearance=np.nan) -> ControlResult:
        velocity = np.zeros(6) if command is None else command.copy()
        return ControlResult(None, velocity, velocity.copy(), "pickup_" + self.phase,
                             position_error, 0.0, minimum_clearance, self.diagnostic)
