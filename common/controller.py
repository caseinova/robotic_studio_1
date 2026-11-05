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

from common.camera_processing import CameraProcessor, TaggedCubeProcessor, CubeObservation, MarkerObservation
from common.controller_config import ControllerConfig
from common.robot_state import RGBDFrame
from common.kinematics import Kinematics, inverse_pose
from common.estimator import EKFConfig, RelativePoseEKF, RelativePoseEstimate

if TYPE_CHECKING:
    from common.interface import RobotInterface


@dataclass(frozen=True)
class MPCConfig:
    dt: float = 0.05
    horizon: int = 8
    velocity_limit: float = 0.35  # rad/s
    acceleration_limit: float = 1.5  # rad/s^2
    clearance: float = 0.015  # meters; UR3e wrist meshes have ~19 mm separation at home
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
    """Relative pose error -> controlled-frame twist -> damped joint reference.

    The pose and Jacobian must share the controlled frame's axes. Eye-to-hand
    control uses tool-from-marker and a tool-frame flange Jacobian.
    """

    def __init__(self, desired_camera_from_marker, *, gain=1.2, damping=0.025,
                 linear_speed=0.06, angular_speed=0.3):
        self.desired = np.asarray(desired_camera_from_marker, dtype=float)
        if self.desired.shape != (4, 4) or not np.all(np.isfinite(self.desired)):
            raise ValueError("desired camera pose must be finite and 4x4")
        if not np.isfinite(gain) or gain <= 0 or not np.isfinite(damping) or damping <= 0:
            raise ValueError("PBVS gain and damping must be finite and positive")
        self.gain = gain
        self.damping = damping
        for name, value in (("linear_speed", linear_speed), ("angular_speed", angular_speed)):
            if (isinstance(value, bool) or not isinstance(value, (int, float))
                    or not np.isfinite(value) or value <= 0):
                raise ValueError(f"{name} must be finite and positive")
        self.linear_speed, self.angular_speed = linear_speed, angular_speed

    def reference(self, camera_from_marker, camera_jacobian):
        # Desired controlled-frame pose expressed in its current frame.
        error_pose = camera_from_marker @ inverse_pose(self.desired)
        rotation_error, _ = cv2.Rodrigues(error_pose[:3, :3])
        error = np.concatenate((error_pose[:3, 3], rotation_error.reshape(3)))
        twist = self.gain * error
        for part, maximum in ((slice(0, 3), self.linear_speed), (slice(3, 6), self.angular_speed)):
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

    def __init__(self, kinematics: Kinematics, config: MPCConfig | None = None):
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
        # Repeated nominal poses (especially the initial zero plan) have the
        # same geometry/Jacobian within this solve. Never reuse across cycles.
        linearizations = {}
        for _ in range(3):
            matrices, rhs = [base_matrix], [base_rhs]
            # Constrain both midpoint and endpoint of every prediction interval.
            for step in range(c.horizon):
                for fraction in (0.5, 1.0):
                    integrator = self.integration[step * 6:(step + 1) * 6].copy()
                    integrator[:, step * 6:(step + 1) * 6] *= fraction
                    nominal = q + integrator @ initial
                    key = nominal.tobytes()
                    if key not in linearizations:
                        distances, derivatives = k.distances(nominal, gradients=True)
                        linearizations[key] = (distances.copy(), derivatives.copy())
                    distances, derivatives = linearizations[key]
                    matrices.append(np.einsum("ij,jk->ik", derivatives, integrator, optimize=False))
                    rhs.append(c.clearance + c.clearance_buffer - distances + derivatives @ (nominal - q))
            matrix, bound = np.vstack(matrices), np.concatenate(rhs)
            # Tall, narrow constraints do not benefit from threaded BLAS
            # dispatch on each SLSQP callback.
            def constraints(u):
                return np.einsum("ij,j->i", matrix, u, optimize=False) - bound

            result = minimize(
                objective, initial, jac=gradient, method="SLSQP",
                bounds=[(-c.velocity_limit, c.velocity_limit)] * len(initial),
                constraints={"type": "ineq", "fun": constraints,
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
            if minimum >= c.clearance and np.min(constraints(result.x)) >= -1e-6:
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
    pose_estimate: RelativePoseEstimate | None = None
    tool_observation: MarkerObservation | None = None
    tool_pose_source: str = ""
    tool_pose: np.ndarray | None = None
    tool_pose_timestamp: float | None = None


class ApproachController:
    """Detect the cube and move the tool above its tagged face, without grasping."""

    def __init__(self, interface: "RobotInterface", kinematics: Kinematics, *,
                 tool_clearance: float | None = None, config: MPCConfig | None = None,
                 controller_config: ControllerConfig | None = None, tool_tracker=None,
                 estimator_config: EKFConfig | None = None, processor: TaggedCubeProcessor | None = None):
        self.interface = interface
        self.processor = processor if processor is not None else TaggedCubeProcessor(
            cube_size=getattr(kinematics, 'cube_size', .1),
            marker_size=getattr(kinematics, 'cube_marker_size', .08))
        estimator_config = estimator_config if estimator_config is not None else EKFConfig()
        self.estimator = RelativePoseEKF(estimator_config.for_target("cube"))
        # A separate tracker smooths the moving tool measured by its tag.
        # Its tuning and state are independent of the cube tracker.
        self.tool_estimator = RelativePoseEKF(estimator_config.for_target("tool"))
        self.tool_tracker = tool_tracker
        if (self.tool_tracker is None and self.tool_estimator.config.enabled
                and self.tool_estimator.config.prediction_model == 'fk_increment'
                and (controller_config or ControllerConfig()).tool_pose_source == 'tag_with_fk_fallback'):
            from common.tool_pose import ToolPoseTracker
            self.tool_tracker = ToolPoseTracker(interface, kinematics,
                controller_config or ControllerConfig(), estimator_config, (config or MPCConfig()).dt)
        self.pose_estimate = None
        self.kinematics = kinematics
        self.tool_processor = CameraProcessor(marker_id=kinematics.end_effector_marker_id,
                                             marker_size=kinematics.end_effector_marker_size,
                                             cube_size=max(.1, kinematics.end_effector_marker_size))
        self.tool_observation = None
        self.latest_tool_estimate = None
        self.tool_pose_source = ""
        self.latest_tool_pose = None
        self.latest_tool_pose_timestamp = None
        self.settings = controller_config or ControllerConfig()
        if tool_clearance is not None:
            self.settings = replace(self.settings, tool_clearance=tool_clearance)
        self.limits_config = config or MPCConfig()
        self.control_dt = self.limits_config.dt
        self.mpc = (JointSpaceMPC(self.kinematics, self.limits_config)
                    if self.settings.mode == "pbvs_mpc" else None)
        marker_from_tool = np.eye(4)
        yaw = np.deg2rad(self.settings.tool_yaw_deg)
        marker_from_tool[:3, :3] = np.array([
            [np.cos(yaw), -np.sin(yaw), 0],
            [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]]) @ np.diag([1, -1, -1])
        marker_from_tool[:3, 3] = [0, 0, self.settings.tool_clearance]
        self.pbvs = PBVS(inverse_pose(marker_from_tool),
                         linear_speed=self.settings.approach_linear_speed_m_s,
                         angular_speed=self.settings.approach_angular_speed_rad_s)

    def update(self, frame: RGBDFrame | None = None) -> ControlResult:
        """Acquire through the interface and command one control interval.

        A caller displaying the camera can pass an already acquired interface
        frame to avoid capturing twice. Joint state is read after image detection
        so kinematics and command limits use the same recent encoder snapshot.
        """
        if frame is None:
            frame = self.interface.capture_rgbd()
        observation = self.processor.process(frame)
        self.tool_observation = None
        self.latest_tool_estimate = None
        self.tool_pose_source = ""
        self.latest_tool_pose = None
        self.latest_tool_pose_timestamp = None
        if observation is not None and self.settings.tool_pose_source != "fk":
            self.tool_observation = self.tool_processor.process_marker(frame)
        state = self.interface.read_state()
        self.kinematics.synchronize(state)
        now = self.interface.timestamp
        if self.estimator.timestamp is not None and now < self.estimator.timestamp:
            self.estimator.reset()  # Simulation/interface clock was restarted.
        if self.tool_estimator.timestamp is not None and now < self.tool_estimator.timestamp:
            self.tool_estimator.reset()
        out_of_order = (self.estimator.config.enabled and self.estimator.timestamp is not None
                        and frame.timestamp < self.estimator.timestamp)
        stale_frame = abs(now - frame.timestamp) > self.settings.frame_max_age_intervals * self.control_dt or out_of_order
        stale_state = abs(now - state.timestamp) > 2 * self.control_dt
        self.pose_estimate = None
        if self.estimator.config.enabled and not stale_frame and not stale_state:
            self.pose_estimate = self.estimator.step(frame, observation)
        if observation is None or stale_frame or stale_state:
            self.interface.set_joint_velocities(np.zeros(6))
            if self.mpc is not None:
                self.mpc.reset()
            return ControlResult(observation, np.zeros(6), np.zeros(6), "target_lost_or_stale",
                                 np.inf, np.inf, np.inf,
                                 diagnostic=("stale_rgbd_frame" if stale_frame
                                             else "stale_robot_state" if stale_state
                                             else self.processor.last_status))
        if self.estimator.config.enabled and (self.pose_estimate is None
                                             or not self.pose_estimate.measurement_accepted):
            self.interface.set_joint_velocities(np.zeros(6))
            if self.mpc is not None:
                self.mpc.reset()
            return ControlResult(observation, np.zeros(6), np.zeros(6), "target_lost_or_stale",
                                 np.inf, np.inf, np.inf,
                                 diagnostic="ekf_" + self.estimator.last_status,
                                 pose_estimate=self.pose_estimate)
        positions = state.joint_positions
        minimum = (float(np.min(self.kinematics.distances(positions), initial=np.inf))
                   if self.mpc is not None else np.nan)
        if self.mpc is not None and minimum < self.mpc.config.clearance:
            self.interface.set_joint_velocities(np.zeros(6))
            self.mpc.reset()
            return ControlResult(observation, np.zeros(6), np.zeros(6), "clearance_violation",
                                 np.inf, np.inf, minimum)
        jacobian = self.kinematics.tool_jacobian(positions, local=True)
        camera_from_marker = observation.camera_from_marker
        if self.pose_estimate is not None:
            cube_from_marker = self.processor.cube_from_markers[self.processor.reference_marker_id]
            camera_from_marker = self.pose_estimate.camera_from_cube @ cube_from_marker
        # Both tags are measured in this same exposure. Their relative pose
        # cancels camera-to-world extrinsics; the rigid tool-tag mounting
        # calibration converts the tool tag frame into the flange frame.
        tool_diagnostic = self.tool_processor.last_status
        if self.tool_tracker is not None:
            world_from_tool = self.tool_tracker.select(frame, self.tool_observation, state.timestamp)
            self.tool_pose_source = self.tool_tracker.source
            if world_from_tool is None:
                self.interface.set_joint_velocities(np.zeros(6))
                if self.mpc is not None:
                    self.mpc.reset()
                return ControlResult(observation, np.zeros(6), np.zeros(6), 'tool_lost_or_rejected',
                    np.inf, np.inf, minimum, diagnostic=self.tool_tracker.diagnostic,
                    pose_estimate=self.pose_estimate, tool_pose_source='unavailable')
            camera_from_tool = inverse_pose(frame.world_from_camera) @ world_from_tool
        elif self.tool_observation is not None and self.settings.tool_pose_source != "fk":
            camera_from_tool = (self.tool_observation.camera_from_marker
                                @ inverse_pose(self.kinematics.tool_from_end_effector_marker))
            if self.tool_estimator.config.enabled:
                tool_estimate = self.tool_estimator.update(
                    frame.timestamp, camera_from_tool, frame.world_from_camera)
                self.latest_tool_estimate = tool_estimate
                if tool_estimate is not None and tool_estimate.measurement_accepted:
                    # This generic rigid-body tracker retains cube-named pose
                    # fields for compatibility; here its tracked body is the tool.
                    camera_from_tool = tool_estimate.camera_from_cube
                else:
                    self.tool_observation = None
                    tool_diagnostic = "tool_ekf_" + self.tool_estimator.last_status
        if self.tool_tracker is not None:
            pass
        elif self.tool_observation is not None and self.settings.tool_pose_source != "fk":
            self.tool_pose_source = "tag"
        elif self.settings.tool_pose_source == "tag_only":
            self.interface.set_joint_velocities(np.zeros(6))
            if self.mpc is not None:
                self.mpc.reset()
            return ControlResult(observation, np.zeros(6), np.zeros(6), "tool_lost_or_rejected",
                np.inf, np.inf, minimum, diagnostic=tool_diagnostic,
                pose_estimate=self.pose_estimate, tool_pose_source="unavailable")
        else:
            camera_from_tool = inverse_pose(frame.world_from_camera) @ self.kinematics.tool_pose()
            self.tool_pose_source = "fk" if self.settings.tool_pose_source == "fk" else "fk_fallback"
        self.latest_tool_pose = frame.world_from_camera @ camera_from_tool
        self.latest_tool_pose_timestamp = self.interface.timestamp
        tool_from_marker = inverse_pose(camera_from_tool) @ camera_from_marker
        reference, error = self.pbvs.reference(tool_from_marker, jacobian)
        position_error, rotation_error = np.linalg.norm(error[:3]), np.linalg.norm(error[3:])
        if position_error < 0.003 and rotation_error < 0.025:
            self.interface.set_joint_velocities(np.zeros(6))
            if self.mpc is not None:
                self.mpc.reset()
            return ControlResult(observation, reference, np.zeros(6), "reached",
                                 position_error, rotation_error, minimum, pose_estimate=self.pose_estimate,
                                 tool_observation=self.tool_observation, tool_pose_source=self.tool_pose_source,
                                 tool_pose=self.latest_tool_pose.copy(),
                                 tool_pose_timestamp=self.latest_tool_pose_timestamp)
        command, status, minimum = pose_reference_command(self.kinematics, self.limits_config,
            self.mpc, positions, state.joint_velocities, reference)
        self.interface.set_joint_velocities(command)
        return ControlResult(observation, reference, command, status,
                             position_error, rotation_error, minimum,
                             pose_estimate=self.pose_estimate, tool_observation=self.tool_observation,
                             tool_pose_source=self.tool_pose_source,
                             tool_pose=self.latest_tool_pose.copy(),
                             tool_pose_timestamp=self.latest_tool_pose_timestamp)


def pose_reference_command(kinematics, config, mpc, positions, velocities, reference):
    """Shared PBVS joint command policy for approach and transfer."""
    if mpc is not None:
        solution = mpc.solve(positions, velocities, reference)
        return solution.velocity, solution.status, solution.minimum_clearance
    limit = config.velocity_limit
    command = reference / max(1., np.max(np.abs(reference)) / limit)
    lower, upper = kinematics.limits
    if np.any(positions < lower) or np.any(positions > upper):
        return np.zeros(6), 'joint_limit_violation', np.nan
    command = np.clip(command, np.maximum(-limit, (lower - positions) / config.dt),
                      np.minimum(limit, (upper - positions) / config.dt))
    return command, 'tracking', np.nan


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

    def __init__(self, interface: "RobotInterface", kinematics: Kinematics, *,
                 cube_size: float | None = None, speed: float = 0.03, angular_speed: float = 0.3,
                 control_dt: float = 0.05, phase_timeout: float = 20.0,
                 velocity_limit: float = 0.35, clearance: float = 0.015,
                 frame_max_age_intervals: float = 2.0, grasp_height_offset_m: float = 0.0,
                 pickup_dwell_seconds: float = 0.0, pickup_dwell_radius_m: float = 0.01,
                 tool_tracker=None):
        self.tool_tracker = tool_tracker
        self.latest_tool_pose = None
        self.latest_tool_pose_timestamp = None
        if cube_size is None:
            cube_size = getattr(kinematics, 'cube_size', .1)
        for name, value, positive in (("pickup_dwell_seconds", pickup_dwell_seconds, False),
                                      ("pickup_dwell_radius_m", pickup_dwell_radius_m, True)):
            if (isinstance(value, bool) or not isinstance(value, (int, float))
                    or not np.isfinite(value) or value < 0 or (positive and value == 0)):
                raise ValueError(f"{name} must be finite and {'positive' if positive else 'nonnegative'}")
        self.pickup_dwell_seconds = pickup_dwell_seconds
        self.pickup_dwell_radius_m = pickup_dwell_radius_m
        self._dwell_since = None
        self._cube_center = None
        if (isinstance(grasp_height_offset_m, bool)
                or not isinstance(grasp_height_offset_m, (int, float))
                or not np.isfinite(grasp_height_offset_m)):
            raise ValueError("grasp_height_offset_m must be a finite number")
        self.grasp_height_offset_m = grasp_height_offset_m
        if (isinstance(frame_max_age_intervals, bool)
                or not isinstance(frame_max_age_intervals, (int, float))
                or not np.isfinite(frame_max_age_intervals) or frame_max_age_intervals <= 0):
            raise ValueError("frame_max_age_intervals must be finite and positive")
        self.frame_max_age_intervals = frame_max_age_intervals
        for name, value in (("cube_size", cube_size), ("speed", speed), ("angular_speed", angular_speed),
                            ("control_dt", control_dt), ("phase_timeout", phase_timeout),
                            ("velocity_limit", velocity_limit), ("clearance", clearance)):
            if (isinstance(value, bool) or not isinstance(value, (int, float))
                    or not np.isfinite(value) or value <= 0):
                raise ValueError(f"{name} must be finite and positive")
        self.interface = interface
        self.kinematics = kinematics
        self.cube_size = cube_size
        self.speed = speed
        self.angular_speed = angular_speed
        self.control_dt = control_dt
        self.phase_timeout = phase_timeout
        self.velocity_limit, self.clearance = velocity_limit, clearance
        self.phase = "idle"
        self.diagnostic = ""
        self._phase_started = 0.0
        self._width_history = []
        self._entry_pose = None
        self._descent_pose = None
        self._pose_integral = np.zeros(6)
        self._hold_positions = None
        self.tool_from_cube = None
        self._world_from_cube = None
        self._handoff_centering_rejected = False
        self._reapproach_attempts = 0

        # Existing prediction pairs remain untouched. Only the two finger/cube
        # pairs may make contact during pickup; the palm must stay clear.
        self._collision_indices = kinematics.pickup_collision_indices()

    @property
    def finished(self) -> bool:
        return self.phase in ("complete", "failed")

    def retry_centering(self, maximum):
        """Reset only a rejected, pre-gripper centering handoff for fresh vision."""
        if (self.phase != "failed" or not self._handoff_centering_rejected
                or self._reapproach_attempts >= maximum):
            return False
        self.interface.set_joint_velocities(np.zeros(6))
        self._reapproach_attempts += 1
        self._handoff_centering_rejected = False
        self._entry_pose = self._descent_pose = self._cube_center = self._world_from_cube = None
        self._dwell_since = None
        self._hold_positions = self.tool_from_cube = None
        self._change_phase("idle", self.interface.timestamp)
        return True

    def start(self, observation: CubeObservation) -> None:
        """Latch the final approach pose and the observed cube center."""
        if self.phase != "idle":
            raise RuntimeError("pickup has already started; construct a new controller to retry")
        state = self.interface.read_state()
        now = self.interface.timestamp
        if (observation is None or not np.isfinite(now)
                or not np.isfinite(observation.timestamp) or not np.isfinite(state.timestamp)
                or abs(now - observation.timestamp) > self.frame_max_age_intervals * self.control_dt
                or abs(now - state.timestamp) > 2 * self.control_dt):
            self._fail("stale_pickup_target_or_state")
            return
        cube_pose = np.asarray(observation.world_from_cube, dtype=float)
        if (cube_pose.shape != (4, 4) or not np.all(np.isfinite(cube_pose))
                or not np.allclose(cube_pose[3], [0, 0, 0, 1], atol=1e-8)
                or not np.allclose(cube_pose[:3, :3].T @ cube_pose[:3, :3], np.eye(3), atol=1e-8)
                or not np.isclose(np.linalg.det(cube_pose[:3, :3]), 1., atol=1e-8)
                or not np.all(np.isfinite(state.joint_positions))
                or not np.isfinite(state.gripper_width)):
            self._fail("invalid_pickup_target_or_state")
            return
        self.kinematics.synchronize(state)
        if self.tool_tracker is not None and self.tool_tracker.pose is None:
            self._fail('tool_lost_or_rejected')
            return
        self._entry_pose = self._tool_pose().copy()
        self.latest_tool_pose = self._entry_pose.copy()
        self.latest_tool_pose_timestamp = now
        self._descent_pose = self._entry_pose.copy()
        grasp_position = self._grasp_pose()[:3, 3]
        # Move along the latched cube normal, preserving local XY and the
        # entry tool orientation. The grasp offset is measured along cube +Z.
        cube_center = cube_pose[:3, 3]
        self._cube_center = cube_center.copy()
        self._world_from_cube = cube_pose.copy()
        grasp_in_cube = cube_pose[:3, :3].T @ (grasp_position - cube_center)
        target_grasp = grasp_position
        # Articulated simulated fingers change pad depth as they close.
        width_pose = getattr(type(self.kinematics), 'grasp_pose_at_width', None)
        if width_pose is not None:
            target_grasp = (self.tool_tracker.grasp_pose(self.cube_size) if self.tool_tracker is not None
                            else self.kinematics.grasp_pose_at_width(self.cube_size))[:3, 3]
        target_grasp_z = (cube_pose[:3, :3].T @ (target_grasp - cube_center))[2]
        self._descent_pose[:3, 3] += cube_pose[:3, 2] * (self.grasp_height_offset_m - target_grasp_z)
        horizontal_error = grasp_in_cube[:2]
        horizontal_distance = np.linalg.norm(horizontal_error)
        centering_limit = self.pickup_dwell_radius_m if self.pickup_dwell_seconds > 0 else .01
        if horizontal_distance > centering_limit:
            self._handoff_centering_rejected = True
            self._fail(
                "tool_not_centered_over_cube: "
                f"dx={horizontal_error[0] * 1000:.1f} mm, "
                f"dy={horizontal_error[1] * 1000:.1f} mm, "
                f"distance={horizontal_distance * 1000:.1f} mm, limit={centering_limit * 1000:.1f} mm; frame=cube; "
                f"grasp_xy_mm={np.round(grasp_position[:2] * 1000, 1).tolist()}, "
                f"cube_xy_mm={np.round(cube_center[:2] * 1000, 1).tolist()}"
            )
            return
        if grasp_in_cube[2] <= self.grasp_height_offset_m:
            self._fail("cube_not_below_gripper")
            return
        if self.interface.gripper_max_width < self.cube_size + 0.01:
            self._fail("gripper_opening_too_small")
            return
        self.interface.set_joint_velocities(np.zeros(6))
        self.interface.open_gripper()
        self._change_phase("opening", now)

    def update(self) -> ControlResult:
        """Advance one phase with fresh tool feedback when fusion is enabled."""
        self.latest_tool_pose = None
        self.latest_tool_pose_timestamp = None
        tool_available = True
        if self.tool_tracker is not None and self.phase != 'idle' and not self.finished:
            tool_available = self.tool_tracker.refresh() is not None
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
        phase_budget = self.phase_timeout + (self.pickup_dwell_seconds if self.phase == "dwelling" else 0.)
        if now - self._phase_started > phase_budget:
            detail = ""
            if self.phase == "opening":
                detail = "; " + self._opening_diagnostic(state.gripper_width)
            elif self.phase == "closing":
                width, settled = self._gripper_settling(now, state.gripper_width)
                detail = "; " + self._closing_diagnostic(state.gripper_width, width, settled)
            self._fail("phase_timeout: " + self.phase + detail)
            return self._result()
        self.kinematics.synchronize(state)

        if self.phase == 'lifting' and state.gripper_width < self.cube_size * 0.5:
            self._fail('grasp_lost')
            return self._result()
        if not tool_available:
            self.interface.set_joint_velocities(np.zeros(6))
            self._dwell_since = None
            self._pose_integral[:] = 0
            self.diagnostic = 'tool_lost_or_rejected: ' + self.tool_tracker.diagnostic
            return self._result()
        self.latest_tool_pose = self._tool_pose().copy()
        self.latest_tool_pose_timestamp = now

        if self.phase == "opening":
            self.interface.set_joint_velocities(np.zeros(6))
            self.diagnostic = self._opening_diagnostic(state.gripper_width)
            if state.gripper_width >= self.interface.gripper_max_width - 0.001:
                self._change_phase("dwelling" if self.pickup_dwell_seconds > 0 else "descending", now)
            return self._result()

        if self.phase == "dwelling":
            self.interface.set_joint_velocities(np.zeros(6))
            grasp = self._grasp_pose()[:3, 3]
            grasp_in_cube = self._world_from_cube[:3, :3].T @ (grasp - self._cube_center)
            distance = np.linalg.norm(grasp_in_cube[:2])
            if grasp_in_cube[2] <= self.grasp_height_offset_m:
                self._fail("dwell_grasp_not_above_target")
                return self._result()
            ready = (distance <= self.pickup_dwell_radius_m
                     and state.gripper_width >= self.interface.gripper_max_width - .001)
            if not ready:
                self._dwell_since = None
            elif self._dwell_since is None:
                self._dwell_since = now
            elapsed = 0. if self._dwell_since is None else now-self._dwell_since
            self.diagnostic = (f"frame=cube, xy_distance={distance*1000:.1f} mm, radius={self.pickup_dwell_radius_m*1000:.1f} mm, "
                               f"dwell={elapsed:.2f}/{self.pickup_dwell_seconds:.2f} s")
            if ready and elapsed >= self.pickup_dwell_seconds:
                self._change_phase("descending", now)
            return self._result()

        if self.phase == "closing":
            self.interface.set_joint_velocities(np.zeros(6))
            width, settled = self._gripper_settling(now, state.gripper_width)
            self.diagnostic = self._closing_diagnostic(state.gripper_width, width, settled)
            elapsed = now - self._phase_started
            if elapsed >= 0.5:
                if width < self.cube_size * 0.5:
                    self._fail("empty_grasp")
                elif width <= self.cube_size * 1.15 and settled:
                    self.tool_from_cube = np.linalg.inv(self._tool_pose()) @ self._world_from_cube
                    self._change_phase("lifting", now)
            return self._result()

        target = self._descent_pose if self.phase == "descending" else self._entry_pose
        current = self._tool_pose()
        error = target[:3, 3] - current[:3, 3]
        rotation_error = cv2.Rodrigues(target[:3, :3] @ current[:3, :3].T)[0].reshape(3)
        if np.linalg.norm(error) < 0.0015 and np.linalg.norm(rotation_error) < 0.02:
            self.interface.set_joint_velocities(np.zeros(6))
            if self.phase == "descending":
                self.interface.close_gripper()
                self._change_phase("closing", now)
                self._gripper_settling(now, state.gripper_width)
            else:
                # Hold a fixed joint position after lifting, including the
                # payload load, instead of leaving zero-velocity servos free
                # to drift under the cube's external contact forces.
                self._hold_positions = state.joint_positions.copy()
                self.interface.set_joint_positions(self._hold_positions)
                self._change_phase("complete", now)
            return self._result(position_error=np.linalg.norm(error))

        jacobian = self.kinematics.tool_jacobian(state.joint_positions)
        # Integrate and limit translation in the latched cube frame so that
        # lateral correction and travel along the normal remain independent.
        world_from_cube_rotation = self._world_from_cube[:3, :3]
        pose_error = np.concatenate((world_from_cube_rotation.T @ error, rotation_error))
        # Integrate translation only near the target to avoid winding up
        # while normal-axis speed is saturated over a long descent/lift.
        integration_error = pose_error.copy()
        if np.linalg.norm(error) > self.speed / 8:
            integration_error[2] = 0
        self._pose_integral = np.clip(
            self._pose_integral + self.control_dt * integration_error, -0.02, 0.02
        )
        twist = 8.0 * pose_error + 4.0 * self._pose_integral
        linear_velocity = twist[:3]
        # Cap cube XY correction and cube Z travel separately.
        linear_velocity[:2] *= min(1.0, self.speed / max(np.linalg.norm(linear_velocity[:2]), 1e-12))
        linear_velocity[2] = np.clip(linear_velocity[2], -self.speed, self.speed)
        twist[:3] = world_from_cube_rotation @ linear_velocity
        twist[3:] *= min(1.0, self.angular_speed / max(np.linalg.norm(twist[3:]), 1e-12))
        command = jacobian.T @ np.linalg.solve(jacobian @ jacobian.T + 0.01 ** 2 * np.eye(6), twist)
        command /= max(1.0, np.max(np.abs(command)) / self.velocity_limit)
        lower, upper = self.kinematics.limits
        prediction = state.joint_positions + self.control_dt * command
        if np.any(prediction < lower) or np.any(prediction > upper):
            self._fail("joint_limit_violation")
            return self._result()
        current_clearance, current_pair = self._clearance(state.joint_positions)
        predicted_clearance, predicted_pair = self._clearance(prediction)
        minimum = min(current_clearance, predicted_clearance)
        if minimum < self.clearance:
            at_current = current_clearance <= predicted_clearance
            pair = current_pair if at_current else predicted_pair
            self._fail(
                f"pickup_path_blocked: {self.kinematics.collision_pair_name(pair)}; "
                f"at={'current' if at_current else 'predicted'}, "
                f"clearance={minimum * 1000:.1f} mm, required={self.clearance * 1000:.1f} mm; "
                f"current={current_clearance * 1000:.1f} mm, "
                f"predicted={predicted_clearance * 1000:.1f} mm; "
                f"joint_positions_rad={state.joint_positions.tolist()}"
            )
            return self._result(minimum_clearance=minimum)
        self.interface.set_joint_velocities(command)
        return self._result(command, np.linalg.norm(error), minimum)

    def _tool_pose(self):
        return self.kinematics.tool_pose() if self.tool_tracker is None else self.tool_tracker.pose

    def _grasp_pose(self):
        return self.kinematics.grasp_pose() if self.tool_tracker is None else self.tool_tracker.grasp_pose()

    def _clearance(self, positions) -> tuple[float, int | None]:
        distances = self.kinematics.distances(positions)
        if not self._collision_indices:
            return float('inf'), None
        index = self._collision_indices[int(np.argmin(distances[self._collision_indices]))]
        return float(distances[index]), index

    def _gripper_settling(self, now, width) -> tuple[float, bool]:
        """Estimate finger motion over 0.5 s instead of comparing noisy pairs.

        A median width supports the grasp check. A fitted width trend below
        2 mm/s indicates settling, even when individual encoder samples jitter.
        Repeated updates at one timestamp do not count as new measurements.
        """
        if not self._width_history or now > self._width_history[-1][0]:
            self._width_history.append((now, width))
        # Keep the sample bracketing the start of the window. Otherwise an
        # irregular sample cadence can never span the required full 0.5 s.
        while len(self._width_history) > 1 and self._width_history[1][0] <= now - 0.5:
            self._width_history.pop(0)
        times, widths = np.asarray(self._width_history).T
        measured_width = float(np.median(widths))
        if (len(times) < 5 or times[-1] - times[0] < 0.5 - 1e-9
                or np.max(np.diff(times)) > max(.1, 2*self.control_dt) + 1e-9):
            return measured_width, False
        centered_times = times - times.mean()
        width_speed = float(centered_times @ (widths - widths.mean())
                            / (centered_times @ centered_times))
        return measured_width, abs(width_speed) <= 0.002

    def _closing_diagnostic(self, raw_width, width, settled) -> str:
        return (f"gripper_width={raw_width * 1000:.1f} mm, median={width * 1000:.1f} mm, "
                f"expected={self.cube_size * 500:.1f}..{self.cube_size * 1150:.1f} mm, "
                f"settled={settled}")

    def _opening_diagnostic(self, width) -> str:
        return (f"gripper_width={width * 1000:.1f} mm, "
                f"requested={self.interface.gripper_max_width * 1000:.1f} mm, "
                f"required>={(self.interface.gripper_max_width - .001) * 1000:.1f} mm")

    def _change_phase(self, phase, now) -> None:
        self.phase = phase
        self.diagnostic = ""
        self._phase_started = now
        self._pose_integral[:] = 0
        self._width_history.clear()

    def _fail(self, reason) -> None:
        self.phase = "failed"
        self.diagnostic = reason
        self.interface.set_joint_velocities(np.zeros(6))

    def _result(self, command=None, position_error=0.0, minimum_clearance=np.nan) -> ControlResult:
        velocity = np.zeros(6) if command is None else command.copy()
        return ControlResult(None, velocity, velocity.copy(), "pickup_" + self.phase,
                             position_error, 0.0, minimum_clearance, self.diagnostic)


def start_pickup_or_reapproach(approach, pickup, observation):
    """Shared handoff policy for hardware, sequence, and simulation runners."""
    pickup.start(observation)
    if pickup.phase == "failed" and pickup.diagnostic.startswith("tool_not_centered_over_cube:"):
        reason = pickup.diagnostic
        maximum = approach.settings.pickup_reapproach_attempts
        if pickup.retry_centering(maximum):
            if hasattr(approach.interface, "require_camera_for_motion"):
                approach.interface.require_camera_for_motion = True
            if approach.mpc is not None:
                approach.mpc.reset()
            return replace(pickup._result(), status="reapproaching",
                           diagnostic=f"{reason}; returning to approach ({pickup._reapproach_attempts}/{maximum})")
    return pickup.update()
