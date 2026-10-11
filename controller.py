"""PBVS joint-velocity references tracked by obstacle-constrained joint MPC.

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
import mujoco
import numpy as np
from scipy.optimize import minimize

from camera_processing import CameraProcessor, CubeObservation

if TYPE_CHECKING:
    from interface import RobotInterface
    from robot_model import RobotModel, RGBDFrame


def inverse_pose(transform):
    result = np.eye(4)
    result[:3, :3] = transform[:3, :3].T
    result[:3, 3] = -result[:3, :3] @ transform[:3, 3]
    return result


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


class MujocoKinematics:
    """Independent model workspace for Jacobians and known-scene obstacles.

    All collision-enabled environment geoms and nonadjacent robot link pairs
    are checked. The shoulder/floor pair is omitted because this anchored
    joint only rotates about the vertical axis and has a fixed 3 mm mounting
    gap. Same-body and adjacent-body geometries intentionally overlap.
    Hardware use requires a calibrated kinematic model and obstacle map.
    """

    def __init__(self, robot: "RobotModel"):
        self.robot = robot
        self.model = robot.model
        self.data = mujoco.MjData(self.model)
        self.q_indices = self.model.jnt_qposadr[robot.joint_ids]
        self.v_indices = self.model.jnt_dofadr[robot.joint_ids]
        self.camera_id = robot.camera_id
        self.tool_id = self.model.site("attachment_site").id
        self.limits = np.maximum(self.model.jnt_range[robot.joint_ids, 0],
                                 self.model.actuator_ctrlrange[robot.actuator_ids, 0]), np.minimum(
                                     self.model.jnt_range[robot.joint_ids, 1],
                                     self.model.actuator_ctrlrange[robot.actuator_ids, 1])
        base = self.model.body("base").id
        robot_bodies = {base}
        for body in range(base + 1, self.model.nbody):
            if int(self.model.body_parentid[body]) in robot_bodies:
                robot_bodies.add(body)
        collision = [g for g in range(self.model.ngeom)
                     if self.model.geom_contype[g] or self.model.geom_conaffinity[g]]
        moving = [g for g in collision if int(self.model.geom_bodyid[g]) in robot_bodies
                  and int(self.model.geom_bodyid[g]) != base]
        environment = [g for g in collision if int(self.model.geom_bodyid[g]) not in robot_bodies]
        floor = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        shoulder = self.model.body("shoulder_link").id
        self.pairs = [(a, b) for a in moving for b in environment
                      if not (b == floor and self.model.geom_bodyid[a] == shoulder)]
        for i, a in enumerate(moving):
            for b in moving[i + 1:]:
                ba, bb = int(self.model.geom_bodyid[a]), int(self.model.geom_bodyid[b])
                if ba != bb and self.model.body_parentid[ba] != bb and self.model.body_parentid[bb] != ba:
                    self.pairs.append((a, b))
        self.set_positions(robot.joint_positions)
        self.tool_from_camera = inverse_pose(self.tool_pose()) @ self.camera_pose()

    def set_positions(self, positions):
        self.data.qpos[:] = self.robot.data.qpos
        self.data.qpos[self.q_indices] = positions
        mujoco.mj_forward(self.model, self.data)

    def camera_pose(self):
        pose = np.eye(4)
        pose[:3, :3] = self.data.cam_xmat[self.camera_id].reshape(3, 3) @ np.diag([1, -1, -1])
        pose[:3, 3] = self.data.cam_xpos[self.camera_id]
        return pose

    def tool_pose(self):
        pose = np.eye(4)
        pose[:3, :3] = self.data.site_xmat[self.tool_id].reshape(3, 3)
        pose[:3, 3] = self.data.site_xpos[self.tool_id]
        return pose

    def camera_jacobian(self, positions):
        self.set_positions(positions)
        translation, rotation = np.zeros((3, self.model.nv)), np.zeros((3, self.model.nv))
        mujoco.mj_jac(self.model, self.data, translation, rotation,
                      self.data.cam_xpos[self.camera_id], int(self.model.cam_bodyid[self.camera_id]))
        camera_from_world_rotation = self.camera_pose()[:3, :3].T
        return np.vstack((camera_from_world_rotation @ translation[:, self.v_indices],
                          camera_from_world_rotation @ rotation[:, self.v_indices]))

    def distances(self, positions, *, gradients=False):
        self.set_positions(positions)
        distances = np.empty(len(self.pairs))
        derivatives = np.zeros((len(self.pairs), 6))
        segment = np.zeros(6)
        ja, jb = np.zeros((3, self.model.nv)), np.zeros((3, self.model.nv))
        for i, (a, b) in enumerate(self.pairs):
            distances[i] = mujoco.mj_geomDistance(self.model, self.data, a, b, 2.0, segment)
            if gradients and 0 < distances[i] < 2.0:
                direction = segment[3:] - segment[:3]
                length = np.linalg.norm(direction)
                if length > 1e-9:
                    mujoco.mj_jac(self.model, self.data, ja, None, segment[:3], int(self.model.geom_bodyid[a]))
                    mujoco.mj_jac(self.model, self.data, jb, None, segment[3:], int(self.model.geom_bodyid[b]))
                    derivatives[i] = (direction / length) @ (jb - ja)[:, self.v_indices]
        return (distances, derivatives) if gradients else distances


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


class ApproachController:
    """Detect the cube and move the tool above its tagged face, without grasping."""

    def __init__(self, robot: "RobotModel", interface: "RobotInterface", *,
                 tool_clearance: float = 0.18, config: MPCConfig | None = None):
        if not np.isfinite(tool_clearance) or tool_clearance < 0.16:
            raise ValueError("tool_clearance must be at least 0.16 m to keep the offset camera's tag in view")
        self.interface = interface
        self.processor = CameraProcessor()
        self.kinematics = MujocoKinematics(robot)
        self.mpc = JointSpaceMPC(self.kinematics, config)
        marker_from_tool = np.eye(4)
        marker_from_tool[:3, :3] = np.diag([1, -1, -1])
        marker_from_tool[:3, 3] = [0, 0, tool_clearance]
        marker_from_camera = marker_from_tool @ self.kinematics.tool_from_camera
        self.pbvs = PBVS(inverse_pose(marker_from_camera))

    def update(self, frame: "RGBDFrame") -> ControlResult:
        observation = self.processor.process(frame)
        if observation is None or abs(float(self.kinematics.robot.data.time) - frame.timestamp) > 2 * self.mpc.config.dt:
            self.interface.set_joint_velocities(np.zeros(6))
            self.mpc.reset()
            return ControlResult(observation, np.zeros(6), np.zeros(6), "target_lost_or_stale",
                                 np.inf, np.inf, np.inf)
        positions = self.interface.joint_positions
        minimum = float(np.min(self.kinematics.distances(positions), initial=np.inf))
        if minimum < self.mpc.config.clearance:
            self.interface.set_joint_velocities(np.zeros(6))
            self.mpc.reset()
            return ControlResult(observation, np.zeros(6), np.zeros(6), "clearance_violation",
                                 np.inf, np.inf, minimum)
        jacobian = self.kinematics.camera_jacobian(positions)
        reference, error = self.pbvs.reference(observation.camera_from_marker, jacobian)
        position_error, rotation_error = np.linalg.norm(error[:3]), np.linalg.norm(error[3:])
        if position_error < 0.003 and rotation_error < 0.025:
            self.interface.set_joint_velocities(np.zeros(6))
            self.mpc.reset()
            return ControlResult(observation, reference, np.zeros(6), "reached",
                                 position_error, rotation_error, minimum)
        solution = self.mpc.solve(positions, self.interface.joint_velocities, reference)
        self.interface.set_joint_velocities(solution.velocity)
        return ControlResult(observation, reference, solution.velocity, solution.status,
                             position_error, rotation_error, solution.minimum_clearance)
