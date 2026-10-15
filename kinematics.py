"""Prediction-only kinematics and collision geometry, independent of live I/O."""

from typing import TYPE_CHECKING, Protocol

import numpy as np

from robot_state import RobotState

if TYPE_CHECKING:
    from robot_model import RobotModel


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


class MujocoKinematics:
    """Independent model workspace for Jacobians and known-scene obstacles.

    All collision-enabled environment geoms and nonadjacent robot link pairs
    are checked. The shoulder/floor pair is omitted because this anchored
    joint only rotates about the vertical axis and has a fixed 3 mm mounting
    gap. Same-body and adjacent-body geometries intentionally overlap.
    Hardware use requires a calibrated kinematic model and obstacle map.
    """

    def __init__(self, robot: "RobotModel"):
        import mujoco

        # Only the model definition is shared. This workspace never reads the
        # simulator's live data; measured state arrives through synchronize().
        self.model = robot.model
        self.data = mujoco.MjData(self.model)
        self.q_indices = self.model.jnt_qposadr[robot.joint_ids]
        self.v_indices = self.model.jnt_dofadr[robot.joint_ids]
        self.camera_id = robot.camera_id
        self.gripper_indices = self.model.jnt_qposadr[robot.gripper_joint_ids]
        self.tool_id = self.model.site("attachment_site").id
        self.end_effector_marker_id = robot.end_effector_marker_id
        self.end_effector_marker_size = robot.end_effector_marker_size
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
                # Fixed gripper mounts share the wrist's welded body. Respect
                # model contact exclusions (e.g. the intentionally touching fingers).
                wa, wb = int(self.model.body_weldid[ba]), int(self.model.body_weldid[bb])
                signature = (min(ba, bb) << 16) + max(ba, bb)
                adjacent = (self.model.body_weldid[self.model.body_parentid[wa]] == wb
                            or self.model.body_weldid[self.model.body_parentid[wb]] == wa)
                if wa != wb and not adjacent and signature not in self.model.exclude_signature:
                    self.pairs.append((a, b))
        mujoco.mj_resetDataKeyframe(self.model, self.data, self.model.key("home").id)
        mujoco.mj_forward(self.model, self.data)
        # Rigid marker-to-flange calibration from the model definition. Hardware
        # must provide the measured mounting transform in its kinematic model.
        marker_site = self.data.site("end_effector_marker_site")
        world_from_marker = np.eye(4)
        world_from_marker[:3, :3] = marker_site.xmat.reshape(3, 3)
        world_from_marker[:3, 3] = marker_site.xpos
        self.tool_from_end_effector_marker = inverse_pose(self.tool_pose()) @ world_from_marker

    def synchronize(self, state: RobotState) -> None:
        """Set the prediction starting state from interface measurements.

        The parallel gripper is modeled symmetrically from its measured width.
        Its opening stays fixed during arm predictions, not copied from physics.
        """
        self.data.qpos[self.gripper_indices] = state.gripper_width / 2
        self.set_positions(state.joint_positions)

    def set_positions(self, positions):
        import mujoco

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

    def grasp_pose(self):
        site = self.data.site("grasp_site")
        pose = np.eye(4)
        pose[:3, :3] = site.xmat.reshape(3, 3)
        pose[:3, 3] = site.xpos
        return pose

    def pickup_collision_indices(self):
        """Keep all constraints except intended finger/target contact."""
        cube_body = self.model.body("cube").id
        cube_geoms = set(np.flatnonzero(self.model.geom_bodyid == cube_body))
        fingers = {self.model.geom(name).id for name in (
            "gripper_left_geom", "gripper_right_geom")}
        return [i for i, (a, b) in enumerate(self.pairs)
                if not ((a in fingers and b in cube_geoms)
                        or (b in fingers and a in cube_geoms))]

    def tool_jacobian(self, positions, *, local=False):
        """Flange linear/angular velocity Jacobian in world or tool axes.

        Unlike camera_jacobian, this remains nonzero with a fixed camera.
        Positions come from interface measurements, not live simulator data.
        """
        import mujoco

        self.set_positions(positions)
        translation, rotation = np.zeros((3, self.model.nv)), np.zeros((3, self.model.nv))
        mujoco.mj_jacSite(self.model, self.data, translation, rotation, self.tool_id)
        axes = self.tool_pose()[:3, :3].T if local else np.eye(3)
        return np.vstack((axes @ translation[:, self.v_indices],
                          axes @ rotation[:, self.v_indices]))

    def camera_jacobian(self, positions):
        import mujoco

        self.set_positions(positions)
        translation, rotation = np.zeros((3, self.model.nv)), np.zeros((3, self.model.nv))
        mujoco.mj_jac(self.model, self.data, translation, rotation,
                      self.data.cam_xpos[self.camera_id], int(self.model.cam_bodyid[self.camera_id]))
        camera_from_world_rotation = self.camera_pose()[:3, :3].T
        return np.vstack((camera_from_world_rotation @ translation[:, self.v_indices],
                          camera_from_world_rotation @ rotation[:, self.v_indices]))

    def distances(self, positions, *, gradients=False):
        import mujoco

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


