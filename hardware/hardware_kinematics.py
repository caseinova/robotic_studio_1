"""UR3 calculations without MuJoCo, XML, rendering, or robot communication.

Standard DH: A_i = Rz(q_i + offset_i) Tz(d_i) Tx(a_i) Rx(alpha_i).
Nominal UR3 (CB-series, NOT UR3e) parameters are from Universal Robots:
https://www.universal-robots.com/articles/ur/application-installation/dh-parameters-for-calculations-of-kinematics-and-dynamics/

Use the individual arm's calibrated parameters and measured tool transforms
for hardware. Collision shapes are supplied explicitly, not inferred from DH
origins, which are not physical link centerlines. No dynamics are needed by
the current joint-velocity MPC. All lengths are meters and angles radians.
"""

from dataclasses import dataclass

import numpy as np

from common.kinematics import inverse_pose


from common.geometry import _vector, _pose, _positive, LinkCapsule, SphereObstacle, BoxObstacle, PlaneObstacle, _segment_distance, _segment_distances, _segment_box_distance

@dataclass(frozen=True)
class UR3DH:
    """Replace all four arrays with the arm's calibrated standard DH values."""

    a: tuple = (0, -0.24365, -0.21325, 0, 0, 0)
    d: tuple = (0.1519, 0, 0, 0.11235, 0.08535, 0.0819)
    alpha: tuple = (np.pi / 2, 0, 0, np.pi / 2, -np.pi / 2, 0)
    offsets: tuple = (0, 0, 0, 0, 0, 0)

    def __post_init__(self):
        for name in ("a", "d", "alpha", "offsets"):
            object.__setattr__(self, name, _vector(getattr(self, name), 6, name))


class UR3Kinematics:
    """Portable FK, Jacobians, and geometric constraints for the existing MPC.

    Measured q uses RobotInterface joint order and standard-DH encoder zero;
    offsets correct joint zero differences. world_from_base also accounts for
    a different installation/world convention. Joint limits and tool/grasp
    transforms are explicit inputs, never copied from the simulation.

    Collision geometry must be configured before MPC/pickup. Shapes should
    enclose physical links. No default radii or obstacle map are assumed.
    Clearance derivatives use central finite differences; this is a planning
    calculation, not the UR controller's servo or a certified safety system.
    """

    def __init__(self, *, joint_limits, world_from_base, flange_from_tool,
                 tool_from_grasp, tool_from_marker=None, dh=None,
                 link_capsules=(), obstacles=(), excluded_self_pairs=(),
                 marker_id=1, marker_size=0.08, gradient_step=1e-6):
        self.dh = dh if dh is not None else UR3DH()
        if not isinstance(self.dh, UR3DH):
            raise ValueError("dh must be UR3DH")
        if len(joint_limits) != 2:
            raise ValueError("joint_limits must be (lower, upper)")
        self.limits = (_vector(joint_limits[0], 6, "lower limits"),
                       _vector(joint_limits[1], 6, "upper limits"))
        if np.any(self.limits[0] >= self.limits[1]):
            raise ValueError("lower limits must be below upper limits")
        self.world_from_base = _pose(world_from_base, "world_from_base")
        self.flange_from_tool = _pose(flange_from_tool, "flange_from_tool")
        self.tool_from_grasp = _pose(tool_from_grasp, "tool_from_grasp")
        self._tool_from_marker = (None if tool_from_marker is None
                                  else _pose(tool_from_marker, "tool_from_marker"))
        if type(marker_id) is not int or not 0 <= marker_id < 50:
            raise ValueError("marker_id must be in DICT_4X4_50")
        _positive(marker_size, "marker_size")
        _positive(gradient_step, "gradient_step")
        self.end_effector_marker_id = marker_id
        self.end_effector_marker_size = marker_size
        self.gradient_step = gradient_step
        self.links = tuple(link_capsules)
        if any(not isinstance(link, LinkCapsule) for link in self.links):
            raise ValueError("link_capsules must contain LinkCapsule instances")
        if len({link.name for link in self.links}) != len(self.links):
            raise ValueError("link capsule names must be unique")
        self._excluded_self_pairs = {frozenset(pair) for pair in excluded_self_pairs}
        for pair in self._excluded_self_pairs:
            if len(pair) != 2 or not pair <= {link.name for link in self.links}:
                raise ValueError("excluded self pairs must name two configured links")
        self.gripper_width = 0.
        self.set_positions(np.zeros(6))
        self.set_obstacles(obstacles)

    @property
    def tool_from_end_effector_marker(self):
        if self._tool_from_marker is None:
            raise ValueError("tool tag requires a calibrated tool_from_marker transform")
        return self._tool_from_marker.copy()

    def _chain(self, positions):
        q = _vector(positions, 6, "joint positions")
        chain = [self.world_from_base.copy()]
        for theta, a, d, alpha in zip(q + self.dh.offsets, self.dh.a, self.dh.d, self.dh.alpha):
            ct, st, ca, sa = np.cos(theta), np.sin(theta), np.cos(alpha), np.sin(alpha)
            link = np.array([[ct, -st * ca, st * sa, a * ct],
                             [st, ct * ca, -ct * sa, a * st],
                             [0, sa, ca, d], [0, 0, 0, 1]])
            chain.append(chain[-1] @ link)
        return chain

    def set_positions(self, positions):
        q = _vector(positions, 6, "joint positions")
        self._frames = self._chain(q)
        self.positions = q

    def synchronize(self, state):
        if state.gripper_width is None:
            if any(np.any(link.opening_axis) for link in self.links):
                raise ValueError("width-dependent collision geometry requires measured gripper width")
        elif not np.isfinite(state.gripper_width) or state.gripper_width < 0:
            raise ValueError("gripper width must be finite and non-negative")
        self.set_positions(state.joint_positions)
        self.gripper_width = None if state.gripper_width is None else float(state.gripper_width)

    def forward_kinematics(self, positions):
        """World-from-tool, without changing the synchronized state."""
        return self._chain(positions)[-1] @ self.flange_from_tool

    def tool_pose(self):
        return self._frames[-1] @ self.flange_from_tool

    def grasp_pose(self):
        return self.tool_pose() @ self.tool_from_grasp

    def tool_jacobian(self, positions, *, local=False):
        self.set_positions(positions)
        tool = self.tool_pose()
        axes = np.column_stack([frame[:3, 2] for frame in self._frames[:-1]])
        origins = np.column_stack([frame[:3, 3] for frame in self._frames[:-1]])
        linear = np.cross(axes.T, (tool[:3, 3, None] - origins).T).T
        rotation = tool[:3, :3].T if local else np.eye(3)
        return np.vstack((rotation @ linear, rotation @ axes))

    def set_obstacles(self, obstacles):
        """Replace world geometry between updates; reset the MPC plan afterward.

        This map is supplied by the application, not inferred from images.
        Per-obstacle exclude_links can omit a known fixed base/table mounting.
        """
        obstacles = tuple(obstacles)
        if any(not isinstance(o, (SphereObstacle, BoxObstacle, PlaneObstacle)) for o in obstacles):
            raise ValueError("unsupported obstacle shape")
        names = {link.name for link in self.links}
        if len({o.name for o in obstacles}) != len(obstacles) or any(not o.name for o in obstacles):
            raise ValueError("obstacle names must be nonempty and unique")
        if any(not set(o.exclude_links) <= names for o in obstacles):
            raise ValueError("obstacle exclusions must name configured links")
        self.obstacles = obstacles
        self._pairs = [("environment", i, j) for i, link in enumerate(self.links)
                       for j, obstacle in enumerate(obstacles) if link.name not in obstacle.exclude_links]
        for i, a in enumerate(self.links):
            frame_a = 6 if a.frame == "tool" else a.frame
            for j in range(i + 1, len(self.links)):
                b = self.links[j]
                frame_b = 6 if b.frame == "tool" else b.frame
                if (abs(frame_a - frame_b) > 1
                        and frozenset((a.name, b.name)) not in self._excluded_self_pairs):
                    self._pairs.append(("self", i, j))

    def _capsules(self, frames):
        tool = frames[-1] @ self.flange_from_tool
        result = []
        for link in self.links:
            frame = tool if link.frame == "tool" else frames[link.frame]
            shift = link.opening_axis * self.gripper_width if np.any(link.opening_axis) else np.zeros(3)
            start = frame[:3, :3] @ (link.start + shift) + frame[:3, 3]
            end = frame[:3, :3] @ (link.end + shift) + frame[:3, 3]
            result.append((start, end, link.radius))
        return result

    def _distances_at(self, positions):
        capsules = self._capsules(self._chain(positions))
        self_pairs = [(i, j) for kind, i, j in self._pairs if kind == "self"]
        if self_pairs:
            indices = np.asarray(self_pairs)
            starts = np.asarray([c[0] for c in capsules])
            ends = np.asarray([c[1] for c in capsules])
            radii = np.asarray([c[2] for c in capsules])
            i, j = indices.T
            self_distances = iter(_segment_distances(starts[i], ends[i], starts[j], ends[j])
                                  - radii[i] - radii[j])
        result = []
        for kind, i, j in self._pairs:
            a, b, radius = capsules[i]
            if kind == "self":
                distance = next(self_distances)
            else:
                obstacle = self.obstacles[j]
                if isinstance(obstacle, SphereObstacle):
                    distance = (_segment_distance(a, b, obstacle.center, obstacle.center)
                                - radius - obstacle.radius)
                elif isinstance(obstacle, PlaneObstacle):
                    distance = min(a @ obstacle.normal, b @ obstacle.normal) - obstacle.offset - radius
                else:
                    box = inverse_pose(obstacle.world_from_box)
                    c = box[:3, :3] @ a + box[:3, 3]
                    d = box[:3, :3] @ b + box[:3, 3]
                    distance = _segment_box_distance(c, d, obstacle.half_sizes) - radius
            result.append(distance)
        return np.asarray(result, dtype=float)

    def distances(self, positions, *, gradients=False):
        """Positive clearance / negative collision, plus optional d(clearance)/dq.

        Capsule-box distance is exact outside the box; when the centerline
        intersects it, -radius flags collision but is not penetration depth.
        At contact switches, distance gradients may be nondifferentiable.
        """
        if not self.links:
            raise ValueError("configure physical link_capsules before collision checking")
        self.set_positions(positions)
        result = self._distances_at(self.positions)
        if not gradients:
            return result
        derivatives = np.empty((len(result), 6))
        for joint in range(6):
            delta = np.zeros(6)
            delta[joint] = self.gradient_step
            derivatives[:, joint] = (self._distances_at(self.positions + delta)
                                     - self._distances_at(self.positions - delta)) / (2 * self.gradient_step)
        return result, derivatives

    def pickup_collision_indices(self):
        if not self.links:
            raise ValueError("configure physical link_capsules before pickup")
        return [index for index, (kind, i, j) in enumerate(self._pairs)
                if not (kind == "environment" and self.links[i].contact_group == "finger"
                        and self.obstacles[j].contact_group == "grasp_target")]

    def collision_pair_name(self, index):
        kind, i, j = self._pairs[index]
        other = self.links[j] if kind == "self" else self.obstacles[j]
        return f"{kind}: {self.links[i].name} / {other.name}"
