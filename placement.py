"""Conservative local Cartesian placement, not a global obstacle planner."""

import cv2
import numpy as np

from camera_processing import CameraProcessor
from hardware_kinematics import (BoxObstacle, PlaneObstacle, SphereObstacle,
                                 _segment_box_distance)


def cube_plane_gap(pose, size, plane):
    extent = size / 2 * np.sum(np.abs(pose[:3, :3].T @ plane.normal))
    return float(plane.normal @ pose[:3, 3] - plane.offset - extent)


def vertical_drop_height(pose, size, plane):
    """World-Z fall until the cube first contacts an upward-facing plane."""
    if plane.normal[2] <= 1e-6:
        raise ValueError("destination tag plane must face upward for vertical placement")
    return cube_plane_gap(pose, size, plane) / plane.normal[2]


class PlacementCollisionGuard:
    """Arm/tool capsules plus an oriented payload cube against configured geometry.

    Only finger/payload contact and payload/support contact are permitted.
    Cube/box environment checks use a conservative bounding sphere. No robot
    link is exempted from the support plane's normal clearance requirement.
    """

    def __init__(self, kinematics, size, clearance, tool_from_cube, support_name):
        self.kinematics, self.size, self.clearance = kinematics, size, clearance
        self.tool_from_cube = tool_from_cube.copy()
        self.support_name = support_name
        self.released_pose = None
        self.released_half_sizes = np.full(3, size / 2)

    def check(self, q):
        k = self.kinematics
        distances = k.distances(q)
        if len(distances) and np.min(distances) < self.clearance:
            i = int(np.argmin(distances))
            raise RuntimeError(f"placement_path_blocked: {k.collision_pair_name(i)}; "
                               f"clearance={distances[i]*1000:.1f} mm")
        cube = self.released_pose
        if cube is None:
            cube = k.tool_pose() @ self.tool_from_cube
        inverse = np.linalg.inv(cube)
        half = (np.full(3, self.size / 2) if self.released_pose is None
                else self.released_half_sizes)
        for link, (a, b, radius) in zip(k.links, k._capsules(k._frames)):
            finger = link.frame == "tool" and link.contact_group == "finger"
            if finger and self.released_pose is None:
                continue
            a = inverse[:3, :3] @ a + inverse[:3, 3]
            b = inverse[:3, :3] @ b + inverse[:3, 3]
            gap = _segment_box_distance(a, b, half) - radius
            if gap < (0. if finger else self.clearance):
                raise RuntimeError(f"placement_path_blocked: payload / {link.name}; gap={gap*1000:.1f} mm")
        if self.released_pose is not None:
            return
        for obstacle in k.obstacles:
            required = self.clearance
            if isinstance(obstacle, PlaneObstacle):
                gap = cube_plane_gap(cube, self.size, obstacle)
                if obstacle.name == self.support_name:
                    required = 0.
            elif isinstance(obstacle, SphereObstacle):
                center = inverse[:3, :3] @ obstacle.center + inverse[:3, 3]
                gap = np.linalg.norm(np.maximum(np.abs(center) - half, 0)) - obstacle.radius
            else:
                box_inverse = np.linalg.inv(obstacle.world_from_box)
                center = box_inverse[:3, :3] @ cube[:3, 3] + box_inverse[:3, 3]
                gap = (np.linalg.norm(np.maximum(np.abs(center) - obstacle.half_sizes, 0))
                       - np.sqrt(3) * self.size / 2)
            if gap < required - 1e-9:
                raise RuntimeError(f"placement_path_blocked: payload / {obstacle.name}; gap={gap*1000:.1f} mm")


class PlaceAction:
    def __init__(self, session, destination):
        self.s, self.destination = session, destination
        self.processor = CameraProcessor(marker_id=destination.marker_id,
            marker_size=destination.marker_size_m, cube_size=destination.marker_size_m)
        self.phase = "acquiring"
        self.started = session.interface.timestamp
        self.guard = PlacementCollisionGuard(session.kinematics, session.cube_size,
            session.settings.clearance, session.tool_from_cube, destination.collision_support_plane)
        self.collision_support = next(o for o in session.kinematics.obstacles if o.name == destination.collision_support_plane)
        self.placement_plane = None
        self.targets = []
        self.released = False
        self._progress = "no motion sample yet"

    def _phase(self, name):
        self.phase, self.started = name, self.s.interface.timestamp
        self._progress = "no motion sample yet"
        self.s.diagnostic = ""

    def _destination_candidate(self, tag):
        s = self.s
        plane = PlaneObstacle("destination_tag_plane", tag[:3, 2],
                              float(tag[:3, 2] @ tag[:3, 3]))
        cube = tag @ self.destination.marker_from_cube
        gap = vertical_drop_height(cube, s.cube_size, plane)
        if not -1e-9 <= gap <= self.destination.max_drop_height_m:
            raise ValueError("destination cube outside configured drop-height range relative to tag plane; "
                f"gap={gap*1000:.1f} mm (allowed 0..{self.destination.max_drop_height_m*1000:g} mm)")
        table_gap = cube_plane_gap(cube, s.cube_size, self.collision_support)
        if table_gap < -1e-9:
            raise ValueError("destination cube intersects configured collision table; "
                             f"gap={table_gap*1000:.1f} mm")
        return plane, cube @ np.linalg.inv(s.tool_from_cube)

    def update(self):
        s, interface, k = self.s, self.s.interface, self.s.kinematics
        interface.require_camera_for_motion = False
        phase_limit = (s.config.transfer_timeout_s
                       if self.phase == "transferring" and s.config.transfer_timeout_s is not None
                       else s.config.phase_timeout_s)
        if interface.timestamp - self.started > phase_limit:
            raise TimeoutError(f"placement phase_timeout: {self.phase}; "
                               f"limit={phase_limit:g} s; last_sample: {self._progress}")
        s.status = "placement_" + self.phase
        if self.phase == "acquiring":
            interface.set_joint_velocities(np.zeros(6))
            frame = interface.capture_rgbd()
            if getattr(s, "camera_view", False):
                s.publish_preview(frame)
            observation = self.processor.process_marker(frame)
            # Read feedback AFTER perception, as in the approach controller.
            state = interface.read_state()
            if not s.cube_size * .5 <= state.gripper_width <= s.cube_size * 1.15:
                raise RuntimeError("grasp_lost during destination acquisition")
            if observation is None:
                s.diagnostic = "destination_acquisition_waiting: tag missing or rejected"
                return False
            age = interface.timestamp - observation.timestamp
            if not 0 <= age <= min(s.hardware.control["frame_max_age"],
                    s.controller_config.frame_max_age_intervals * s.settings.dt):
                s.diagnostic = "destination_acquisition_waiting: stale observation"
                return False
            k.synchronize(state)
            tag = frame.world_from_camera @ observation.camera_from_marker
            # The tag's local +Z is the placement surface normal. Keep this
            # local reference separate from the configured collision map.
            try:
                placement_plane, target = self._destination_candidate(tag)
            except ValueError as error:
                s.diagnostic = f"destination_acquisition_rejected: {error}; waiting for first valid target"
                return False
            current = k.tool_pose().copy()
            self.placement_plane = placement_plane
            s.diagnostic = ""
            # Lift first, then translate/rotate above destination, then descend.
            high = max(current[2, 3], target[2, 3] + self.destination.approach_height_m)
            lift, above = current.copy(), target.copy()
            lift[2, 3] = above[2, 3] = high
            self.targets = [lift, above, target]
            self._phase("raising")
            return False

        state = interface.read_state()
        k.synchronize(state)
        if not self.released and not s.cube_size * .5 <= state.gripper_width <= s.cube_size * 1.15:
            raise RuntimeError("grasp_lost during placement")
        self.guard.check(state.joint_positions)
        if self.phase == "releasing":
            interface.set_joint_velocities(np.zeros(6))
            if state.gripper_width >= interface.gripper_max_width - .001:
                self._phase("retreating")
            return False

        target = self.targets[0]
        current = k.tool_pose().copy()
        error = np.r_[target[:3, 3] - current[:3, 3],
                      cv2.Rodrigues(target[:3, :3] @ current[:3, :3].T)[0].reshape(3)]
        self._progress = (
            f"phase={self.phase}, elapsed_s={interface.timestamp-self.started:.1f}, "
            f"current_z_mm={current[2, 3]*1000:.1f}, target_z_mm={target[2, 3]*1000:.1f}, "
            f"position_error_mm={np.linalg.norm(error[:3])*1000:.1f}, "
            f"rotation_error_deg={np.degrees(np.linalg.norm(error[3:])):.2f}, "
            f"measured_max_joint_speed_rad_s={np.max(np.abs(state.joint_velocities)):.4f}")
        s.diagnostic = self._progress
        if np.linalg.norm(error[:3]) < .0015 and np.linalg.norm(error[3:]) < .02:
            interface.set_joint_velocities(np.zeros(6))
            self.targets.pop(0)
            if self.phase == "descending":
                actual_cube = current @ s.tool_from_cube
                drop = vertical_drop_height(actual_cube, s.cube_size, self.placement_plane)
                if not 0 <= drop <= self.destination.max_drop_height_m:
                    raise RuntimeError("release position outside configured drop-height range relative to tag plane: "
                                       f"gap={drop*1000:.1f} mm")
                # Check the full finger-opening sweep before sending the command.
                old_width = k.gripper_width
                try:
                    for width in np.linspace(old_width, interface.gripper_max_width, 6):
                        k.gripper_width = width
                        self.guard.check(state.joint_positions)
                finally:
                    k.gripper_width = old_width
                interface.open_gripper()
                self.released = True
                self.guard.released_pose = actual_cube.copy()
                # Enclose the vertical fall as well as the release pose. Do not
                # pretend the dropped cube stays suspended during retreat.
                fall_direction = np.array([0., 0., 1.])
                self.guard.released_pose[:3, 3] -= fall_direction * drop / 2
                self.guard.released_half_sizes += (
                    np.abs(actual_cube[:3, :3].T @ fall_direction) * drop / 2)
                s.tool_from_cube = None
                retreat = current.copy()
                retreat[2, 3] += self.destination.retreat_height_m
                self.targets = [retreat]
                self._phase("releasing")
            elif self.phase == "retreating":
                # Keep the deposited cube in the map for subsequent actions.
                obstacle = BoxObstacle(f"placed_cube_{len(k.obstacles)}", self.guard.released_pose,
                                       self.guard.released_half_sizes)
                k.set_obstacles((*k.obstacles, obstacle))
                return True
            else:
                self._phase("transferring" if self.phase == "raising" else "descending")
            return False

        twist = 4. * error
        twist[:3] *= min(1., s.config.cartesian_speed_m_s / max(np.linalg.norm(twist[:3]), 1e-12))
        twist[3:] *= min(1., s.config.angular_speed_rad_s / max(np.linalg.norm(twist[3:]), 1e-12))
        jacobian = k.tool_jacobian(state.joint_positions)
        command = jacobian.T @ np.linalg.solve(jacobian @ jacobian.T + .01**2 * np.eye(6), twist)
        command /= max(1., np.max(np.abs(command)) / s.settings.velocity_limit)
        acceleration = s.hardware.control["acceleration_limit"] * s.settings.dt
        command = np.clip(command, state.joint_velocities - acceleration, state.joint_velocities + acceleration)
        self._progress += f", commanded_max_joint_speed_rad_s={np.max(np.abs(command)):.4f}"
        s.diagnostic = self._progress
        if np.max(np.abs(command)) > s.settings.velocity_limit:
            raise RuntimeError("measured joint velocity exceeds placement speed limit")
        prediction = state.joint_positions + s.settings.dt * command
        lower, upper = k.limits
        if np.any(prediction < lower) or np.any(prediction > upper):
            raise RuntimeError("placement joint_limit_violation")
        for fraction in (.25, .5, .75, 1.):
            self.guard.check(state.joint_positions + fraction * (prediction - state.joint_positions))
        interface.set_joint_velocities(command)
        return False
