"""Conservative local Cartesian placement, not a global obstacle planner."""

import cv2
from dataclasses import replace
import numpy as np

from common.camera_processing import CameraProcessor
from common.estimator import RelativePoseEKF
from common.controller import PBVS, JointSpaceMPC, pose_reference_command
from common.geometry import BoxObstacle, PlaneObstacle, SphereObstacle, _segment_box_distance


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
    def __init__(self, session, destination, *, guard_factory=PlacementCollisionGuard,
                 frame_max_age=None, acceleration_limit=None):
        self.s, self.destination = session, destination
        self.tool_tracker = getattr(session, 'tool_tracker', None)
        if self.tool_tracker is not None:
            self.tool_tracker.kinematics = session.kinematics
        placement_clearance = getattr(session, 'placement_clearance', session.settings.clearance)
        self.processor = CameraProcessor(marker_id=destination.marker_id,
            marker_size=destination.marker_size_m, cube_size=destination.marker_size_m)
        self.phase = "acquiring"
        self.started = session.interface.timestamp
        self.frame_max_age = session.hardware.control["frame_max_age"] if frame_max_age is None else frame_max_age
        self.acceleration_limit = (session.hardware.control["acceleration_limit"]
                                   if acceleration_limit is None else acceleration_limit)
        self.guard = guard_factory(session.kinematics, session.cube_size,
            placement_clearance, session.tool_from_cube, destination.collision_support_plane)
        self.collision_support = next(o for o in session.kinematics.obstacles if o.name == destination.collision_support_plane)
        self.placement_plane = None
        self.targets = []
        self.released = False
        self._progress = "no motion sample yet"
        self._last_joint_positions = None
        self.tool_processor = None
        self.tool_estimator = None
        self.tool_pose_source = 'fk'
        self.transfer_pbvs = PBVS(np.eye(4), linear_speed=session.config.cartesian_speed_m_s,
                                  angular_speed=session.config.angular_speed_rad_s)
        transfer_settings = (replace(session.settings, clearance=placement_clearance)
                             if hasattr(session, 'placement_clearance') else session.settings)
        self.transfer_mpc = (JointSpaceMPC(session.kinematics, transfer_settings)
                             if session.controller_config.mode == 'pbvs_mpc' else None)

    def _transfer_observation(self):
        s = self.s
        if self.phase != 'transferring' or s.controller_config.tool_pose_source == 'fk':
            return None, None
        if self.tool_processor is None:
            k = s.kinematics
            self.tool_processor = CameraProcessor(marker_id=k.end_effector_marker_id,
                marker_size=k.end_effector_marker_size,
                cube_size=max(.1, k.end_effector_marker_size))
            self.tool_estimator = RelativePoseEKF(s.estimator_config.for_target('tool'))
        frame = s.interface.capture_rgbd()
        if getattr(s, 'camera_view', False):
            s.publish_preview(frame)
        return frame, self.tool_processor.process_marker(frame)

    def _transfer_pose(self, frame, observation, state):
        s = self.s
        self.tool_pose_source = 'fk'
        if frame is None:
            return s.kinematics.tool_pose().copy()
        now = s.interface.timestamp
        age = now - frame.timestamp
        state_age = now - state.timestamp
        limit = min(self.frame_max_age, s.controller_config.frame_max_age_intervals * s.settings.dt)
        estimator = self.tool_estimator
        if estimator.timestamp is not None and now < estimator.timestamp:
            estimator.reset()
        if (not np.isfinite(age) or not 0 <= age <= limit
                or not np.isfinite(state_age) or not 0 <= state_age <= 2 * s.settings.dt
                or (estimator.config.enabled and estimator.timestamp is not None
                    and frame.timestamp < estimator.timestamp)):
            s.diagnostic = 'stale_transfer_tool_measurement'
            self.tool_pose_source = 'unavailable'
            return None
        reason = self.tool_processor.last_status
        if observation is not None:
            camera_from_tool = observation.camera_from_marker @ np.linalg.inv(
                s.kinematics.tool_from_end_effector_marker)
            if estimator.config.enabled:
                estimate = estimator.update(frame.timestamp, camera_from_tool, frame.world_from_camera)
                if estimate is not None and estimate.measurement_accepted:
                    camera_from_tool = estimate.camera_from_cube
                else:
                    observation = None
                    reason = 'tool_ekf_' + estimator.last_status
            if observation is not None:
                self.tool_pose_source = 'tag'
                s.interface.require_camera_for_motion = True
                return frame.world_from_camera @ camera_from_tool
        if s.controller_config.tool_pose_source == 'tag_only':
            s.diagnostic = 'tool_lost_or_rejected: ' + reason
            self.tool_pose_source = 'unavailable'
            return None
        self.tool_pose_source = 'fk_fallback'
        return s.kinematics.tool_pose().copy()

    def _phase(self, name):
        if self.transfer_mpc is not None:
            self.transfer_mpc.reset()
        self.phase, self.started = name, self.s.interface.timestamp
        self._progress = "no motion sample yet"
        self._last_joint_positions = None
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
                               f"limit={phase_limit:g} s; last_sample: {self._progress}; "
                               f"joint_positions_rad={self._last_joint_positions}")
        s.status = "placement_" + self.phase
        if self.phase == "acquiring":
            interface.set_joint_velocities(np.zeros(6))
            frame = interface.capture_rgbd()
            if getattr(s, "camera_view", False):
                s.publish_preview(frame)
            observation = self.processor.process_marker(frame)
            tracked_tool = self.tool_tracker.refresh(frame) if self.tool_tracker is not None else None
            # Read feedback AFTER perception, as in the approach controller.
            state = interface.read_state()
            if not s.cube_size * .5 <= state.gripper_width <= s.cube_size * 1.15:
                raise RuntimeError("grasp_lost during destination acquisition")
            if observation is None:
                s.diagnostic = "destination_acquisition_waiting: tag missing or rejected"
                return False
            age = interface.timestamp - observation.timestamp
            if not 0 <= age <= min(self.frame_max_age,
                    s.controller_config.frame_max_age_intervals * s.settings.dt):
                s.diagnostic = "destination_acquisition_waiting: stale observation"
                return False
            k.synchronize(state)
            if self.tool_tracker is not None and tracked_tool is None:
                s.diagnostic = 'tool_lost_or_rejected: ' + self.tool_tracker.diagnostic
                return False
            tag = frame.world_from_camera @ observation.camera_from_marker
            # The tag's local +Z is the placement surface normal. Keep this
            # local reference separate from the configured collision map.
            try:
                placement_plane, target = self._destination_candidate(tag)
            except ValueError as error:
                s.diagnostic = f"destination_acquisition_rejected: {error}; waiting for first valid target"
                return False
            current = (tracked_tool if self.tool_tracker is not None else k.tool_pose()).copy()
            self.placement_plane = placement_plane
            s.diagnostic = ""
            # Lift first, then translate/rotate above destination, then descend.
            high = max(current[2, 3], target[2, 3] + self.destination.approach_height_m)
            lift, above = current.copy(), target.copy()
            lift[2, 3] = above[2, 3] = high
            self.targets = [lift, above, target]
            self._phase("raising")
            return False

        if self.tool_tracker is not None:
            tracked_tool = self.tool_tracker.refresh()
            if getattr(s, 'camera_view', False) and self.tool_tracker.frame is not None:
                s.publish_preview(self.tool_tracker.frame)
            frame, tool_observation = None, None
        else:
            tracked_tool = None
            frame, tool_observation = self._transfer_observation()
        state = interface.read_state()
        k.synchronize(state)
        self._last_joint_positions = state.joint_positions.tolist()
        if not self.released and not s.cube_size * .5 <= state.gripper_width <= s.cube_size * 1.15:
            raise RuntimeError("grasp_lost during placement")
        if self.tool_tracker is not None:
            current = tracked_tool
            self.tool_pose_source = self.tool_tracker.source
            if current is None:
                s.diagnostic = 'tool_lost_or_rejected: ' + self.tool_tracker.diagnostic
        else:
            current = self._transfer_pose(frame, tool_observation, state)
        if current is None:
            if self.transfer_mpc is not None:
                self.transfer_mpc.reset()
            interface.set_joint_velocities(np.zeros(6))
            self._progress = s.diagnostic
            return False
        self.guard.check(state.joint_positions)
        if self.phase == "releasing":
            interface.set_joint_velocities(np.zeros(6))
            if state.gripper_width >= interface.gripper_max_width - .001:
                self._phase("retreating")
            return False

        target = self.targets[0]
        error = np.r_[target[:3, 3] - current[:3, 3],
                      cv2.Rodrigues(target[:3, :3] @ current[:3, :3].T)[0].reshape(3)]
        self._progress = (
            f"phase={self.phase}, tool={self.tool_pose_source}, elapsed_s={interface.timestamp-self.started:.1f}, "
            f"current_z_mm={current[2, 3]*1000:.1f}, target_z_mm={target[2, 3]*1000:.1f}, "
            f"position_error_mm={np.linalg.norm(error[:3])*1000:.1f}, "
            f"xy_error_mm={np.linalg.norm(error[:2])*1000:.1f}, z_error_mm={error[2]*1000:.1f}, "
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

        if self.phase == 'transferring':
            jacobian = k.tool_jacobian(state.joint_positions, local=True)
            reference, _ = self.transfer_pbvs.reference(np.linalg.inv(current) @ target, jacobian)
            command, status, _ = pose_reference_command(k, s.settings, self.transfer_mpc,
                state.joint_positions, state.joint_velocities, reference)
            if status != 'tracking':
                interface.set_joint_velocities(np.zeros(6))
                s.diagnostic = 'transfer_controller: ' + status
                if self.transfer_mpc is not None:
                    self.transfer_mpc.reset()
                return False
        else:
            twist = 4. * error
            twist[:3] *= min(1., s.config.cartesian_speed_m_s / max(np.linalg.norm(twist[:3]), 1e-12))
            twist[3:] *= min(1., s.config.angular_speed_rad_s / max(np.linalg.norm(twist[3:]), 1e-12))
            jacobian = k.tool_jacobian(state.joint_positions)
            command = jacobian.T @ np.linalg.solve(jacobian @ jacobian.T + .01**2 * np.eye(6), twist)
            command /= max(1., np.max(np.abs(command)) / s.settings.velocity_limit)
        acceleration = self.acceleration_limit * s.settings.dt
        limited = np.clip(command, state.joint_velocities - acceleration, state.joint_velocities + acceleration)
        clipped_joints = int(np.count_nonzero(np.abs(limited - command) > 1e-10))
        command = limited
        self._progress += f", commanded_max_joint_speed_rad_s={np.max(np.abs(command)):.4f}"
        self._progress += (f", jacobian_sigma_min={np.linalg.svd(jacobian, compute_uv=False)[-1]:.6f}, "
                           f"acceleration_clipped_joints={clipped_joints}, "
                           f"state_age_ms={(interface.timestamp-state.timestamp)*1000:.1f}")
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
