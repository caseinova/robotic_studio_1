"""Shared tool-pose selection; encoder kinematics remain authoritative for safety."""
import numpy as np

from common.camera_processing import CameraProcessor
from common.estimator import RelativePoseEKF
from common.tool_increment_ekf import ToolIncrementEKF


class ToolPoseTracker:
    def __init__(self, interface, kinematics, settings, estimator_config, dt):
        self.interface, self.kinematics = interface, kinematics
        self.settings, self.dt = settings, dt
        self.processor = CameraProcessor(marker_id=kinematics.end_effector_marker_id,
            marker_size=kinematics.end_effector_marker_size,
            cube_size=max(.1, kinematics.end_effector_marker_size))
        self.estimator = RelativePoseEKF(estimator_config.for_target('tool'))
        self.increment_estimator = ToolIncrementEKF(self.estimator.config)
        self.pose = None
        self.source = 'unavailable'
        self.diagnostic = ''
        self.observation = None
        self.frame = None

    def select(self, frame, observation):
        self.frame, self.observation = frame, observation
        self.pose = None
        self.source = 'unavailable'
        self.diagnostic = self.processor.last_status
        if (self.estimator.config.enabled
                and self.estimator.config.prediction_model == 'fk_increment'
                and self.settings.tool_pose_source != 'fk'):
            measured = None if observation is None else (
                frame.world_from_camera @ observation.camera_from_marker
                @ np.linalg.inv(self.kinematics.tool_from_end_effector_marker))
            self.pose = self.increment_estimator.step(self.interface.timestamp,
                self.kinematics.tool_pose(), measured, frame.timestamp)
            self.diagnostic = self.increment_estimator.last_status
            if not self.increment_estimator.measurement_accepted:
                self.observation = None
                if self.settings.tool_pose_source == 'tag_only':
                    self.pose = None
            if self.pose is not None:
                self.source = 'tag' if self.increment_estimator.measurement_accepted else 'fk_predicted'
            return self.pose
        if self.estimator.timestamp is not None and self.interface.timestamp < self.estimator.timestamp:
            self.estimator.reset()
        if observation is not None and self.settings.tool_pose_source != 'fk':
            camera_from_tool = observation.camera_from_marker @ np.linalg.inv(
                self.kinematics.tool_from_end_effector_marker)
            if self.estimator.config.enabled:
                estimate = self.estimator.update(frame.timestamp, camera_from_tool, frame.world_from_camera)
                if estimate is not None and estimate.measurement_accepted:
                    camera_from_tool = estimate.camera_from_cube
                else:
                    self.observation = None
                    self.diagnostic = 'tool_ekf_' + self.estimator.last_status
            if self.observation is not None:
                self.pose = frame.world_from_camera @ camera_from_tool
                self.source = 'tag'
        if self.pose is None and self.settings.tool_pose_source != 'tag_only':
            self.pose = self.kinematics.tool_pose().copy()
            self.source = 'fk' if self.settings.tool_pose_source == 'fk' else 'fk_fallback'
        return self.pose

    def refresh(self, frame=None):
        observation = None
        self.frame = None
        if self.settings.tool_pose_source != 'fk':
            frame = self.interface.capture_rgbd() if frame is None else frame
            observation = self.processor.process_marker(frame)
            self.frame = frame
        state = self.interface.read_state()
        self.kinematics.synchronize(state)
        now = self.interface.timestamp
        state_age = now - state.timestamp
        frame_age = 0. if self.frame is None else now - frame.timestamp
        if (not np.isfinite(state_age) or not 0 <= state_age <= 2 * self.dt
                or not np.isfinite(frame_age)
                or not 0 <= frame_age <= self.settings.frame_max_age_intervals * self.dt):
            self.pose, self.source = None, 'unavailable'
            self.diagnostic = 'stale_tool_pose_measurement'
        else:
            self.select(frame, observation)
        self.interface.require_camera_for_motion = self.source in ('tag', 'fk_predicted')
        return self.pose

    def grasp_pose(self, width=None):
        k = self.kinematics
        grasp = k.grasp_pose() if width is None else k.grasp_pose_at_width(width)
        return self.pose @ np.linalg.inv(k.tool_pose()) @ grasp
