"""Six-state pose error EKF driven by body-local encoder FK increments.

Absolute FK seeds startup when no tag is available; it is not a recurring
measurement. Expiry is latched: reset/restart
while stopped is required, rather than snapping a moving robot to a new tag.
"""
import numpy as np
from scipy.spatial.transform import Rotation

from common.estimator import _pose, _inverse, _skew, _left_jacobian


class ToolIncrementEKF:
    def __init__(self, config):
        self.config = config
        self.reset()

    def reset(self):
        self.timestamp = None
        self.measurement_timestamp = None
        self.initialization_timestamp = None
        self.last_camera_timestamp = None
        self.fk = None
        self.pose = None
        self.covariance = np.zeros((6, 6))
        self.last_status = 'uninitialized'
        self.blocked = False
        self.measurement_accepted = False

    def step(self, timestamp, fk, measurement=None, measurement_timestamp=None):
        fk = _pose(fk, 'FK pose')
        if not np.isfinite(timestamp) or (self.timestamp is not None and timestamp < self.timestamp):
            raise ValueError('tool FK timestamps must be finite and monotonic')
        self.measurement_accepted = False
        if self.blocked:
            return None
        c = self.config
        fresh = (measurement is not None and measurement_timestamp is not None
                 and np.isfinite(measurement_timestamp)
                 and 0 <= timestamp - measurement_timestamp <= c.max_prediction_age
                 and (self.last_camera_timestamp is None or measurement_timestamp > self.last_camera_timestamp))
        if self.pose is None:
            self.initialization_timestamp = timestamp
            self.pose = (_pose(measurement, 'tool measurement') if fresh else fk).copy()
            self.covariance = np.diag([c.position_measurement_std**2]*3
                                     + [c.orientation_measurement_std**2]*3)
            if fresh:
                self.measurement_timestamp = self.last_camera_timestamp = measurement_timestamp
            self.measurement_accepted = fresh
            self.last_status = 'initialized' if fresh else 'fk_initialized'
        else:
            # Stop before any reacquisition can silently reinitialize the track.
            anchor = (self.initialization_timestamp if self.measurement_timestamp is None
                      else self.measurement_timestamp)
            if timestamp - anchor > c.max_prediction_age:
                self.blocked, self.last_status = True, 'expired_reset_required'
                return None
            dt = timestamp - self.timestamp
            delta = _inverse(self.fk) @ fk
            rotation = self.pose[:3, :3].copy()
            transition = np.eye(6)
            transition[:3, 3:] = -_skew(rotation @ delta[:3, 3])
            self.pose = self.pose @ delta
            noise = np.diag([c.fk_position_noise**2]*3 + [c.fk_orientation_noise**2]*3) * dt
            self.covariance = transition @ self.covariance @ transition.T + noise
            self.last_status = 'fk_predicted'
            if fresh:
                measured = _pose(measurement, 'tool measurement')
                self.last_camera_timestamp = measurement_timestamp
                residual = np.r_[measured[:3, 3] - self.pose[:3, 3],
                    Rotation.from_matrix(measured[:3, :3] @ self.pose[:3, :3].T).as_rotvec()]
                measurement_cov = np.diag([c.position_measurement_std**2]*3
                                          + [c.orientation_measurement_std**2]*3)
                innovation = self.covariance + measurement_cov
                distance = float(residual @ np.linalg.solve(innovation, residual))
                if distance <= c.innovation_gate:
                    gain = np.linalg.solve(innovation, self.covariance).T
                    correction = gain @ residual
                    self.pose[:3, 3] += correction[:3]
                    self.pose[:3, :3] = Rotation.from_rotvec(correction[3:]).as_matrix() @ self.pose[:3, :3]
                    remaining = np.eye(6) - gain
                    covariance = remaining @ self.covariance @ remaining.T + gain @ measurement_cov @ gain.T
                    reset = np.eye(6)
                    reset[3:, 3:] = _left_jacobian(correction[3:])
                    self.covariance = reset @ covariance @ reset.T
                    self.measurement_timestamp = measurement_timestamp
                    self.measurement_accepted = True
                    self.last_status = 'updated'
                else:
                    self.last_status = 'measurement_rejected_fk_predicted'
        self.timestamp, self.fk = timestamp, fk.copy()
        self.covariance = (self.covariance + self.covariance.T) / 2
        if (np.linalg.eigvalsh(self.covariance[:3, :3])[-1] > c.max_position_std**2
                or np.linalg.eigvalsh(self.covariance[3:, 3:])[-1] > c.max_orientation_std**2):
            self.blocked, self.last_status = True, 'uncertain_reset_required'
            return None
        return self.pose.copy()
