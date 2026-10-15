"""Moving cube/camera tracking, rotation geometry, and controller integration."""

from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from scipy.spatial.transform import Rotation

from estimator import EKFConfig, RelativePoseEKF


def pose(position=(0, 0, 0), rotvec=(0, 0, 0)):
    result = np.eye(4)
    result[:3, :3] = Rotation.from_rotvec(rotvec).as_matrix()
    result[:3, 3] = position
    return result


class EstimatorTests(unittest.TestCase):
    def test_configuration_validation(self):
        for values in ({"enabled": 1}, {"position_measurement_std": 0},
                       {"innovation_gate": float("nan")}, {"max_prediction_age": -1}):
            with self.assertRaises(ValueError):
                EKFConfig(**values)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ekf.json"
            path.write_text(json.dumps({"enabled": False, "max_prediction_age": 0.5}))
            self.assertFalse(EKFConfig.load(path).enabled)
            path.write_text('{"typo": 1}')
            with self.assertRaisesRegex(ValueError, "unknown"):
                EKFConfig.load(path)

    def test_camera_motion_is_not_cube_motion_and_inverse_direction_is_correct(self):
        tracker = RelativePoseEKF()
        cube = pose((0.1, 0.5, 0.05), (0.1, 0.2, 0.3))
        for i in range(60):
            camera = pose((0.003 * i, 0.1, 0.4), (0.002 * i, -0.003 * i, 0.001 * i))
            estimate = tracker.update(i * 0.05, np.linalg.inv(camera) @ cube, camera)
            np.testing.assert_allclose(estimate.world_from_cube, cube, atol=1e-10)
            np.testing.assert_allclose(estimate.cube_from_camera, np.linalg.inv(cube) @ camera, atol=1e-10)
            np.testing.assert_allclose(estimate.camera_from_cube @ estimate.cube_from_camera, np.eye(4), atol=1e-10)
            np.testing.assert_allclose(estimate.linear_velocity, 0, atol=1e-10)
            np.testing.assert_allclose(estimate.angular_velocity, 0, atol=1e-10)

    def test_noisy_moving_cube_and_camera_estimate_motion_and_reduce_pose_error(self):
        tracker = RelativePoseEKF()
        rng = np.random.default_rng(12)
        velocity = np.array([0.02, -0.01, 0.003])
        omega = np.array([0.03, -0.04, 0.15])
        initial_rotation = Rotation.from_rotvec([0.3, -0.2, 0.5]).as_matrix()
        raw_errors, filtered_errors, raw_angles, filtered_angles = [], [], [], []
        for i in range(200):
            t = i * 0.05
            cube = pose(np.array([0.1, 0.4, 0.05]) + velocity * t)
            cube[:3, :3] = Rotation.from_rotvec(omega * t).as_matrix() @ initial_rotation
            camera = pose([0.1 * np.sin(t), 0.1, 0.4], [0.05 * t, -0.03 * t, 0.1])
            measurement = np.linalg.inv(camera) @ cube
            measurement[:3, 3] += rng.normal(0, 0.004, 3)
            measurement[:3, :3] = Rotation.from_rotvec(rng.normal(0, 0.03, 3)).as_matrix() @ measurement[:3, :3]
            estimate = tracker.update(t, measurement, camera)
            self.assertTrue(estimate.measurement_accepted)
            self.assertGreaterEqual(np.linalg.eigvalsh(estimate.covariance).min(), -1e-12)
            if i >= 30:
                raw = camera @ measurement
                raw_errors.append(np.linalg.norm(raw[:3, 3] - cube[:3, 3]))
                filtered_errors.append(np.linalg.norm(estimate.world_from_cube[:3, 3] - cube[:3, 3]))
                raw_angles.append(np.linalg.norm(Rotation.from_matrix(raw[:3, :3] @ cube[:3, :3].T).as_rotvec()))
                filtered_angles.append(np.linalg.norm(Rotation.from_matrix(estimate.world_from_cube[:3, :3] @ cube[:3, :3].T).as_rotvec()))
        self.assertLess(np.mean(filtered_errors), 0.8 * np.mean(raw_errors))
        self.assertLess(np.mean(filtered_angles), 0.8 * np.mean(raw_angles))
        np.testing.assert_allclose(estimate.linear_velocity, velocity, atol=0.04)
        np.testing.assert_allclose(estimate.angular_velocity, omega, atol=0.08)

    def test_prediction_gap_expiry_and_reacquisition(self):
        tracker = RelativePoseEKF()
        camera = np.eye(4)
        self.assertIsNone(tracker.predict(0, camera))
        for i in range(60):
            tracker.update(i * 0.05, pose((0.02 * i * 0.05, 0, 0.5), (0, 0, 0.1 * i * 0.05)), camera)
        before = tracker.predict(2.95, camera)
        prediction = tracker.predict(3.15, pose((0.01, 0, 0)))
        self.assertEqual(prediction.status, "predicted")
        self.assertFalse(prediction.measurement_accepted)
        np.testing.assert_allclose(prediction.world_from_cube[:3, 3], [0.063, 0, 0.5], atol=0.001)
        self.assertGreater(np.trace(prediction.covariance), np.trace(before.covariance))
        self.assertIsNone(tracker.predict(4, camera))
        self.assertEqual(tracker.last_status, "expired")
        estimate = tracker.update(4.05, pose((0.3, 0, 0.5)), camera)
        self.assertEqual(estimate.status, "reinitialized")
        np.testing.assert_allclose(estimate.world_from_cube[:3, 3], [0.3, 0, 0.5])

    def test_pose_outlier_does_not_move_track(self):
        tracker = RelativePoseEKF()
        cube = pose((0, 0, 0.5))
        tracker.update(0, cube, np.eye(4))
        result = tracker.update(0.05, pose((2, 0, 0.5), (0, 0, 2)), np.eye(4))
        self.assertEqual(result.status, "measurement_rejected")
        self.assertFalse(result.measurement_accepted)
        np.testing.assert_allclose(result.world_from_cube, cube)
        self.assertEqual(result.measurement_timestamp, 0)

    def test_rotation_crosses_pi_without_discontinuity(self):
        tracker = RelativePoseEKF()
        for i in range(80):
            angle = 3.0 + 0.2 * i * 0.05
            cube = pose((0, 0, 0.5), (0, 0, angle))
            result = tracker.update(i * 0.05, cube, np.eye(4))
            self.assertTrue(result.measurement_accepted)
            np.testing.assert_allclose(result.world_from_cube[:3, :3].T @ result.world_from_cube[:3, :3], np.eye(3), atol=1e-10)
        self.assertAlmostEqual(result.angular_velocity[2], 0.2, delta=0.005)

    def test_duplicate_time_reset_validation_and_owned_arrays(self):
        tracker = RelativePoseEKF()
        cube = pose((0, 0, 0.5))
        first = tracker.update(1, cube, np.eye(4))
        second = tracker.update(1, pose((1, 0, 0.5)), np.eye(4))
        self.assertEqual(second.status, "duplicate_measurement")
        np.testing.assert_array_equal(first.covariance, second.covariance)
        first.world_from_cube[:] = 0
        first.covariance[:] = 0
        np.testing.assert_allclose(tracker.predict(1, np.eye(4)).world_from_cube, cube)
        with self.assertRaises(ValueError):
            tracker.predict(0, np.eye(4))
        invalid = np.eye(4)
        invalid[0, 0] = 2
        with self.assertRaises(ValueError):
            tracker.update(1.05, invalid, np.eye(4))
        tracker.reset()
        self.assertIsNone(tracker.predict(0, np.eye(4)))

    def test_controller_consumes_filtered_pose_and_stops_on_rejected_measurement(self):
        from controller import ApproachController
        from controller_config import ControllerConfig
        from kinematics import MujocoKinematics
        from simulate import Simulation
        simulation = Simulation(controller_config=ControllerConfig(mode="pure_pbvs"))
        controller = ApproachController(simulation.interface, MujocoKinematics(simulation.robot),
                                        controller_config=simulation.controller_config)
        with simulation.interface:
            frame = simulation.interface.capture_rgbd()
        observation = controller.processor.process(frame)
        for i, shift in enumerate((0, 0.004, 1.0)):
            simulation.robot.data.time = i * 0.05
            current_frame = replace(frame, timestamp=i * 0.05)
            measured = observation.camera_from_cube.copy()
            measured[0, 3] += shift
            current_observation = replace(observation, timestamp=i * 0.05, camera_from_cube=measured)
            with patch.object(controller.processor, "process", return_value=current_observation):
                result = controller.update(current_frame)
            if i == 1:
                self.assertIsNotNone(result.pose_estimate)
                self.assertFalse(np.allclose(result.pose_estimate.camera_from_cube, measured))
                marker_offset = pose((0, 0, 0.05015))
                tool_observation = controller.tool_processor.process_marker(current_frame)
                camera_from_tool = tool_observation.camera_from_marker @ np.linalg.inv(
                    controller.kinematics.tool_from_end_effector_marker)
                expected, _ = controller.pbvs.reference(
                    np.linalg.inv(camera_from_tool)
                    @ result.pose_estimate.camera_from_cube @ marker_offset,
                    controller.kinematics.tool_jacobian(simulation.interface.joint_positions, local=True),
                )
                np.testing.assert_allclose(result.reference_velocity, expected)
            if i == 2:
                self.assertEqual(result.diagnostic, "ekf_measurement_rejected")
                np.testing.assert_array_equal(result.command_velocity, np.zeros(6))
        # A delayed exposure inside the freshness window must not reset the
        # tracker or apply an older observation as a new measurement.
        with patch.object(controller.processor, "process", return_value=observation):
            delayed = controller.update(frame)
        self.assertEqual(delayed.diagnostic, "stale_rgbd_frame")
        self.assertEqual(controller.estimator.timestamp, 0.1)
        # Restarting the interface clock intentionally starts a new track.
        simulation.robot.data.time = 0.0
        with patch.object(controller.processor, "process", return_value=observation):
            restarted = controller.update(frame)
        self.assertEqual(restarted.pose_estimate.status, "initialized")


if __name__ == "__main__":
    unittest.main()
