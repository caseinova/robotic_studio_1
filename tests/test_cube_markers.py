"""Rendered face redundancy, common control frame, and EKF continuity."""

from dataclasses import replace
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from common.camera_processing import TaggedCubeProcessor
from common.estimator import RelativePoseEKF
from simulation.robot_model import RobotModel


class CubeMarkersTests(unittest.TestCase):
    @staticmethod
    def label_name(marker_id):
        return 'cube_aruco' if marker_id == 0 else f'cube_aruco_{marker_id}'

    def test_each_face_alone_recovers_same_cube_and_reference_frame(self):
        robot = RobotModel()
        processor = TaggedCubeProcessor()
        for marker_id, mount in processor.cube_from_markers.items():
            with self.subTest(marker_id=marker_id):
                for other_id in processor.cube_from_markers:
                    robot.model.geom(self.label_name(other_id)).rgba[3] = float(other_id == marker_id)
                # Put this face on top; tests may access truth, perception may not.
                rotation = mount[:3, :3].T
                q = Rotation.from_matrix(rotation).as_quat()
                robot.model.body('cube').quat[:] = [q[3], *q[:3]]
                frame = robot.capture_rgbd()
                observation = processor.process(frame)
                self.assertIsNotNone(observation, processor.last_status)
                self.assertEqual([m.marker_id for m in observation.marker_observations], [marker_id])
                np.testing.assert_allclose(observation.world_from_cube[:3, 3],
                                           robot.model.body('cube').pos, atol=.002)
                np.testing.assert_allclose(observation.world_from_cube[:3, :3], rotation, atol=.012)
                np.testing.assert_allclose(observation.camera_from_marker,
                    observation.camera_from_cube @ processor.cube_from_markers[0], atol=1e-12)

    def test_multiple_faces_fuse_and_top_occlusion_keeps_ekf_tracking(self):
        robot = RobotModel()
        # Use a view exposing two faces to isolate fusion from task occlusion.
        robot.model.cam_fovy[robot.camera_id] = 50
        robot.model.cam_pos[robot.camera_id] = [-.074, 1.05, .95]
        robot.model.cam_quat[robot.camera_id] = [-.01358635, -.00634502, .42309666, .90596047]
        processor = TaggedCubeProcessor()
        estimator = RelativePoseEKF()
        frame = robot.capture_rgbd()
        observation = processor.process(frame)
        self.assertIn(0, {m.marker_id for m in observation.marker_observations})
        self.assertGreaterEqual(len(observation.marker_observations), 2)
        initial = estimator.step(frame, observation)
        self.assertTrue(initial.measurement_accepted)
        robot.model.geom('cube_aruco').rgba[3] = 0
        frame = replace(robot.capture_rgbd(), timestamp=.05)
        observation = processor.process(frame)
        self.assertNotIn(0, [m.marker_id for m in observation.marker_observations])
        self.assertTrue(observation.marker_observations)
        estimate = estimator.step(frame, observation)
        self.assertTrue(estimate.measurement_accepted)
        np.testing.assert_allclose(estimate.world_from_cube[:3, 3], initial.world_from_cube[:3, 3], atol=.002)
        for marker_id in processor.cube_from_markers:
            robot.model.geom(self.label_name(marker_id)).rgba[3] = 0
        self.assertIsNone(processor.process(robot.capture_rgbd()))

    def test_approach_reaches_goal_with_top_tag_hidden(self):
        from common.controller_config import ControllerConfig
        from simulation.noise_config import NoiseConfig
        from simulation.simulate import Simulation

        simulation = Simulation(controller_config=ControllerConfig(
            mode='pure_pbvs', tool_pose_source='tag_only'),
            noise_config=NoiseConfig(enabled=False))
        simulation.robot.model.geom('cube_aruco').rgba[3] = 0
        result = simulation.run_controlled(6000)
        self.assertEqual(result.status, 'reached', result.diagnostic)
        self.assertEqual(result.tool_observation.marker_id, 1)
        self.assertNotIn(0, [m.marker_id for m in result.observation.marker_observations])


if __name__ == '__main__':
    unittest.main()
