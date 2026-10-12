"""Headless RGB-D perception checks against the rendered UR5e scene."""

from dataclasses import replace
import unittest

import mujoco
import numpy as np

from camera_processing import CameraProcessor
from robot_model import RobotModel


class CameraProcessingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.robot = RobotModel()
        cls.frame = cls.robot.capture_rgbd()

    def test_pose_and_segmentation_from_images(self):
        processor = CameraProcessor()
        observation = processor.process(self.frame)
        self.assertIsNotNone(observation)
        np.testing.assert_allclose(observation.world_from_cube[:3, 3],
                                   [-0.074, 0.492, 0.05], atol=0.001)
        np.testing.assert_allclose(observation.world_from_cube[:3, 2], [0, 0, 1], atol=0.001)
        self.assertEqual(observation.marker_id, 0)
        self.assertEqual(observation.timestamp, self.frame.timestamp)
        self.assertLess(observation.reprojection_error, 1)
        self.assertLess(observation.depth_plane_error, 1e-5)
        self.assertEqual(observation.points_camera.shape, (observation.mask.sum(), 3))
        self.assertGreater(self._mask_iou(self.robot, observation.mask), 0.98)
        annotated = processor.annotate(self.frame.rgb, observation, self.frame.intrinsics)
        self.assertFalse(np.array_equal(annotated, self.frame.rgb))
        self.assertFalse(np.shares_memory(annotated, self.frame.rgb))

    def test_tilted_cube(self):
        robot = RobotModel()
        angle = 0.18
        robot.model.body("cube").quat[:] = [np.cos(angle / 2), np.sin(angle / 2), 0, 0]
        # Raise the tilted fixed cube so its lowest corner remains on the floor.
        robot.model.body("cube").pos[2] = 0.05 * (np.cos(angle) + np.sin(angle))
        frame = robot.capture_rgbd()
        observation = CameraProcessor().process(frame)
        self.assertIsNotNone(observation)
        np.testing.assert_allclose(observation.world_from_cube[:3, 3],
                                   robot.model.body("cube").pos, atol=0.002)
        np.testing.assert_allclose(observation.world_from_cube[:3, 2],
                                   [0, -np.sin(angle), np.cos(angle)], atol=0.01)
        self.assertGreater(self._mask_iou(robot, observation.mask), 0.95)

    def test_no_stale_observation_on_tag_loss(self):
        processor = CameraProcessor()
        self.assertIsNotNone(processor.process(self.frame))
        blank = replace(self.frame, rgb=np.full_like(self.frame.rgb, 255))
        self.assertIsNone(processor.process(blank))
        self.assertEqual(processor.last_status, "tag_not_detected")
        self.assertIsNone(processor.last_marker_corners)
        self.assertIsNone(CameraProcessor(marker_id=1).process(self.frame))

    def test_invalid_depth_is_rejected(self):
        processor = CameraProcessor()
        for value in (0, np.nan, np.inf, self.frame.depth_range[1]):
            with self.subTest(depth=value):
                frame = replace(self.frame, depth=np.full_like(self.frame.depth, value))
                self.assertIsNone(processor.process(frame))
                self.assertEqual(processor.last_status, "insufficient_valid_tag_depth")

    def test_inconsistent_depth_is_rejected(self):
        # Wrong metric scale cannot fit the known 80 mm tag dimensions.
        frame = replace(self.frame, depth=self.frame.depth * 2)
        self.assertIsNone(CameraProcessor().process(frame))

    def test_configuration_and_registration_validation(self):
        for config in ({"marker_size": 0}, {"marker_id": 50}, {"cube_size": 0.01}):
            with self.assertRaises(ValueError):
                CameraProcessor(**config)
        with self.assertRaises(ValueError):
            CameraProcessor().process(replace(self.frame, depth=self.frame.depth[:10]))

    def test_renderer_reuse_keeps_rgb_mode_and_owned_resources(self):
        with mujoco.Renderer(self.robot.model, height=480, width=640) as renderer:
            first = self.robot.capture_rgbd(renderer=renderer)
            second = self.robot.capture_rgbd(renderer=renderer)
            # Color rendering may vary slightly as lighting buffers refresh.
            self.assertIsNotNone(CameraProcessor().process(first))
            self.assertIsNotNone(CameraProcessor().process(second))
            self.assertFalse(np.shares_memory(first.rgb, second.rgb))
            np.testing.assert_allclose(first.depth, second.depth, atol=1e-5)
            self.assertEqual(renderer.render().shape, (480, 640, 3))
            with self.assertRaises(ValueError):
                self.robot.capture_rgbd(width=320, renderer=renderer)

    @staticmethod
    def _mask_iou(robot, mask):
        # Ground truth is used only in this test, never by the perception code.
        with mujoco.Renderer(robot.model, height=480, width=640) as renderer:
            renderer.enable_segmentation_rendering()
            renderer.update_scene(robot.data, camera=robot.camera_id)
            labels = renderer.render()
        cube_ids = [robot.model.geom("cube_geom").id, robot.model.geom("cube_aruco").id]
        truth = np.isin(labels[..., 0], cube_ids) & (labels[..., 1] == mujoco.mjtObj.mjOBJ_GEOM)
        return np.count_nonzero(mask & truth) / np.count_nonzero(mask | truth)


if __name__ == "__main__":
    unittest.main()
