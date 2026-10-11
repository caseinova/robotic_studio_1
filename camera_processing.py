"""RGB-D perception for the cube carrying ArUco DICT_4X4_50 marker 0.

Inputs are aligned, rectified RGB-D images with metric optical-axis depth.
All camera coordinates use +X right, +Y down, +Z forward. No robot commands
or visual servo controller are implemented here.

Example::

    processor = CameraProcessor()
    frame = robot.capture_rgbd()
    observation = processor.process(frame)
    display_rgb = processor.annotate(frame.rgb, observation, frame.intrinsics)
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING

import cv2
import numpy as np

if TYPE_CHECKING:
    from robot_model import RGBDFrame


@dataclass(frozen=True)
class CubeObservation:
    """Features and estimated poses from one frame, suitable for later PBVS.

    Transforms named A_from_B map coordinates in B into A, in meters.
    The marker origin is its center; +X points right on the canonical tag,
    +Y up, and +Z outward from the tagged cube face. The cube frame shares
    these axes with its origin at the cube center. These are estimates from
    pixels and depth, not MuJoCo's object pose.
    """

    marker_id: int
    timestamp: float
    corners: np.ndarray  # (4, 2), canonical top-left clockwise, pixels
    center_pixel: np.ndarray  # (2,), mean of marker corners
    camera_from_marker: np.ndarray  # (4, 4)
    camera_from_cube: np.ndarray  # (4, 4)
    world_from_cube: np.ndarray  # (4, 4), using supplied camera extrinsics
    mask: np.ndarray  # bool (H, W), visible cube surface including tag
    bounding_box: tuple[int, int, int, int]  # x0, y0, x1, y1 (exclusive end)
    points_camera: np.ndarray  # (N, 3), visible segmented surface in meters
    reprojection_error: float  # RMS pixels, after depth refinement
    depth_plane_error: float  # RMS meters on the tag plane


class CameraProcessor:
    """Detect a known tagged cube and segment its visible RGB-D surface.

    Default dimensions match scene.xml: an 80 mm marker centered on a
    100 mm cube, with the label surface 0.15 mm above the cube face.
    A missing tag, insufficient depth, or inconsistent pose returns None;
    observations are never reused from an earlier frame.
    """

    def __init__(
        self, *, marker_id: int = 0, marker_size: float = 0.08,
        cube_size: float = 0.10, label_offset: float = 0.00015,
        depth_tolerance: float = 0.003, max_reprojection_error: float = 3.0,
    ):
        if not isinstance(marker_id, int) or not 0 <= marker_id < 50:
            raise ValueError("marker_id must be an integer in DICT_4X4_50 (0..49)")
        for name, value in (("marker_size", marker_size), ("cube_size", cube_size),
                            ("depth_tolerance", depth_tolerance),
                            ("max_reprojection_error", max_reprojection_error)):
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if marker_size > cube_size or not np.isfinite(label_offset) or label_offset < 0:
            raise ValueError("marker must fit on the cube and label_offset must be non-negative")
        self.marker_id = marker_id
        self.marker_size = marker_size
        self.cube_size = cube_size
        self.label_offset = label_offset
        self.depth_tolerance = depth_tolerance
        self.max_reprojection_error = max_reprojection_error
        parameters = cv2.aruco.DetectorParameters()
        parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        self.detector = cv2.aruco.ArucoDetector(
            cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50), parameters
        )
        half = marker_size / 2
        self._marker_points = np.array([
            [-half, half, 0], [half, half, 0],
            [half, -half, 0], [-half, -half, 0],
        ], dtype=np.float64)

    def process(self, frame: "RGBDFrame") -> CubeObservation | None:
        """Extract a fresh cube observation, or None if not reliably visible."""
        rgb, depth = frame.rgb, frame.depth
        if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
            raise ValueError("rgb must be an RGB uint8 array shaped (H, W, 3)")
        if depth.shape != rgb.shape[:2]:
            raise ValueError("depth must be registered to RGB with shape (H, W)")
        intrinsics = np.asarray(frame.intrinsics, dtype=np.float64)
        if (intrinsics.shape != (3, 3) or not np.all(np.isfinite(intrinsics))
                or intrinsics[0, 0] <= 0 or intrinsics[1, 1] <= 0
                or not np.allclose(intrinsics[2], [0, 0, 1])):
            raise ValueError("intrinsics must be a finite pinhole calibration matrix")
        if (frame.world_from_camera.shape != (4, 4)
                or not np.all(np.isfinite(frame.world_from_camera))):
            raise ValueError("world_from_camera must be a finite 4x4 transform")
        corners, ids, _ = self.detector.detectMarkers(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY))
        if ids is None:
            return None
        candidates = [corner.reshape(4, 2).astype(np.float64)
                      for corner, tag_id in zip(corners, ids.reshape(-1))
                      if int(tag_id) == self.marker_id]
        # If duplicate IDs appear, prefer the largest visible tag that has depth.
        candidates.sort(key=lambda c: abs(cv2.contourArea(c.astype(np.float32))), reverse=True)
        for corner in candidates:
            observation = self._extract(frame, intrinsics, corner)
            if observation is not None:
                return observation
        return None

    def _extract(self, frame, intrinsics, corners) -> CubeObservation | None:
        success, rvec, tvec = cv2.solvePnP(
            self._marker_points, corners, intrinsics, None, flags=cv2.SOLVEPNP_ITERATIVE
        )
        if (not success or not np.all(np.isfinite(tvec))
                or not np.all(np.isfinite(rvec)) or tvec[2, 0] <= 0):
            return None
        projected, _ = cv2.projectPoints(self._marker_points, rvec, tvec, intrinsics, None)
        if self._pixel_error(projected.reshape(4, 2), corners) > self.max_reprojection_error:
            return None

        height, width = frame.depth.shape
        rows, columns = np.indices((height, width))
        pixels = np.stack((columns, rows, np.ones_like(rows)), axis=-1)
        rays = pixels @ np.linalg.inv(intrinsics).T
        valid = frame.valid_depth & np.isfinite(frame.depth) & (frame.depth > 0)
        points = rays * np.where(valid, frame.depth, 0)[..., None]
        tag_mask = np.zeros((height, width), dtype=np.uint8)
        cv2.fillConvexPoly(tag_mask, np.rint(corners).astype(np.int32), 1)
        # Avoid mixing the marker face with neighboring pixels at its boundary.
        tag_mask = cv2.erode(tag_mask, np.ones((5, 5), dtype=np.uint8))
        samples = points[(tag_mask > 0) & valid]
        if len(samples) < 16:
            return None
        samples = samples[::max(1, len(samples) // 2048)]
        # Depth resolves the ambiguity of nearly front-facing planar tags.
        center = samples.mean(axis=0)
        _, _, axes = np.linalg.svd(samples - center, full_matrices=False)
        normal = axes[-1]
        residual = np.abs((samples - center) @ normal)
        inliers = samples[residual <= self.depth_tolerance]
        if len(inliers) < max(16, int(0.8 * len(samples))):
            return None
        center = inliers.mean(axis=0)
        _, singular_values, axes = np.linalg.svd(inliers - center, full_matrices=False)
        if singular_values[1] < 1e-6:
            return None
        normal = axes[-1]
        plane_error = float(np.sqrt(np.mean(((inliers - center) @ normal) ** 2)))
        corner_rays = np.column_stack((corners, np.ones(4))) @ np.linalg.inv(intrinsics).T
        denominators = corner_rays @ normal
        if np.any(np.abs(denominators) < 1e-8):
            return None
        corner_depth = float(center @ normal) / denominators
        if np.any(corner_depth <= 0) or not np.all(np.isfinite(corner_depth)):
            return None
        measured_corners = corner_rays * corner_depth[:, None]
        edge_lengths = np.linalg.norm(measured_corners - np.roll(measured_corners, -1, axis=0), axis=1)
        if np.any(np.abs(edge_lengths - self.marker_size) > self.marker_size * 0.1):
            return None
        # Rigid alignment fits the known marker square to its measured 3D corners.
        translation = measured_corners.mean(axis=0)
        covariance = self._marker_points.T @ (measured_corners - translation)
        u, _, vt = np.linalg.svd(covariance)
        correction = np.diag([1, 1, np.linalg.det(vt.T @ u.T)])
        rotation = vt.T @ correction @ u.T
        camera_from_marker = np.eye(4)
        camera_from_marker[:3, :3] = rotation
        camera_from_marker[:3, 3] = translation
        refined_rvec, _ = cv2.Rodrigues(rotation)
        projected, _ = cv2.projectPoints(
            self._marker_points, refined_rvec, translation, intrinsics, None
        )
        reprojection_error = self._pixel_error(projected.reshape(4, 2), corners)
        if reprojection_error > self.max_reprojection_error:
            return None

        camera_from_cube = camera_from_marker.copy()
        camera_from_cube[:3, 3] -= rotation[:, 2] * (self.cube_size / 2 + self.label_offset)
        points_cube = (points - camera_from_cube[:3, 3]) @ rotation
        mask = valid & np.all(np.abs(points_cube) <= self.cube_size / 2 + self.depth_tolerance, axis=-1)
        ys, xs = np.nonzero(mask)
        if len(xs) == 0:
            return None
        return CubeObservation(
            marker_id=self.marker_id, timestamp=float(frame.timestamp),
            corners=corners.copy(), center_pixel=corners.mean(axis=0),
            camera_from_marker=camera_from_marker, camera_from_cube=camera_from_cube,
            world_from_cube=frame.world_from_camera @ camera_from_cube,
            mask=mask, bounding_box=(int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1),
            points_camera=points[mask].copy(), reprojection_error=reprojection_error,
            depth_plane_error=plane_error,
        )

    @staticmethod
    def _pixel_error(projected, measured) -> float:
        return float(np.sqrt(np.mean(np.sum((projected - measured) ** 2, axis=1))))

    def annotate(self, rgb, observation: CubeObservation | None, intrinsics) -> np.ndarray:
        """Return an RGB copy with segmentation, tag corners, and pose axes."""
        image = rgb.copy()
        if observation is None:
            cv2.putText(image, "Tagged cube not detected", (12, 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 80, 80), 2)
            return image
        mask = observation.mask
        image[mask] = (0.65 * image[mask] + 0.35 * np.array([0, 255, 80])).astype(np.uint8)
        cv2.aruco.drawDetectedMarkers(
            image, [observation.corners.astype(np.float32).reshape(1, 4, 2)],
            np.array([[observation.marker_id]], dtype=np.int32)
        )
        rotation_vector, _ = cv2.Rodrigues(observation.camera_from_marker[:3, :3])
        # OpenCV chooses axis colors for BGR; swap them back for the RGB API.
        bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.drawFrameAxes(bgr, intrinsics, None, rotation_vector,
                          observation.camera_from_marker[:3, 3], self.marker_size / 2)
        image = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        z = observation.camera_from_cube[2, 3]
        cv2.putText(image, f"Cube ID {observation.marker_id} | center Z {z:.3f} m", (12, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        return image
