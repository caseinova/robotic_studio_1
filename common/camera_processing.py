"""RGB-D perception for the tagged cube and calibrated tool marker.

Inputs are aligned, rectified RGB-D images with metric optical-axis depth.
All camera coordinates use +X right, +Y down, +Z forward. No robot commands
or visual servo controller are implemented here.

Example::

    processor = CameraProcessor()
    frame = interface.capture_rgbd()
    observation = processor.process(frame)
    display_rgb = processor.annotate(frame.rgb, observation, frame.intrinsics)
"""

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

if TYPE_CHECKING:
    from common.robot_state import RGBDFrame


@dataclass(frozen=True)
class MarkerObservation:
    """A validated RGB-D tag pose, without assuming a cube behind it."""

    marker_id: int
    timestamp: float
    corners: np.ndarray
    camera_from_marker: np.ndarray
    reprojection_error: float
    depth_plane_error: float


@dataclass(frozen=True)
class CubeObservation:
    """Features and estimated poses from one frame, suitable for later PBVS.

    Transforms named A_from_B map coordinates in B into A, in meters.
    The marker origin is its center; +X points right on the canonical tag,
    +Y up, and +Z outward from the canonical top face. camera_from_marker
    always refers to that top reference frame in TaggedCubeProcessor; corners
    and marker_id identify a visible representative face. Actual measured
    face poses are available in marker_observations. The cube frame shares
    the top frame's axes with its origin at the cube center. These are estimates from
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
    marker_observations: tuple[MarkerObservation, ...] = ()


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
        # Fit the complete marker boundary rather than chasing local shadow
        # gradients at individual corners. The dictionary remains ArUco 4x4_50.
        parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_APRILTAG
        self.detector = cv2.aruco.ArucoDetector(
            cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50), parameters
        )
        # Adaptive thresholding can separate a small tag border from a dark
        # gripper immediately behind it when AprilTag's contour fit cannot.
        fallback_parameters = cv2.aruco.DetectorParameters()
        fallback_parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        fallback_parameters.cornerRefinementWinSize = 2
        self._fallback_detector = cv2.aruco.ArucoDetector(
            cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50), fallback_parameters
        )
        half = marker_size / 2
        self._marker_points = np.array([
            [-half, half, 0], [half, half, 0],
            [half, -half, 0], [-half, -half, 0],
        ], dtype=np.float64)
        self.last_status = "not_processed"
        self.last_marker_corners = None

    def process(self, frame: "RGBDFrame") -> CubeObservation | None:
        """Extract a fresh cube observation, or None if not reliably visible."""
        return self._process(frame, marker_only=False)

    def process_marker(self, frame: "RGBDFrame") -> MarkerObservation | None:
        """Measure this tag alone, e.g. a calibrated tag on the tool."""
        return self._process(frame, marker_only=True)

    def _process(self, frame, *, marker_only):
        self.last_status = "processing"
        self.last_marker_corners = None
        intrinsics = self._validate_frame(frame)
        rgb = frame.rgb
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        corners, ids, _ = self.detector.detectMarkers(gray)
        # AprilTag refinement uses pixel-edge coordinates; RGB-D rays and K
        # use pixel centers. Subpixel refinement already uses pixel centers.
        corner_offset = 0.5
        if ids is None or self.marker_id not in ids:
            corners, ids, _ = self._fallback_detector.detectMarkers(gray)
            corner_offset = 0.0
        if ids is None:
            return self._reject("tag_not_detected")
        candidates = [corner.reshape(4, 2).astype(np.float64) - corner_offset
                      for corner, tag_id in zip(corners, ids.reshape(-1))
                      if int(tag_id) == self.marker_id]
        # If duplicate IDs appear, prefer the largest visible tag that has depth.
        candidates.sort(key=lambda c: abs(cv2.contourArea(c.astype(np.float32))), reverse=True)
        if not candidates:
            return self._reject("target_id_not_detected")
        for corner in candidates:
            self.last_marker_corners = corner.copy()
            observation = self._extract(frame, intrinsics, corner, marker_only=marker_only)
            if observation is not None:
                self.last_status = "detected"
                return observation
        return None

    @staticmethod
    def _validate_frame(frame):
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
        return intrinsics

    def _extract(self, frame, intrinsics, corners, *, marker_only=False):
        self.last_metric_edges = None
        success, rvec, tvec = cv2.solvePnP(
            self._marker_points, corners, intrinsics, None, flags=cv2.SOLVEPNP_ITERATIVE
        )
        if (not success or not np.all(np.isfinite(tvec))
                or not np.all(np.isfinite(rvec)) or tvec[2, 0] <= 0):
            return self._reject("pnp_pose_invalid")
        projected, _ = cv2.projectPoints(self._marker_points, rvec, tvec, intrinsics, None)
        if self._pixel_error(projected.reshape(4, 2), corners) > self.max_reprojection_error:
            return self._reject("pnp_reprojection_error")

        height, width = frame.depth.shape
        valid = frame.valid_depth & np.isfinite(frame.depth) & (frame.depth > 0)
        tag_mask = np.zeros((height, width), dtype=np.uint8)
        cv2.fillConvexPoly(tag_mask, np.rint(corners).astype(np.int32), 1)
        # Avoid mixing the marker face with neighboring pixels at its boundary.
        tag_mask = cv2.erode(tag_mask, np.ones((5, 5), dtype=np.uint8))
        # Marker-only fits need only the eroded tag pixels, not a full-frame cloud.
        ys, xs = np.nonzero((tag_mask > 0) & valid)
        rays = np.column_stack((xs, ys, np.ones(len(xs)))) @ np.linalg.inv(intrinsics).T
        samples = rays * frame.depth[ys, xs, None]
        if len(samples) < 16:
            return self._reject("insufficient_valid_tag_depth")
        samples = samples[::max(1, len(samples) // 2048)]
        # Depth resolves the ambiguity of nearly front-facing planar tags.
        fitted_plane = self._fit_depth_plane(samples)
        if fitted_plane is None:
            return self._reject("tag_depth_degenerate")
        normal, plane_distance = fitted_plane
        signed_residual = samples @ normal - plane_distance
        # Estimate random scatter robustly: a fixed 3 mm gate rejects most
        # legitimate samples when the camera's depth sigma is several mm.
        # Cap the adaptive gate at 10% of marker size so arbitrary rough or
        # mixed surfaces cannot expand the gate without limit. Metric-size
        # and reprojection checks below still validate the fitted pose.
        median_residual = np.median(signed_residual)
        depth_sigma = 1.4826 * np.median(np.abs(signed_residual - median_residual))
        plane_tolerance = max(self.depth_tolerance,
                              min(3 * depth_sigma, self.marker_size * 0.1))
        residual = np.abs(signed_residual - median_residual)
        inliers = samples[residual <= plane_tolerance]
        if len(inliers) < max(16, int(0.8 * len(samples))):
            return self._reject("tag_depth_not_planar")
        center = inliers.mean(axis=0)
        _, singular_values, _ = np.linalg.svd(inliers - center, full_matrices=False)
        if singular_values[1] < 1e-6:
            return self._reject("tag_depth_degenerate")
        fitted_plane = self._fit_depth_plane(inliers)
        if fitted_plane is None:
            return self._reject("tag_depth_degenerate")
        normal, plane_distance = fitted_plane
        plane_error = float(np.sqrt(np.mean((inliers @ normal - plane_distance) ** 2)))
        corner_rays = np.column_stack((corners, np.ones(4))) @ np.linalg.inv(intrinsics).T
        denominators = corner_rays @ normal
        if np.any(np.abs(denominators) < 1e-8):
            return self._reject("tag_plane_parallel_to_view")
        corner_depth = plane_distance / denominators
        if np.any(corner_depth <= 0) or not np.all(np.isfinite(corner_depth)):
            return self._reject("tag_corner_depth_invalid")
        measured_corners = corner_rays * corner_depth[:, None]
        edge_lengths = np.linalg.norm(measured_corners - np.roll(measured_corners, -1, axis=0), axis=1)
        if np.any(np.abs(edge_lengths - self.marker_size) > self.marker_size * 0.1):
            self.last_metric_edges = edge_lengths.copy()
            return self._reject("tag_metric_size_inconsistent")
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
            return self._reject(f"depth_pose_reprojection_error: {reprojection_error:.2f} px")

        if marker_only:
            return MarkerObservation(self.marker_id, float(frame.timestamp),
                corners.copy(), camera_from_marker, reprojection_error, plane_error)

        camera_from_cube = camera_from_marker.copy()
        camera_from_cube[:3, 3] -= rotation[:, 2] * (self.cube_size / 2 + self.label_offset)
        rows, columns = np.indices((height, width))
        pixels = np.stack((columns, rows, np.ones_like(rows)), axis=-1)
        rays = pixels @ np.linalg.inv(intrinsics).T
        points = rays * np.where(valid, frame.depth, 0)[..., None]
        points_cube = (points - camera_from_cube[:3, 3]) @ rotation
        mask = valid & np.all(np.abs(points_cube) <= self.cube_size / 2 + self.depth_tolerance, axis=-1)
        ys, xs = np.nonzero(mask)
        if len(xs) == 0:
            return self._reject("cube_surface_not_found")
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
    def _fit_depth_plane(points):
        """Fit 1/Z = a*(X/Z) + b*(Y/Z) + c on known pixel rays.

        Optical-depth noise moves a point along its ray. Cartesian SVD treats
        X/Y/Z scatter as isotropic and biases the normal of an oblique tag.
        Ray coordinates remain exact under depth noise; inverse-depth fitting
        avoids that slope bias while retaining metric residual validation.
        """
        z = points[:, 2]
        rays = np.column_stack((points[:, :2] / z[:, None], np.ones(len(points))))
        coefficients, _, rank, _ = np.linalg.lstsq(rays, 1 / z, rcond=None)
        magnitude = np.linalg.norm(coefficients)
        if rank < 3 or not np.isfinite(magnitude) or magnitude < 1e-12:
            return None
        return coefficients / magnitude, 1 / magnitude

    @staticmethod
    def _pixel_error(projected, measured) -> float:
        return float(np.sqrt(np.mean(np.sum((projected - measured) ** 2, axis=1))))

    def _reject(self, reason: str) -> None:
        self.last_status = reason
        return None

    def annotate(self, rgb, observation: CubeObservation | None, intrinsics) -> np.ndarray:
        """Return an RGB copy with segmentation, tag corners, and pose axes."""
        image = rgb.copy()
        if observation is None:
            found = self.last_marker_corners is not None
            if found:
                cv2.aruco.drawDetectedMarkers(
                    image, [self.last_marker_corners.astype(np.float32).reshape(1, 4, 2)],
                    np.array([[self.marker_id]], dtype=np.int32)
                )
            message = "Tag found; cube pose rejected" if found else "Tagged cube not detected"
            cv2.putText(image, message, (12, 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 80, 80), 2)
            cv2.putText(image, self.last_status.replace("_", " "), (12, 72),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 180, 80), 1)
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


class TaggedCubeProcessor(CameraProcessor):
    """Six calibrated faces produce one cube pose per camera exposure.

    ID 0 is the top; IDs 2..6 are +Y, +X, -Y, -X, and bottom.
    ID 1 belongs exclusively to the tool. Every pose is converted to the
    cube center before fusion, so changing visible faces does not change
    the control target. The EKF consumes the fused measurement once; shared
    camera errors must not be counted as six independent measurements.
    """

    def __init__(self, *, cube_from_markers=None, reference_marker_id=0, **kwargs):
        super().__init__(**kwargs)
        self.cube_from_markers = {}
        faces = [(0, (0, 0, 1), (0, 0, 0)),
                 (2, (0, 1, 0), (-90, 0, 0)),
                 (3, (1, 0, 0), (0, 90, 0)),
                 (4, (0, -1, 0), (90, 0, 0)),
                 (5, (-1, 0, 0), (0, -90, 0)),
                 (6, (0, 0, -1), (180, 0, 0))]
        for marker_id, normal, angles in faces:
            pose = np.eye(4)
            pose[:3, :3] = Rotation.from_euler('xyz', angles, degrees=True).as_matrix()
            pose[:3, 3] = np.asarray(normal) * (self.cube_size / 2 + self.label_offset)
            self.cube_from_markers[marker_id] = pose
        if cube_from_markers is not None:
            from common.geometry import _pose

            self.cube_from_markers = {}
            for key, value in cube_from_markers.items():
                marker_id = int(key)
                if str(marker_id) != str(key) or not 0 <= marker_id < 50:
                    raise ValueError("cube marker IDs must be integers in DICT_4X4_50")
                self.cube_from_markers[marker_id] = _pose(value, f"cube_from_markers[{key}]")
        if type(reference_marker_id) is not int or reference_marker_id not in self.cube_from_markers:
            raise ValueError("reference_marker_id must identify a configured cube marker")
        self.reference_marker_id = reference_marker_id

    def process(self, frame):
        self._validate_frame(frame)
        self.metric_size_rejections = []
        self.last_status = "tag_not_detected"
        self.last_marker_corners = None
        gray = cv2.cvtColor(frame.rgb, cv2.COLOR_RGB2GRAY)
        corners, ids, _ = self.detector.detectMarkers(gray)
        detected = {} if ids is None else {
            int(i): c.reshape(4, 2).astype(float) - 0.5
            for c, i in zip(corners, ids.reshape(-1))}
        fallback_corners, fallback_ids, _ = self._fallback_detector.detectMarkers(gray)
        if fallback_ids is not None:
            for c, i in zip(fallback_corners, fallback_ids.reshape(-1)):
                detected.setdefault(int(i), c.reshape(4, 2).astype(float))
        measurements, poses, weights = [], [], []
        for marker_id, mount in self.cube_from_markers.items():
            if marker_id not in detected:
                continue
            self.last_marker_corners = detected[marker_id].copy()
            measurement = self._extract(frame, frame.intrinsics, detected[marker_id], marker_only=True)
            if measurement is None:
                if self.last_status == "tag_metric_size_inconsistent" and self.last_metric_edges is not None:
                    self.metric_size_rejections.append((marker_id, self.last_metric_edges.copy()))
                continue
            measurement = replace(measurement, marker_id=marker_id)
            inverse_mount = np.eye(4)
            inverse_mount[:3, :3] = mount[:3, :3].T
            inverse_mount[:3, 3] = -mount[:3, :3].T @ mount[:3, 3]
            poses.append(measurement.camera_from_marker @ inverse_mount)
            measurements.append(measurement)
            # Larger tags constrain pose better; downweight poor image fits.
            weights.append(abs(cv2.contourArea(measurement.corners.astype(np.float32)))
                           / (1 + measurement.reprojection_error ** 2))
        if not measurements:
            return None
        # Discard faces inconsistent with the best validated measurement.
        best = int(np.argmax(weights))
        keep = [i for i, pose in enumerate(poses)
                if np.linalg.norm(pose[:3, 3] - poses[best][:3, 3]) <= 0.02
                and Rotation.from_matrix(pose[:3, :3] @ poses[best][:3, :3].T).magnitude() <= 0.35]
        weights = np.asarray([weights[i] for i in keep])
        weights /= weights.sum()
        measurements = tuple(measurements[i] for i in keep)
        poses = [poses[i] for i in keep]
        pose = np.eye(4)
        pose[:3, 3] = sum(w * p[:3, 3] for w, p in zip(weights, poses))
        u, _, vt = np.linalg.svd(sum(w * p[:3, :3] for w, p in zip(weights, poses)))
        pose[:3, :3] = u @ np.diag([1, 1, np.linalg.det(u @ vt)]) @ vt
        rows, columns = np.indices(frame.depth.shape)
        rays = np.stack((columns, rows, np.ones_like(rows)), axis=-1) @ np.linalg.inv(frame.intrinsics).T
        valid = frame.valid_depth & np.isfinite(frame.depth) & (frame.depth > 0)
        points = rays * np.where(valid, frame.depth, 0)[..., None]
        points_cube = (points - pose[:3, 3]) @ pose[:3, :3]
        mask = valid & np.all(np.abs(points_cube) <= self.cube_size / 2 + self.depth_tolerance, axis=-1)
        ys, xs = np.nonzero(mask)
        if not len(xs):
            return self._reject('cube_surface_not_found')
        representative = measurements[int(np.argmax(weights))]
        self.last_status = 'detected'
        # Preserve the canonical top-marker control frame even if it is hidden.
        return CubeObservation(representative.marker_id, float(frame.timestamp),
            representative.corners, representative.corners.mean(axis=0),
            pose @ self.cube_from_markers[self.reference_marker_id], pose, frame.world_from_camera @ pose,
            mask, (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1),
            points[mask].copy(), float(weights @ np.array([m.reprojection_error for m in measurements])),
            float(weights @ np.array([m.depth_plane_error for m in measurements])), measurements)

    def annotate(self, rgb, observation, intrinsics):
        image = super().annotate(rgb, observation, intrinsics)
        if observation is not None:
            for marker in observation.marker_observations:
                cv2.aruco.drawDetectedMarkers(image,
                    [marker.corners.astype(np.float32).reshape(1, 4, 2)],
                    np.array([[marker.marker_id]], dtype=np.int32))
            ids = ','.join(str(m.marker_id) for m in observation.marker_observations)
            cv2.putText(image, f'Cube faces used: {ids}', (12, 72),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        return image
