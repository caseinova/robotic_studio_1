"""Load the bundled UR5e model and its MuJoCo state.

Control access lives in interface.py; simulation execution lives in simulate.py.
"""

from pathlib import Path
from dataclasses import dataclass
from contextlib import nullcontext

import mujoco
import numpy as np


MODEL_DIRECTORY = (
    Path(__file__).resolve().parent
    / "universal_robots_ur5e"
    / "universal_robots_ur5e"
)


@dataclass(frozen=True)
class RGBDFrame:
    """Registered RGB and metric depth from one simulation state.

    rgb: uint8 (H, W, 3), RGB channel order.
    depth: float32 (H, W), optical-axis Z in meters, not ray distance.
    intrinsics: 3x3 pinhole K with pixel coordinates (column, row); integer
        coordinates denote pixel centers. The simulated lens has no distortion.
    world_from_camera: 4x4 transform from optical coordinates to world meters.
        Optical axes are +X right, +Y down, +Z forward.
    timestamp: simulation time in seconds, shared by both images and the pose.
    depth_range: near/far clipping distances in meters; background is at far.

    Capture must run on the simulation thread, without concurrent state writes.
    """

    rgb: np.ndarray
    depth: np.ndarray
    intrinsics: np.ndarray
    world_from_camera: np.ndarray
    timestamp: float
    depth_range: tuple[float, float]

    @property
    def valid_depth(self) -> np.ndarray:
        """Mask out background and clipped or nonfinite depth pixels."""
        near, far = self.depth_range
        return (
            np.isfinite(self.depth)
            & (self.depth > near)
            & (self.depth < far * (1 - 1e-5))
        )


class RobotModel:
    """UR5e model definition and state.

    ``model`` and ``data`` are public MuJoCo objects for controllers,
    estimators, rendering, and other simulation integrations.
    """

    camera_name = "wrist_camera"

    joint_names = (
        "shoulder_pan_joint",
        "shoulder_lift_joint",
        "elbow_joint",
        "wrist_1_joint",
        "wrist_2_joint",
        "wrist_3_joint",
    )
    actuator_names = (
        "shoulder_pan", "shoulder_lift", "elbow", "wrist_1", "wrist_2", "wrist_3"
    )

    def __init__(self, *, include_scene: bool = True):
        self.model_path = MODEL_DIRECTORY / (
            "scene.xml" if include_scene else "ur5e.xml"
        )
        self.model = mujoco.MjModel.from_xml_path(str(self.model_path))
        self.data = mujoco.MjData(self.model)
        self.camera_id = self.model.camera(self.camera_name).id
        self.joint_ids = np.array([
            self.model.joint(name).id for name in self.joint_names
        ])
        self.actuator_ids = np.array([
            self.model.actuator(name).id for name in self.actuator_names
        ])
        self._qpos_indices = self.model.jnt_qposadr[self.joint_ids]
        self._qvel_indices = self.model.jnt_dofadr[self.joint_ids]
        self.reset()

    @property
    def joint_positions(self) -> np.ndarray:
        """Return a copy of the six joint angles in radians."""
        return self.data.qpos[self._qpos_indices].copy()

    @property
    def joint_velocities(self) -> np.ndarray:
        """Return a copy of the six joint velocities in radians/second."""
        return self.data.qvel[self._qvel_indices].copy()

    def reset(self) -> None:
        """Reset time, state, and actuator targets to the model's home pose."""
        mujoco.mj_resetDataKeyframe(
            self.model, self.data, self.model.key("home").id
        )
        mujoco.mj_forward(self.model, self.data)

    def capture_image(
        self, *, width: int = 640, height: int = 480, depth: bool = False
    ) -> np.ndarray:
        """Render the wrist camera at the current robot state.

        Returns RGB uint8 pixels shaped (height, width, 3), or a floating
        point depth image shaped (height, width) with optical-axis Z in meters.
        Use capture_rgbd() for a synchronized pair with calibration and pose.
        The camera moves with wrist_3_link and looks along the tool axis.
        Rendering requires an OpenGL context but does not open a viewer.
        """
        self._prepare_camera(width, height)
        with mujoco.Renderer(self.model, height=height, width=width) as renderer:
            if depth:
                renderer.enable_depth_rendering()
            renderer.update_scene(self.data, camera=self.camera_id)
            return renderer.render().copy()

    def _prepare_camera(self, width: int, height: int) -> None:
        for name, value in (("width", width), ("height", height)):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        # Support resolutions larger than MuJoCo's default offscreen buffer.
        self.model.vis.global_.offwidth = max(self.model.vis.global_.offwidth, width)
        self.model.vis.global_.offheight = max(self.model.vis.global_.offheight, height)
        # Refresh transforms after stepping or direct changes to qpos.
        mujoco.mj_forward(self.model, self.data)

    def capture_rgbd(
        self, *, width: int = 640, height: int = 480,
        renderer: mujoco.Renderer | None = None,
    ) -> RGBDFrame:
        """Capture aligned RGB/depth and calibration without advancing time.

        Both passes use one frozen render scene and the same pinhole camera.
        This is an ideal simulated RGB-D sensor with no noise or distortion.
        An optional renderer for this model is reused and remains caller-owned.
        """
        if renderer is not None and (renderer.width != width or renderer.height != height):
            raise ValueError("renderer dimensions must match the requested RGB-D dimensions")
        self._prepare_camera(width, height)
        timestamp = float(self.data.time)
        world_from_camera = np.eye(4)
        # MuJoCo camera axes: +X right, +Y up, -Z forward.
        world_from_camera[:3, :3] = (
            self.data.cam_xmat[self.camera_id].reshape(3, 3)
            @ np.diag([1.0, -1.0, -1.0])
        )
        world_from_camera[:3, 3] = self.data.cam_xpos[self.camera_id]
        focal = height / (2 * np.tan(np.deg2rad(self.model.cam_fovy[self.camera_id]) / 2))
        intrinsics = np.array([
            [focal, 0, (width - 1) / 2],
            [0, focal, (height - 1) / 2],
            [0, 0, 1],
        ])
        extent = self.model.stat.extent
        depth_range = (
            float(self.model.vis.map.znear * extent),
            float(self.model.vis.map.zfar * extent),
        )
        context = (nullcontext(renderer) if renderer is not None
                   else mujoco.Renderer(self.model, height=height, width=width))
        with context as active_renderer:
            active_renderer.disable_depth_rendering()
            active_renderer.disable_segmentation_rendering()
            active_renderer.update_scene(self.data, camera=self.camera_id)
            rgb = active_renderer.render().copy()
            active_renderer.enable_depth_rendering()
            try:
                depth = active_renderer.render().copy()
            finally:
                active_renderer.disable_depth_rendering()
        return RGBDFrame(rgb, depth, intrinsics, world_from_camera, timestamp, depth_range)

