"""Load the bundled UR5e model and its MuJoCo state.

Control access lives in interface.py; simulation execution lives in simulate.py.
"""

from pathlib import Path
from contextlib import nullcontext

import mujoco
import numpy as np

from robot_state import RGBDFrame


MODEL_DIRECTORY = (
    Path(__file__).resolve().parent
    / "universal_robots_ur5e"
    / "universal_robots_ur5e"
)


class RobotModel:
    """UR5e model definition and state.

    ``model`` and ``data`` are public MuJoCo objects for controllers,
    estimators, rendering, and other simulation integrations.
    """

    camera_name = "external_camera"
    camera_names = ("external_camera",)
    end_effector_marker_id = 1
    end_effector_marker_size = 0.08  # black square side length, meters
    gripper_joint_names = ("gripper_left_joint", "gripper_right_joint")
    gripper_actuator_names = ("gripper_left", "gripper_right")
    gripper_max_width = 0.12  # meters between the inner finger faces

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

    def __init__(self, *, include_scene: bool = True, camera_name: str = "external_camera"):
        if camera_name not in self.camera_names:
            raise ValueError("camera_name must be external_camera")
        self.camera_name = camera_name
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
        self.gripper_joint_ids = np.array([
            self.model.joint(name).id for name in self.gripper_joint_names
        ])
        self.gripper_actuator_ids = np.array([
            self.model.actuator(name).id for name in self.gripper_actuator_names
        ])
        self.grasp_site_id = self.model.site("grasp_site").id
        self.reset()

    @property
    def joint_positions(self) -> np.ndarray:
        """Return a copy of the six joint angles in radians."""
        return self.data.qpos[self._qpos_indices].copy()

    @property
    def joint_velocities(self) -> np.ndarray:
        """Return a copy of the six joint velocities in radians/second."""
        return self.data.qvel[self._qvel_indices].copy()

    @property
    def gripper_width(self) -> float:
        """Measured inner finger opening in meters (0 to 0.12 nominally)."""
        return float(np.sum(self.data.qpos[self.model.jnt_qposadr[self.gripper_joint_ids]]))

    @property
    def grasp_pose(self) -> np.ndarray:
        """World-from-grasp-site pose; +Z is the finger approach direction.

        The site is 105 mm beyond the gripper mount, between the fingertips.
        Call after forwarding the simulation state for current transforms.
        """
        pose = np.eye(4)
        pose[:3, :3] = self.data.site_xmat[self.grasp_site_id].reshape(3, 3)
        pose[:3, 3] = self.data.site_xpos[self.grasp_site_id]
        return pose

    def reset(self) -> None:
        """Reset time, state, and actuator targets to the model's home pose."""
        mujoco.mj_resetDataKeyframe(
            self.model, self.data, self.model.key("home").id
        )
        mujoco.mj_forward(self.model, self.data)

    def capture_image(
        self, *, width: int = 640, height: int = 480, depth: bool = False
    ) -> np.ndarray:
        """Render the selected RGB-D camera at the current robot state.

        Returns RGB uint8 pixels shaped (height, width, 3), or a floating
        point depth image shaped (height, width) with optical-axis Z in meters.
        Use capture_rgbd() for a synchronized pair with calibration and pose.
        external_camera is fixed above and diagonally in front-right of the robot.
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

