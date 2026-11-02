"""Run the UR3e simulation with a viewer or headlessly."""

import argparse
import time
import math
from contextlib import ExitStack
from dataclasses import replace

import mujoco
import numpy as np

from simulation.interface import MujocoInterface
from simulation.viewer import launch_viewer
from simulation.robot_model import RobotModel
from simulation.simulation_config import DEFAULT_CONFIG_PATH, SimulationConfig
from common.controller_config import DEFAULT_CONTROLLER_CONFIG_PATH, ControllerConfig
from simulation.noise_config import DEFAULT_NOISE_CONFIG_PATH, NoiseConfig
from common.estimator import DEFAULT_ESTIMATOR_CONFIG_PATH, EKFConfig
from simulation.plots import SimulationPlots
from common.controller import start_pickup_or_reapproach


def _pickup_observation(result):
    """Hand the picker the same filtered target used by the approach."""
    observation, estimate = result.observation, result.pose_estimate
    if observation is None or estimate is None:
        return observation
    # The marker/cube offset is fixed by perception's geometric model.
    cube_from_marker = np.linalg.inv(observation.camera_from_cube) @ observation.camera_from_marker
    return replace(observation, camera_from_cube=estimate.camera_from_cube.copy(),
                   camera_from_marker=estimate.camera_from_cube @ cube_from_marker,
                   world_from_cube=estimate.world_from_cube.copy())


class Simulation:
    """Own physics execution and expose the portable controller interface."""

    def __init__(self, robot: RobotModel | None = None, *, config: SimulationConfig | None = None,
                 controller_config: ControllerConfig | None = None, pickup: bool = False,
                 noise_config: NoiseConfig | None = None, estimator_config: EKFConfig | None = None,
                 camera_name: str | None = None):
        self.robot = robot if robot is not None else RobotModel(camera_name=camera_name or RobotModel.camera_name)
        if robot is not None and camera_name is not None:
            if camera_name not in RobotModel.camera_names:
                raise ValueError("camera_name must be external_camera")
            self.robot.camera_name = camera_name
            self.robot.camera_id = self.robot.model.camera(camera_name).id
        self.config = config if config is not None else SimulationConfig()
        self.controller_config = controller_config or ControllerConfig(tool_clearance=.28)
        self.estimator_config = estimator_config or EKFConfig()
        # Preserve an explicitly supplied custom model unless settings are supplied.
        if robot is None or config is not None:
            cube_id = mujoco.mj_name2id(self.robot.model, mujoco.mjtObj.mjOBJ_BODY, "cube")
            if cube_id < 0:
                raise ValueError("simulation experiment configuration requires a scene with a cube")
            self.robot.model.body_pos[cube_id] = self.config.cube_position
            self.robot.model.body_quat[cube_id] = self.config.cube_quaternion
            mujoco.mj_setConst(self.robot.model, self.robot.data)
            self.robot.reset()
        self.pickup = pickup
        if pickup:
            # A free joint is required for real contact-driven lifting. Compile
            # a pickup scene without changing the bundled approach-only model.
            spec = mujoco.MjSpec.from_file(str(self.robot.model_path))
            cube = spec.body("cube")
            cube_id = self.robot.model.body("cube").id
            cube.pos = self.robot.model.body_pos[cube_id]
            cube.quat = self.robot.model.body_quat[cube_id]
            cube.add_freejoint(name="cube_free_joint")
            # Stiffer tangential contacts prevent slow numerical sliding while
            # the fingers carry the cube, without welding it to the gripper.
            spec.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
            spec.option.impratio = 10
            self.robot.model = spec.compile()
            self.robot.data = mujoco.MjData(self.robot.model)
            self.robot.reset()
        self.interface = MujocoInterface(self.robot, noise_config=noise_config)
        self.plots = SimulationPlots()
        self._plot_kinematics = None

    def _record_control(self, result):
        """Reconstruct measured motion in a separate telemetry workspace."""
        from simulation.kinematics import MujocoKinematics

        if self._plot_kinematics is None:
            self._plot_kinematics = MujocoKinematics(self.robot)
        self.plots.record(self.interface.read_state(), self._plot_kinematics, result)

    def reset(self) -> None:
        self.robot.reset()
        self.interface.reset_noise()
        self.plots.reset()
        # Reset targets are model commands, not noisy measurements of home.
        targets = self.robot.model.key("home").ctrl[self.robot.actuator_ids].copy()
        self.interface.set_joint_positions(targets)

    def step(self, steps: int = 1) -> None:
        """Advance by a non-negative number of fixed physics timesteps."""
        if isinstance(steps, bool) or not isinstance(steps, int) or steps < 0:
            raise ValueError("steps must be a non-negative integer")
        for _ in range(steps):
            self.interface.prepare_step()
            mujoco.mj_step(self.robot.model, self.robot.data)

    def run_controlled(self, steps: int, *, tool_clearance: float | None = None):
        """Run approach and optional pickup headlessly, stopping at completion."""
        from common.controller import ApproachController, CubePickupController
        from simulation.kinematics import MujocoKinematics

        if isinstance(steps, bool) or not isinstance(steps, int) or steps < 0:
            raise ValueError("steps must be a non-negative integer")
        controller = ApproachController(self.interface, MujocoKinematics(self.robot), tool_clearance=tool_clearance,
                                        controller_config=self.controller_config,
                                        estimator_config=self.estimator_config)
        self.plots.reset()
        pickup_controller = (CubePickupController(self.interface, controller.kinematics,
                             cube_size=self.robot.cube_size,
                             speed=controller.settings.pickup_linear_speed_m_s,
                             angular_speed=controller.settings.pickup_angular_speed_rad_s,
                             frame_max_age_intervals=controller.settings.frame_max_age_intervals,
                             grasp_height_offset_m=controller.settings.grasp_height_offset_m,
                             pickup_dwell_seconds=controller.settings.pickup_dwell_seconds,
                             pickup_dwell_radius_m=controller.settings.pickup_dwell_radius_m)
                             if self.pickup else None)
        next_update, result = 0.0, None
        try:
            with self.interface:
                for _ in range(steps):
                    if self.interface.timestamp + 1e-9 >= next_update:
                        if pickup_controller is not None and pickup_controller.phase != "idle":
                            result = pickup_controller.update()
                        else:
                            result = controller.update()
                            if result.status == "reached" and pickup_controller is not None:
                                result = start_pickup_or_reapproach(controller, pickup_controller,
                                                                   _pickup_observation(result))
                        self._record_control(result)
                        next_update = self.interface.timestamp + controller.control_dt
                        if (result.status == "reached"
                                or pickup_controller is not None and pickup_controller.finished):
                            break
                    self.step()
        finally:
            if pickup_controller is None or pickup_controller.phase != "complete":
                self.interface.set_joint_velocities([0.0] * 6)
        return result

    def launch_viewer(self, *, camera: bool = False, servo: bool = False,
                      tool_clearance: float | None = None, plots: bool = False,
                      annotations: bool = True) -> None:
        """Show the model overview and optionally a separate fixed-camera RGB window.

        Close either window, or press Q/Escape in the camera window, to stop.
        """
        import mujoco.viewer

        servo = servo or self.pickup
        if plots and not servo:
            raise ValueError("plots require --servo or --pick")
        self.plots.reset()
        controller = None
        pickup_controller = None
        if servo:
            from common.controller import ApproachController, CubePickupController
            from simulation.kinematics import MujocoKinematics

            controller = ApproachController(self.interface, MujocoKinematics(self.robot), tool_clearance=tool_clearance,
                                            controller_config=self.controller_config,
                                            estimator_config=self.estimator_config)
            if self.pickup:
                pickup_controller = CubePickupController(self.interface, controller.kinematics,
                    cube_size=self.robot.cube_size,
                    speed=controller.settings.pickup_linear_speed_m_s,
                    angular_speed=controller.settings.pickup_angular_speed_rad_s,
                    frame_max_age_intervals=controller.settings.frame_max_age_intervals,
                    grasp_height_offset_m=controller.settings.grasp_height_offset_m,
                    pickup_dwell_seconds=controller.settings.pickup_dwell_seconds,
                    pickup_dwell_radius_m=controller.settings.pickup_dwell_radius_m)
            camera = True
        label = "PBVS+MPC" if self.controller_config.mode == "pbvs_mpc" else "PBVS"
        print("Cube approach controller: " + (label if servo else "OFF (enable with --servo)"), flush=True)
        with ExitStack() as stack:
            if controller is not None:
                stack.callback(self.interface.set_joint_velocities, [0.0] * 6)
            if plots:
                self.plots.start_live()
            cv2 = None
            if camera:
                import cv2
                from common.camera_processing import TaggedCubeProcessor

                processor = controller.processor if controller is not None else TaggedCubeProcessor(
                    cube_size=self.robot.cube_size, marker_size=self.robot.cube_marker_size)

            viewer = stack.enter_context(
                launch_viewer(self.robot.model, self.robot.data, self.config.viewer)
            )
            window_name = "UR3e " + self.robot.camera_name.replace("_", " ") + " - RGB"
            if camera:
                stack.enter_context(self.interface)
                cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
                stack.callback(cv2.destroyWindow, window_name)
            next_camera_frame = 0.0
            next_control_update = 0.0
            next_plot_refresh = 0.0
            last_control = None
            last_status = None
            # Rendering/optimization can cost more than one 2 ms physics step.
            # Advance a complete control/display interval before drawing again.
            interval = (controller.control_dt if controller is not None
                        else 1 / 30 if camera else 1 / 60)
            frame_steps = max(1, round(interval / self.robot.model.opt.timestep))
            simulated_interval = frame_steps * self.robot.model.opt.timestep
            while viewer.is_running():
                start = time.perf_counter()
                self.step(frame_steps)
                viewer.sync()
                control_due = controller is not None and self.interface.timestamp + 1e-9 >= next_control_update
                display_due = camera and start >= next_camera_frame
                if camera and (display_due or control_due):
                    frame = self.interface.capture_rgbd()
                    if control_due:
                        if pickup_controller is not None and pickup_controller.phase != "idle":
                            last_control = pickup_controller.update()
                        else:
                            last_control = controller.update(frame)
                            if last_control.status == "reached" and pickup_controller is not None:
                                last_control = start_pickup_or_reapproach(controller, pickup_controller,
                                                                         _pickup_observation(last_control))
                        self._record_control(last_control)
                        if plots and time.perf_counter() >= next_plot_refresh:
                            self.plots.refresh()
                            next_plot_refresh = time.perf_counter() + 0.5
                        active_label = "Pickup" if last_control.status.startswith("pickup_") else label
                        status_key = (last_control.status, last_control.diagnostic, last_control.tool_pose_source)
                        if status_key != last_status:
                            error = (f"{last_control.position_error:.4f} m"
                                     if math.isfinite(last_control.position_error) else "unavailable")
                            clearance = (f"{last_control.minimum_clearance:.4f} m"
                                         if math.isfinite(last_control.minimum_clearance) else "not evaluated")
                            details = f" | {last_control.diagnostic}" if last_control.diagnostic else ""
                            if last_control.tool_pose_source:
                                details += f" | tool pose: {last_control.tool_pose_source}"
                            print(f"{active_label}: {last_control.status} | error {error} | "
                                  f"clearance {clearance}{details}", flush=True)
                            last_status = status_key
                        next_control_update = self.interface.timestamp + controller.control_dt
                        observation = (processor.process(frame) if last_control.status.startswith("pickup_")
                                       else last_control.observation)
                    else:
                        observation = processor.process(frame)
                    rgb = (processor.annotate(frame.rgb, observation, frame.intrinsics)
                           if annotations else frame.rgb)
                    # The calibration marker is distinct from the cube target.
                    tag_corners, tag_ids = [], None
                    if annotations:
                        tag_corners, tag_ids, _ = processor.detector.detectMarkers(
                            cv2.cvtColor(frame.rgb, cv2.COLOR_RGB2GRAY))
                    if tag_ids is not None:
                        for corners, marker_id in zip(tag_corners, tag_ids.reshape(-1)):
                            if int(marker_id) == self.robot.end_effector_marker_id:
                                cv2.aruco.drawDetectedMarkers(
                                    rgb, [corners], np.array([[int(marker_id)]], dtype=np.int32))
                    if annotations and last_control is not None:
                        error = (f"{last_control.position_error:.3f} m"
                                 if math.isfinite(last_control.position_error) else "unavailable")
                        cv2.putText(rgb, f"{active_label}: {last_control.status} | error {error}",
                                    (12, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                        if last_control.tool_pose_source:
                            cv2.putText(rgb, f"Tool pose: {last_control.tool_pose_source}",
                                        (12, 94), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    elif annotations:
                        cv2.putText(rgb, "Controller OFF: restart with --servo", (12, 48),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 80), 1)
                    cv2.imshow(window_name, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
                    key = cv2.waitKey(1) & 0xFF
                    if key in (27, ord("q"), ord("Q")):
                        break
                    if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                        break
                    next_camera_frame = start + 1 / 30
                remaining = simulated_interval - (time.perf_counter() - start)
                if remaining > 0:
                    time.sleep(remaining)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="simulation experiment JSON file")
    parser.add_argument("--controller-config", default=str(DEFAULT_CONTROLLER_CONFIG_PATH), help="controller selection JSON file")
    parser.add_argument("--noise-config", default=str(DEFAULT_NOISE_CONFIG_PATH), help="simulation measurement noise JSON file")
    parser.add_argument("--estimator-config", default=str(DEFAULT_ESTIMATOR_CONFIG_PATH), help="relative pose EKF JSON file")
    parser.add_argument("--no-noise", action="store_true", help="disable measurement noise for this run")
    parser.add_argument("--headless", action="store_true", help="run without a viewer")
    parser.add_argument("--camera", action="store_true", help="show the selected camera RGB window alongside the model viewer")
    parser.add_argument("--annotations", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--camera-name", choices=RobotModel.camera_names, default=None,
                        help="fixed RGB-D camera for viewing/calibration")
    parser.add_argument("--plots", action="store_true", help="show live control plots in viewer mode and save PNG/CSV results")
    parser.add_argument("--plot-dir", default=None, help="save control plots and CSV to this directory (enables export)")
    parser.add_argument("--steps", type=int, default=1000, help="headless simulation steps")
    parser.add_argument("--servo", action="store_true", help="approach the detected cube using the selected controller mode")
    parser.add_argument("--pick", action="store_true", help="approach, descend, close the gripper, and lift the movable cube (implies --servo)")
    parser.add_argument("--tool-clearance", type=float, default=None, help="override controller config tool standoff in meters (minimum 0.16)")
    args = parser.parse_args()
    args.servo = args.servo or args.pick
    if args.steps < 0:
        parser.error("--steps must be non-negative")
    if args.headless and args.camera:
        parser.error("--camera requires a viewer; omit --headless")
    if (args.plots or args.plot_dir is not None) and not args.servo:
        parser.error("--plots/--plot-dir require --servo or --pick")
    camera_name = args.camera_name or RobotModel.camera_name
    try:
        config = SimulationConfig.load(args.config)
        controller_config = ControllerConfig.load(args.controller_config)
        noise_config = NoiseConfig.load(args.noise_config)
        estimator_config = EKFConfig.load(args.estimator_config)
        if args.no_noise:
            noise_config = replace(noise_config, enabled=False)
        if args.tool_clearance is not None:
            controller_config = replace(controller_config, tool_clearance=args.tool_clearance)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    simulation = Simulation(config=config, controller_config=controller_config, pickup=args.pick,
                            noise_config=noise_config, estimator_config=estimator_config, camera_name=camera_name)
    print(f"Camera: {camera_name}", flush=True)
    print(f"Experiment cube position (world meters): {config.cube_position}", flush=True)
    print(f"Measurement noise: {'ON' if noise_config.enabled else 'OFF'} (seed {noise_config.seed})", flush=True)
    if args.servo:
        print(f"Controller mode: {controller_config.mode}", flush=True)
        print(f"Tool pose source: {controller_config.tool_pose_source}; target yaw: {controller_config.tool_yaw_deg:g} degrees", flush=True)
        print(f"Relative pose EKF: {'ON' if estimator_config.enabled else 'OFF'}", flush=True)
    # Export even if the viewer is interrupted, retaining the recorded run.
    try:
        _run_simulation(simulation, args)
    finally:
        if args.plots or args.plot_dir is not None:
            paths = simulation.plots.save(args.plot_dir or "simulation_plots")
            if paths:
                print(f"Saved control plots and CSV to {paths[0].parent.resolve()}", flush=True)
        simulation.plots.close()


def _run_simulation(simulation, args):
    """Run without mixing plot export and viewer/headless control flow."""
    if args.headless:
        if args.servo:
            result = simulation.run_controlled(args.steps, tool_clearance=args.tool_clearance)
            if result is not None:
                position = f"{result.position_error:.4f} m" if math.isfinite(result.position_error) else "unavailable"
                rotation = f"{result.rotation_error:.4f} rad" if math.isfinite(result.rotation_error) else "unavailable"
                clearance = f"{result.minimum_clearance:.4f} m" if math.isfinite(result.minimum_clearance) else "not evaluated"
                details = f"; {result.diagnostic}" if result.diagnostic else ""
                if result.tool_pose_source:
                    details += f"; tool pose: {result.tool_pose_source}"
                print(f"Controller: {result.status}; position error {position}; "
                      f"rotation error {rotation}; clearance {clearance}{details}")
        else:
            simulation.step(args.steps)
        print(f"Simulated {simulation.interface.timestamp:.3f} seconds")
        print("Joint positions (radians):", simulation.interface.joint_positions)
    else:
        simulation.launch_viewer(camera=args.camera, servo=args.servo,
                                 tool_clearance=args.tool_clearance, plots=args.plots,
                                 annotations=args.annotations)


if __name__ == "__main__":
    main()
