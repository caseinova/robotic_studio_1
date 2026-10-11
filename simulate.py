"""Run the UR5e simulation with a viewer or headlessly."""

import argparse
import time
from contextlib import ExitStack

import mujoco

from interface import MujocoInterface
from robot_model import RobotModel
from simulation_config import DEFAULT_CONFIG_PATH, SimulationConfig


class Simulation:
    """Own physics execution and expose the portable controller interface."""

    def __init__(self, robot: RobotModel | None = None, *, config: SimulationConfig | None = None):
        self.robot = robot if robot is not None else RobotModel()
        self.config = config if config is not None else SimulationConfig()
        # Preserve an explicitly supplied custom model unless settings are supplied.
        if robot is None or config is not None:
            cube_id = mujoco.mj_name2id(self.robot.model, mujoco.mjtObj.mjOBJ_BODY, "cube")
            if cube_id < 0:
                raise ValueError("simulation experiment configuration requires a scene with a cube")
            self.robot.model.body_pos[cube_id] = self.config.cube_position
            self.robot.model.body_quat[cube_id] = self.config.cube_quaternion
            mujoco.mj_setConst(self.robot.model, self.robot.data)
            self.robot.reset()
        self.interface = MujocoInterface(self.robot)

    def reset(self) -> None:
        self.robot.reset()
        self.interface.set_joint_positions(self.robot.joint_positions)

    def step(self, steps: int = 1) -> None:
        """Advance by a non-negative number of fixed physics timesteps."""
        if isinstance(steps, bool) or not isinstance(steps, int) or steps < 0:
            raise ValueError("steps must be a non-negative integer")
        for _ in range(steps):
            self.interface.prepare_step()
            mujoco.mj_step(self.robot.model, self.robot.data)

    def run_controlled(self, steps: int, *, tool_clearance: float | None = None):
        """Run vision/PBVS/MPC headlessly; stop early when the approach is reached."""
        from controller import ApproachController

        if isinstance(steps, bool) or not isinstance(steps, int) or steps < 0:
            raise ValueError("steps must be a non-negative integer")
        clearance = self.config.tool_clearance if tool_clearance is None else tool_clearance
        controller = ApproachController(self.robot, self.interface, tool_clearance=clearance)
        next_update, result = 0.0, None
        try:
            with mujoco.Renderer(self.robot.model, height=480, width=640) as renderer:
                for _ in range(steps):
                    if self.robot.data.time + 1e-9 >= next_update:
                        result = controller.update(self.robot.capture_rgbd(renderer=renderer))
                        next_update = self.robot.data.time + controller.mpc.config.dt
                        if result.status == "reached":
                            break
                    self.step()
        finally:
            self.interface.set_joint_velocities([0.0] * 6)
        return result

    def launch_viewer(self, *, camera: bool = False, servo: bool = False,
                      tool_clearance: float | None = None) -> None:
        """Show the model overview and optionally a separate wrist RGB window.

        Close either window, or press Q/Escape in the camera window, to stop.
        """
        import mujoco.viewer

        controller = None
        if servo:
            from controller import ApproachController

            clearance = self.config.tool_clearance if tool_clearance is None else tool_clearance
            controller = ApproachController(self.robot, self.interface, tool_clearance=clearance)
            camera = True
        print("Cube approach controller: " + ("ON" if servo else "OFF (enable with --servo)"), flush=True)
        with ExitStack() as stack:
            if controller is not None:
                stack.callback(self.interface.set_joint_velocities, [0.0] * 6)
            cv2 = None
            if camera:
                import cv2
                from camera_processing import CameraProcessor

                processor = controller.processor if controller is not None else CameraProcessor()

            viewer = stack.enter_context(
                mujoco.viewer.launch_passive(self.robot.model, self.robot.data)
            )
            renderer = None
            window_name = "UR5e wrist camera - RGB"
            if camera:
                renderer = stack.enter_context(
                    mujoco.Renderer(self.robot.model, height=480, width=640)
                )
                cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
                stack.callback(cv2.destroyWindow, window_name)
            next_camera_frame = 0.0
            next_control_update = 0.0
            last_control = None
            last_status = None
            # Rendering/optimization can cost more than one 2 ms physics step.
            # Advance a complete control/display interval before drawing again.
            interval = (controller.mpc.config.dt if controller is not None
                        else 1 / 30 if camera else 1 / 60)
            frame_steps = max(1, round(interval / self.robot.model.opt.timestep))
            simulated_interval = frame_steps * self.robot.model.opt.timestep
            while viewer.is_running():
                start = time.perf_counter()
                self.step(frame_steps)
                viewer.sync()
                control_due = controller is not None and self.robot.data.time + 1e-9 >= next_control_update
                display_due = camera and start >= next_camera_frame
                if renderer is not None and (display_due or control_due):
                    frame = self.robot.capture_rgbd(renderer=renderer)
                    if control_due:
                        last_control = controller.update(frame)
                        if last_control.status != last_status:
                            print(f"MPC: {last_control.status} | error {last_control.position_error:.4f} m | "
                                  f"clearance {last_control.minimum_clearance:.4f} m", flush=True)
                            last_status = last_control.status
                        next_control_update = self.robot.data.time + controller.mpc.config.dt
                        observation = last_control.observation
                    else:
                        observation = processor.process(frame)
                    rgb = processor.annotate(frame.rgb, observation, frame.intrinsics)
                    if last_control is not None:
                        cv2.putText(rgb, f"MPC: {last_control.status} | error {last_control.position_error:.3f} m",
                                    (12, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    else:
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
    parser.add_argument("--headless", action="store_true", help="run without a viewer")
    parser.add_argument("--camera", action="store_true", help="show a wrist-camera RGB window alongside the model viewer")
    parser.add_argument("--steps", type=int, default=1000, help="headless simulation steps")
    parser.add_argument("--servo", action="store_true", help="approach the detected cube using PBVS and obstacle-constrained MPC")
    parser.add_argument("--tool-clearance", type=float, default=None, help="override config tool standoff in meters (minimum 0.16)")
    args = parser.parse_args()
    if args.steps < 0:
        parser.error("--steps must be non-negative")
    if args.headless and args.camera:
        parser.error("--camera requires a viewer; omit --headless")
    try:
        config = SimulationConfig.load(args.config)
        if args.tool_clearance is not None:
            from dataclasses import replace

            config = replace(config, tool_clearance=args.tool_clearance)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    simulation = Simulation(config=config)
    print(f"Experiment cube position (world meters): {config.cube_position}", flush=True)
    if args.headless:
        if args.servo:
            result = simulation.run_controlled(args.steps, tool_clearance=args.tool_clearance)
            if result is not None:
                print(f"Controller: {result.status}; position error {result.position_error:.4f} m; "
                      f"rotation error {result.rotation_error:.4f} rad; clearance {result.minimum_clearance:.4f} m")
        else:
            simulation.step(args.steps)
        print(f"Simulated {simulation.robot.data.time:.3f} seconds")
        print("Joint positions (radians):", simulation.interface.joint_positions)
    else:
        simulation.launch_viewer(camera=args.camera, servo=args.servo, tool_clearance=args.tool_clearance)


if __name__ == "__main__":
    main()
