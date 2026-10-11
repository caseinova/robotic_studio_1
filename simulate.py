"""Run the UR5e simulation with a viewer or headlessly."""

import argparse
import time
from contextlib import ExitStack

import mujoco

from interface import MujocoInterface
from robot_model import RobotModel


class Simulation:
    """Own physics execution and expose the portable controller interface."""

    def __init__(self, robot: RobotModel | None = None):
        self.robot = robot if robot is not None else RobotModel()
        self.interface = MujocoInterface(self.robot)

    def reset(self) -> None:
        self.robot.reset()

    def step(self, steps: int = 1) -> None:
        """Advance by a non-negative number of fixed physics timesteps."""
        if isinstance(steps, bool) or not isinstance(steps, int) or steps < 0:
            raise ValueError("steps must be a non-negative integer")
        for _ in range(steps):
            mujoco.mj_step(self.robot.model, self.robot.data)

    def launch_viewer(self, *, camera: bool = False) -> None:
        """Show the model overview and optionally a separate wrist RGB window.

        Close either window, or press Q/Escape in the camera window, to stop.
        """
        import mujoco.viewer

        with ExitStack() as stack:
            cv2 = None
            if camera:
                import cv2
                from camera_processing import CameraProcessor

                processor = CameraProcessor()

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
            while viewer.is_running():
                start = time.perf_counter()
                self.step()
                viewer.sync()
                if renderer is not None and start >= next_camera_frame:
                    frame = self.robot.capture_rgbd(renderer=renderer)
                    observation = processor.process(frame)
                    rgb = processor.annotate(frame.rgb, observation, frame.intrinsics)
                    cv2.imshow(window_name, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
                    key = cv2.waitKey(1) & 0xFF
                    if key in (27, ord("q"), ord("Q")):
                        break
                    if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                        break
                    next_camera_frame = start + 1 / 30
                remaining = self.robot.model.opt.timestep - (time.perf_counter() - start)
                if remaining > 0:
                    time.sleep(remaining)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--headless", action="store_true", help="run without a viewer")
    parser.add_argument("--camera", action="store_true", help="show a wrist-camera RGB window alongside the model viewer")
    parser.add_argument("--steps", type=int, default=1000, help="headless simulation steps")
    args = parser.parse_args()
    if args.steps < 0:
        parser.error("--steps must be non-negative")
    if args.headless and args.camera:
        parser.error("--camera requires a viewer; omit --headless")
    simulation = Simulation()
    if args.headless:
        simulation.step(args.steps)
        print(f"Simulated {simulation.robot.data.time:.3f} seconds")
        print("Joint positions (radians):", simulation.interface.joint_positions)
    else:
        simulation.launch_viewer(camera=args.camera)


if __name__ == "__main__":
    main()
