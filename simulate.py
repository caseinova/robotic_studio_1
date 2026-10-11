"""Run the UR5e simulation with a viewer or headlessly."""

import argparse
import time

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
        """Run the viewer, optionally looking through the wrist RGB camera."""
        import mujoco.viewer

        with mujoco.viewer.launch_passive(self.robot.model, self.robot.data) as viewer:
            if camera:
                with viewer.lock():
                    viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
                    viewer.cam.fixedcamid = self.robot.camera_id
            while viewer.is_running():
                start = time.perf_counter()
                self.step()
                viewer.sync()
                remaining = self.robot.model.opt.timestep - (time.perf_counter() - start)
                if remaining > 0:
                    time.sleep(remaining)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--headless", action="store_true", help="run without a viewer")
    parser.add_argument("--camera", action="store_true", help="show the live wrist-camera RGB view")
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
