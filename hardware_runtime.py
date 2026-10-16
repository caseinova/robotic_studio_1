"""Build configured devices and run portable controllers on a monotonic clock."""

from contextlib import ExitStack
from dataclasses import replace
import time

from controller import ApproachController, CubePickupController
from hardware_devices import URRTDE, OnRobotRG, custom_device
from interface import HardwareInterface


def open_hardware(config, *, clock=time.monotonic):
    """Transfer successfully constructed devices to one owning interface."""
    from controller_config import ControllerConfig

    config.validate_ready(ControllerConfig(mode="pure_pbvs", tool_pose_source="fk"), task="observe")
    with ExitStack() as stack:
        def managed(device, methods):
            closer = getattr(device, "close", None)
            if callable(closer):
                stack.callback(closer)
            if any(not callable(getattr(device, name, None)) for name in methods):
                raise ValueError(f"device adapter does not implement {methods}")
            return device

        camera = managed(custom_device(config.camera, clock), ("capture", "close"))
        gripper = managed(OnRobotRG(config.gripper, clock) if config.gripper["driver"] == "onrobot_rg"
                          else custom_device(config.gripper, clock), ("read_width", "command_width", "close"))
        robot = managed(URRTDE(config.robot, config.control, clock) if config.robot["driver"] == "ur_rtde"
                        else custom_device(config.robot, clock),
                        ("read_joint_state", "command_velocity", "command_position", "stop", "close"))
        stack.callback(robot.stop)
        interface = HardwareInterface(config, robot, camera, gripper, clock=clock)
        stack.pop_all()
        return interface


def _pickup_target(result):
    observation = result.observation
    if result.pose_estimate is None:
        return observation
    import numpy as np

    estimate = result.pose_estimate
    cube_from_reference = np.linalg.inv(observation.camera_from_cube) @ observation.camera_from_marker
    return replace(observation, camera_from_cube=estimate.camera_from_cube.copy(),
        camera_from_marker=estimate.camera_from_cube @ cube_from_reference,
        world_from_cube=estimate.world_from_cube.copy())


def run_hardware(config, controller_config, estimator_config, *, task="servo", steps=6000,
                 camera_view=False, clock=time.monotonic, sleep=time.sleep,
                 interface_factory=open_hardware):
    """Stop on completion, interruption, timing failure, or transport error.

    Existing pickup behavior is preserved: its target is latched at handoff.
    The command worker independently expires commands if perception/planning
    blocks; this loop never substitutes delivery time for acquisition time.
    """
    if task not in ("observe", "servo", "pick"):
        raise ValueError("hardware task must be observe, servo, or pick")
    if type(steps) is not int or steps < 0:
        raise ValueError("steps must be a non-negative integer")
    config.validate_ready(controller_config, task=task)
    processor = config.build_processor()
    kinematics = config.build_kinematics()
    settings = config.build_mpc_config()
    result = None
    cv2 = None
    if camera_view:
        import cv2
    try:
        with interface_factory(config, clock=clock) as interface:
            controller = (None if task == "observe" else ApproachController(interface, kinematics,
                config=settings, controller_config=controller_config, estimator_config=estimator_config,
                processor=processor))
            pickup = (CubePickupController(interface, kinematics, cube_size=processor.cube_size,
                                           control_dt=settings.dt, velocity_limit=settings.velocity_limit,
                                           clearance=settings.clearance) if task == "pick" else None)
            started = deadline = clock()
            last_status = None
            for _ in range(steps):
                if clock() - started >= config.control["max_run_seconds"]:
                    raise TimeoutError("hardware run exceeded max_run_seconds")
                cycle_start = clock()
                if pickup is not None and pickup.phase != "idle":
                    interface.require_camera_for_motion = False
                    result = pickup.update()
                else:
                    interface.require_camera_for_motion = True
                    frame = interface.capture_rgbd()
                    if controller is None:
                        interface.read_state()
                        observation = processor.process(frame)
                    else:
                        result = controller.update(frame)
                        observation = result.observation
                        if pickup is not None and result.status == "reached":
                            interface.require_camera_for_motion = False
                            pickup.start(_pickup_target(result))
                            result = pickup.update()
                    if cv2 is not None:
                        rgb = processor.annotate(frame.rgb, observation, frame.intrinsics)
                        cv2.imshow("Hardware RGB-D", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
                if cv2 is not None:
                    key = cv2.waitKey(1) & 255
                    if key in (27, ord('q')) or cv2.getWindowProperty("Hardware RGB-D", cv2.WND_PROP_VISIBLE) < 1:
                        break
                if clock() - cycle_start > config.control["max_cycle_time"]:
                    raise TimeoutError("hardware control cycle exceeded max_cycle_time")
                if result is not None:
                    if result.status != last_status:
                        print(f"Hardware: {result.status} | {result.diagnostic}", flush=True)
                        last_status = result.status
                    if pickup is not None and pickup.finished:
                        if pickup.phase == "failed":
                            raise RuntimeError("pickup failed: " + result.diagnostic)
                        break
                    if result.status == "reached" and pickup is None:
                        break
                deadline = max(deadline + config.control["dt"], clock())
                sleep(max(0., deadline - clock()))
    finally:
        if cv2 is not None:
            cv2.destroyAllWindows()
    return result
