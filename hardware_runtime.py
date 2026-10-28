"""Build configured devices and run portable controllers on a monotonic clock."""

from contextlib import ExitStack
from dataclasses import replace
import os
from pathlib import Path
import time

import numpy as np

from controller import ApproachController, CubePickupController
from hardware_devices import URRTDE, OnRobotRG, custom_device
from hardware_rosbridge import RosbridgeArm
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
        gripper = None
        if config.gripper["driver"] != "disabled":
            gripper = managed(OnRobotRG(config.gripper, clock) if config.gripper["driver"] == "onrobot_rg"
                              else custom_device(config.gripper, clock), ("read_width", "command_width", "close"))
        arm_type = {"ur_rtde": URRTDE, "ur3_rg2_rosbridge": RosbridgeArm}.get(config.robot["driver"])
        robot = managed(arm_type(config.robot, config.control, clock) if arm_type is not None
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
        if cv2 is not None:
            fonts = Path("/usr/share/fonts/truetype/dejavu")
            if fonts.is_dir():
                os.environ.setdefault("QT_QPA_FONTDIR", str(fonts))
            # Initialize Qt and first paint before devices open or motion is possible.
            cv2.namedWindow("Hardware RGB-D", cv2.WINDOW_AUTOSIZE)
            cv2.imshow("Hardware RGB-D", np.zeros(
                (config.camera["height"], config.camera["width"], 3), dtype=np.uint8))
            key = cv2.waitKey(1) & 255
            if key in (27, ord('q')) or cv2.getWindowProperty("Hardware RGB-D", cv2.WND_PROP_VISIBLE) < 1:
                return None
        with interface_factory(config, clock=clock) as interface:
            if task != "observe":
                interface.prepare_motion()
            controller = (None if task == "observe" else ApproachController(interface, kinematics,
                config=settings, controller_config=controller_config, estimator_config=estimator_config,
                processor=processor))
            pickup = (CubePickupController(interface, kinematics, cube_size=processor.cube_size,
                                           control_dt=settings.dt, velocity_limit=settings.velocity_limit,
                                           clearance=settings.clearance,
                                           frame_max_age_intervals=controller_config.frame_max_age_intervals,
                                           grasp_height_offset_m=controller_config.grasp_height_offset_m)
                      if task == "pick" else None)
            started = deadline = clock()
            last_status = None
            last_stale_report = float('-inf')
            last_pickup_report = float('-inf')
            for _ in range(steps):
                if clock() - started >= config.control["max_run_seconds"]:
                    raise TimeoutError("hardware run exceeded max_run_seconds")
                cycle_start = clock()
                timings = {}
                if pickup is not None and pickup.phase != "idle":
                    interface.require_camera_for_motion = False
                    result = pickup.update()
                    timings["pickup"] = clock() - cycle_start
                    if cv2 is not None:
                        # Pickup uses a latched target; fresh images are display-only.
                        capture_start = clock()
                        frame = interface.capture_rgbd()
                        timings["camera"] = clock() - capture_start
                        display_start = clock()
                        cv2.imshow("Hardware RGB-D", cv2.cvtColor(frame.rgb, cv2.COLOR_RGB2BGR))
                        timings["display"] = clock() - display_start
                else:
                    interface.require_camera_for_motion = True
                    frame = interface.capture_rgbd()
                    timings["camera"] = clock() - cycle_start
                    processing_start = clock()
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
                    timings["observe" if controller is None else "controller"] = clock() - processing_start
                    frame_age_after_processing = clock() - frame.timestamp
                    display_start = clock()
                    if cv2 is not None:
                        rgb = processor.annotate(frame.rgb, observation, frame.intrinsics)
                        cv2.imshow("Hardware RGB-D", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
                    timings["display"] = clock() - display_start
                if cv2 is not None:
                    events_start = clock()
                    key = cv2.waitKey(1) & 255
                    timings["window_events"] = clock() - events_start
                    if key in (27, ord('q')) or cv2.getWindowProperty("Hardware RGB-D", cv2.WND_PROP_VISIBLE) < 1:
                        break
                elapsed = clock() - cycle_start
                if elapsed > config.control["max_cycle_time"]:
                    details = ", ".join(f"{name}={seconds * 1000:.1f} ms" for name, seconds in timings.items())
                    raise TimeoutError("hardware control cycle exceeded max_cycle_time: "
                        f"total={elapsed * 1000:.1f} ms, limit={config.control['max_cycle_time'] * 1000:.1f} ms; "
                        f"{details}; task={task}, cycle={_ + 1}")
                if result is not None:
                    if result.diagnostic == "stale_rgbd_frame" and clock() - last_stale_report >= 1.:
                        print(f"Hardware frame timing: age_after_processing={frame_age_after_processing * 1000:.1f} ms, "
                              f"limit={controller_config.frame_max_age_intervals * config.control['dt'] * 1000:.1f} ms, "
                              f"capture={timings.get('camera', 0) * 1000:.1f} ms, "
                              f"processing={timings.get('controller', 0) * 1000:.1f} ms", flush=True)
                        last_stale_report = clock()
                    if result.status != last_status:
                        print(f"Hardware: {result.status} | {result.diagnostic}", flush=True)
                        last_status = result.status
                        last_pickup_report = clock()
                    elif result.status in ("pickup_opening", "pickup_closing") and clock() - last_pickup_report >= 1.:
                        print(f"Hardware: {result.status} | {result.diagnostic}", flush=True)
                        last_pickup_report = clock()
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
