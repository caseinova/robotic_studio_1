"""Run one pick/place sequence, or validate it without opening any devices."""

import argparse
import time
from pathlib import Path

from controller_config import ControllerConfig
from estimator import EKFConfig
from hardware_config import HardwareConfig
from robot_session import RobotSession
from sequence_config import SequenceConfig, DEFAULT_SEQUENCE_CONFIG_PATH
from runtime_config import RuntimeConfig, DEFAULT_RUNTIME_CONFIG_PATH


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination", help="name in config/sequence_config.json")
    parser.add_argument("--sequence-config", default=DEFAULT_SEQUENCE_CONFIG_PATH)
    parser.add_argument("--config", default=DEFAULT_RUNTIME_CONFIG_PATH, help="shared runtime config, as in run.py")
    parser.add_argument("--hardware-config", type=Path, default=None)
    parser.add_argument("--backend", choices=('hardware', 'simulation'), default='hardware')
    parser.add_argument("--validate", action="store_true", help="offline preflight only")
    parser.add_argument("--headless", action="store_true", default=None, help="disable the live camera window")
    args = parser.parse_args(argv)
    preview = None
    try:
        runtime = RuntimeConfig.load(args.config).override(backend=args.backend, task="pick",
            headless=args.headless,
            hardware_config=None if args.hardware_config is None else args.hardware_config.resolve())
        camera_view = runtime.camera and not runtime.headless
        if runtime.backend == 'simulation':
            from simulation_session import run_simulation_sequence
            return run_simulation_sequence(runtime, SequenceConfig.load(args.sequence_config),
                                           args.destination, validate_only=args.validate)
        robot = RobotSession(HardwareConfig.load(runtime.hardware_config),
                             ControllerConfig.load(runtime.controller_config),
                             EKFConfig.load(runtime.estimator_config),
                             SequenceConfig.load(args.sequence_config), camera_view=camera_view)
        print(f"Configs: hardware={runtime.hardware_config}; controller={runtime.controller_config}; "
              f"estimator={runtime.estimator_config}", flush=True)
        robot.validate_destination(args.destination)
        if args.validate:
            print("Sequence configuration is valid. No devices opened; physical geometry is not certified.")
            return 0
        if camera_view:
            from sequence_preview import SequencePreview
            preview = SequencePreview(robot.hardware.camera)
        with robot:
            def wait(action):
                previous = None
                last_report = 0.
                last_diagnostic = None
                last_warning = 0.
                while not action.done():
                    if preview is not None:
                        preview.update(robot)
                    if robot.status != previous:
                        print(f"Hardware: {robot.status} | {robot.diagnostic}", flush=True)
                        previous = robot.status
                    now = time.monotonic()
                    if (robot.diagnostic and robot.diagnostic != last_diagnostic
                            and now - last_warning >= 1.):
                        print(f"Hardware diagnostic: {robot.diagnostic}", flush=True)
                        last_diagnostic, last_warning = robot.diagnostic, now
                    elif not robot.diagnostic:
                        last_diagnostic = None
                    if robot.tracking_report and now - last_report >= 1.:
                        print(f"Tracking: {robot.tracking_report}", flush=True)
                        last_report = now
                    time.sleep(.05)
                print(action.result(), flush=True)

            wait(robot.start_servo_to_pick_up())
            wait(robot.start_move_to_destination_and_drop(args.destination))
        return 0
    except KeyboardInterrupt:
        print('Stopped')
        return 130
    except (ValueError, RuntimeError, TimeoutError) as error:
        print(f"Error: {error}")
        return 1
    finally:
        if preview is not None:
            preview.close()


if __name__ == "__main__":
    raise SystemExit(main())
