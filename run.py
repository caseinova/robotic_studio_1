"""Run shared PBVS/MPC on a simulation or a configured real UR3 installation."""

import argparse
from pathlib import Path

from runtime_config import RuntimeConfig, DEFAULT_RUNTIME_CONFIG_PATH
from controller_config import ControllerConfig
from estimator import EKFConfig


def launch(settings, *, validate_only=False):
    controller_config = ControllerConfig.load(settings.controller_config)
    estimator_config = EKFConfig.load(settings.estimator_config)
    if settings.backend == "hardware":
        from hardware_config import HardwareConfig

        hardware = HardwareConfig.load(settings.hardware_config)
        hardware.validate_ready(controller_config, task=settings.task)
        if validate_only:
            print("Hardware configuration valid; no devices opened")
            return
        from hardware_runtime import run_hardware

        return run_hardware(hardware, controller_config, estimator_config, task=settings.task,
                            steps=settings.steps, camera_view=settings.camera and not settings.headless)
    from simulation_config import SimulationConfig
    from noise_config import NoiseConfig

    experiment = SimulationConfig.load(settings.simulation_config)
    noise = NoiseConfig.load(settings.noise_config)
    if validate_only:
        print("Simulation configuration valid; no simulation started")
        return
    from simulate import Simulation

    simulation = Simulation(config=experiment, controller_config=controller_config,
        estimator_config=estimator_config, noise_config=noise, pickup=settings.task == "pick")
    try:
        if not settings.headless:
            return simulation.launch_viewer(camera=settings.camera, servo=settings.task != "observe")
        if settings.task == "observe":
            return simulation.step(settings.steps)
        result = simulation.run_controlled(settings.steps)
        if result is not None:
            print(f"Simulation: {result.status} | {result.diagnostic}")
        return result
    finally:
        simulation.interface.close()
        simulation.plots.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_RUNTIME_CONFIG_PATH), help="runtime JSON configuration")
    parser.add_argument("--backend", choices=("simulation", "hardware"), default=None)
    parser.add_argument("--task", choices=("observe", "servo", "pick"), default=None)
    parser.add_argument("--hardware-config", type=Path, default=None)
    parser.add_argument("--headless", action="store_true", default=None)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--validate", action="store_true", help="validate configuration without opening any devices")
    args = parser.parse_args(argv)
    try:
        settings = RuntimeConfig.load(args.config).override(backend=args.backend, task=args.task,
            hardware_config=None if args.hardware_config is None else args.hardware_config.resolve(),
            headless=args.headless, steps=args.steps)
        launch(settings, validate_only=args.validate)
    except KeyboardInterrupt:
        print("Stopped")
        return 130
    except (OSError, ValueError, RuntimeError, TimeoutError, ImportError, KeyError, TypeError) as error:
        parser.exit(2, f"Error: {error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
