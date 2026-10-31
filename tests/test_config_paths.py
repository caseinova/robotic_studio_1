from pathlib import Path

from common.runtime_config import RuntimeConfig, DEFAULT_RUNTIME_CONFIG_PATH
from hardware.hardware_config import HardwareConfig, DEFAULT_HARDWARE_CONFIG_PATH
from common.controller_config import ControllerConfig
from simulation.simulation_config import SimulationConfig
from simulation.noise_config import NoiseConfig
from common.estimator import EKFConfig


def test_defaults_load_from_config_outside_project_directory(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(DEFAULT_RUNTIME_CONFIG_PATH.resolve().parents[1]))
    monkeypatch.chdir(tmp_path)
    runtime = RuntimeConfig.load()
    directory = DEFAULT_RUNTIME_CONFIG_PATH.resolve().parent
    assert directory.name == 'config'
    for field in ('hardware_config', 'controller_config', 'simulation_config',
                  'noise_config', 'estimator_config'):
        path = getattr(runtime, field)
        assert path.parent == directory
        assert path.is_file()
    assert DEFAULT_HARDWARE_CONFIG_PATH.resolve() == runtime.hardware_config
    hardware = HardwareConfig.load()
    assert (Path(hardware.robot['api_path']) / 'robot/real_ur_backend.py').is_file()
    ControllerConfig.load()
    SimulationConfig.load()
    NoiseConfig.load()
    EKFConfig.load()
