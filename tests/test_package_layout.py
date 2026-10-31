"""Guard package boundaries and launch-time resource resolution."""
import ast
import json
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('package,forbidden', [
    ('common', {'simulation', 'hardware', 'mujoco'}),
    ('hardware', {'simulation', 'mujoco'}),
    ('simulation', {'hardware'}),
])
def test_backend_dependency_direction(package, forbidden):
    for path in (ROOT / package).glob('*.py'):
        for node in ast.walk(ast.parse(path.read_text())):
            modules = []
            if isinstance(node, ast.Import):
                modules = [item.name for item in node.names]
            elif isinstance(node, ast.ImportFrom):
                modules = [node.module or '']
            assert not {module.split('.')[0] for module in modules} & forbidden, path


def test_only_entry_points_at_root():
    assert {path.name for path in ROOT.glob('*.py')} == {
        'run.py', 'run_sequence.py', 'run_experiments.py', 'calibrate_camera.py',
        'camera_tag_test.py', 'update_camera_intrinsics.py',
    }


@pytest.mark.parametrize('entry', [
    'run.py', 'run_sequence.py', 'run_experiments.py', 'calibrate_camera.py',
    'camera_tag_test.py', 'update_camera_intrinsics.py',
])
def test_entry_point_help_outside_project(entry, tmp_path):
    result = subprocess.run([sys.executable, str(ROOT / entry), '--help'],
                            cwd=tmp_path, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr


def test_simulator_module_help():
    result = subprocess.run([sys.executable, '-m', 'simulation.simulate', '--help'],
                            cwd=ROOT, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert '--pick' in result.stdout
    assert '--plots' in result.stdout


def test_sequence_preflight_outside_project(tmp_path):
    result = subprocess.run([sys.executable, str(ROOT / 'run_sequence.py'),
        'drop_zone_b', '--backend', 'simulation', '--validate'],
        cwd=tmp_path, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr


def test_device_factory_paths_resolve_without_device_access():
    from importlib import import_module
    config = json.loads((ROOT / 'config/hardware_config.json').read_text())
    for section in ('camera', 'gripper'):
        module, function = config[section]['factory'].split(':')
        assert module.startswith('hardware.')
        assert callable(getattr(import_module(module), function))
