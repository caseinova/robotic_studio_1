import copy
import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from hardware_config import DEFAULT_HARDWARE_CONFIG_PATH
from update_camera_intrinsics import select_serial, update_config


@pytest.fixture
def setup(tmp_path):
    path = tmp_path / 'config' / 'hardware_config.json'
    path.parent.mkdir()
    values = json.loads(DEFAULT_HARDWARE_CONFIG_PATH.read_text())
    values['camera']['options']['serial'] = 'old'
    path.write_text(json.dumps(values))
    device = Mock()
    device.get_info.side_effect = lambda key: {'name': 'RealSense D435I', 'serial': 'new'}[key]
    sdk = SimpleNamespace(camera_info=SimpleNamespace(name='name', serial_number='serial'),
                          context=lambda: SimpleNamespace(query_devices=lambda: [device]))
    calibration = dict(width=values['camera']['width'], height=values['camera']['height'],
                       intrinsics=[[600., 0., 320.], [0., 601., 240.], [0., 0., 1.]],
                       distortion=[.01, .02, 0., 0., 0.], depth_scale=.001)
    camera = Mock()
    camera.factory_calibration.return_value = dict(device=dict(serial='new'), camera=calibration)
    factory = Mock(return_value=camera)
    return path, values, sdk, camera, factory


def test_updates_only_calibration_fields_and_serial_with_backup(setup):
    path, values, sdk, camera, factory = setup
    original = path.read_text()
    report, backup, previous = update_config(path, sdk=sdk, camera_factory=factory)
    expected = copy.deepcopy(values)
    for key in ('intrinsics', 'distortion', 'depth_scale'):
        expected['camera'][key] = report['camera'][key]
    expected['camera']['options']['serial'] = 'new'
    assert json.loads(path.read_text()) == expected
    assert backup.read_text() == original
    assert backup.parent == path.parent.parent / 'old_version'
    assert previous == 'old'
    camera.close.assert_called_once()
    settings = factory.call_args.args[0]
    assert settings['options']['rgb_exposure'] is None
    assert settings['options']['serial'] == 'new'


def test_dry_run_never_writes(setup):
    path, _, sdk, camera, factory = setup
    original = path.read_text()
    _, backup, _ = update_config(path, dry_run=True, sdk=sdk, camera_factory=factory)
    assert backup is None and path.read_text() == original
    assert not (path.parent.parent / 'old_version').exists()
    camera.close.assert_called_once()


@pytest.mark.parametrize('failure', ['distortion', 'resolution', 'read'])
def test_failed_read_keeps_config_and_closes_camera(setup, failure):
    path, _, sdk, camera, factory = setup
    original = path.read_text()
    if failure == 'read':
        camera.factory_calibration.side_effect = RuntimeError('read failed')
    elif failure == 'resolution':
        camera.factory_calibration.return_value['camera']['width'] += 1
    else:
        camera.factory_calibration.return_value['camera']['distortion'] = None
    with pytest.raises((RuntimeError, ValueError)):
        update_config(path, sdk=sdk, camera_factory=factory)
    assert path.read_text() == original
    camera.close.assert_called_once()


def test_device_selection_requires_unambiguous_match():
    devices = [{'serial': 'a'}, {'serial': 'b'}]
    assert select_serial(devices, 'a') == 'a'
    assert select_serial(devices, 'a', 'b') == 'b'
    with pytest.raises(ValueError, match='Multiple'):
        select_serial(devices, 'old')
    with pytest.raises(ValueError, match='not connected'):
        select_serial(devices, 'a', 'missing')
    with pytest.raises(RuntimeError, match='No RealSense'):
        select_serial([], 'old')
