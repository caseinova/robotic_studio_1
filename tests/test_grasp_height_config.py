import json
from unittest.mock import Mock

import pytest

from controller_config import ControllerConfig
from controller import CubePickupController


@pytest.mark.parametrize('value', [True, None, '0.005', float('nan'), float('inf')])
def test_invalid_height_rejected(value):
    with pytest.raises(ValueError, match='grasp_height_offset_m'):
        ControllerConfig(grasp_height_offset_m=value)
    with pytest.raises(ValueError, match='grasp_height_offset_m'):
        CubePickupController(Mock(), Mock(), grasp_height_offset_m=value)


def test_config_loads_signed_height_and_defaults_to_zero(tmp_path):
    path = tmp_path / 'controller.json'
    path.write_text('{}')
    assert ControllerConfig.load(path).grasp_height_offset_m == 0.
    for value in (-.005, .005):
        path.write_text(json.dumps({'grasp_height_offset_m': value}))
        assert ControllerConfig.load(path).grasp_height_offset_m == value


@pytest.mark.parametrize('field,value', [('pickup_dwell_seconds', -1), ('pickup_dwell_seconds', True),
    ('pickup_dwell_seconds', float('inf')), ('pickup_dwell_radius_m', 0),
    ('pickup_dwell_radius_m', float('nan')), ('pickup_dwell_radius_m', '0.01')])
def test_invalid_dwell_settings_rejected(field, value):
    with pytest.raises(ValueError, match=field):
        ControllerConfig(**{field: value})
    with pytest.raises(ValueError, match=field):
        CubePickupController(Mock(), Mock(), **{field: value})


def test_dwell_json_settings(tmp_path):
    path = tmp_path / 'controller.json'
    path.write_text(json.dumps({'pickup_dwell_seconds': 3., 'pickup_dwell_radius_m': .015}))
    config = ControllerConfig.load(path)
    assert config.pickup_dwell_seconds == 3.
    assert config.pickup_dwell_radius_m == .015
