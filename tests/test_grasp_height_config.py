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
