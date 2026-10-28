import json

import pytest

from estimator import EKFConfig


@pytest.mark.parametrize('cube', [False, True])
@pytest.mark.parametrize('tool', [False, True])
def test_independent_switches_load(tmp_path, cube, tool):
    path = tmp_path / 'ekf.json'
    path.write_text(json.dumps({'cube_enabled': cube, 'tool_enabled': tool}))
    config = EKFConfig.load(path)
    assert config.for_target('cube').enabled is cube
    assert config.for_target('tool').enabled is tool
    assert config.for_target('cube').position_measurement_std == config.position_measurement_std


@pytest.mark.parametrize('enabled', [False, True])
def test_legacy_switch_and_overrides(enabled):
    config = EKFConfig(enabled=enabled)
    assert config.for_target('cube').enabled is enabled
    assert config.for_target('tool').enabled is enabled
    config = EKFConfig(enabled=enabled, cube_enabled=not enabled)
    assert config.for_target('cube').enabled is not enabled
    assert config.for_target('tool').enabled is enabled


@pytest.mark.parametrize('name', ['cube_enabled', 'tool_enabled'])
@pytest.mark.parametrize('value', [1, 'true', [], .5])
def test_non_boolean_override_rejected(name, value):
    with pytest.raises(ValueError, match=name):
        EKFConfig(**{name: value})
