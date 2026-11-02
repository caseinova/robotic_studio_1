import json

import pytest

from common.estimator import EKFConfig, TUNING_FIELDS


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


def test_all_tuning_fields_are_independent(tmp_path):
    cube = {name: .1 + i for i, name in enumerate(TUNING_FIELDS)}
    tool = {name: .2 + i for i, name in enumerate(TUNING_FIELDS)}
    path = tmp_path / 'ekf.json'
    path.write_text(json.dumps(dict(cube_enabled=True, tool_enabled=False, cube=cube, tool=tool)))
    config = EKFConfig.load(path)
    for target, expected in [('cube', cube), ('tool', tool)]:
        resolved = config.for_target(target)
        for name, value in expected.items():
            assert getattr(resolved, name) == value
        assert resolved.cube is None and resolved.tool is None
    assert config.for_target('cube').enabled
    assert not config.for_target('tool').enabled


def test_partial_tuning_inherits_legacy_defaults_and_copies_input():
    tuning = {'position_measurement_std': .02}
    config = EKFConfig(position_measurement_std=.01, cube=tuning)
    tuning['position_measurement_std'] = .9
    assert config.for_target('cube').position_measurement_std == .02
    assert config.for_target('tool').position_measurement_std == .01
    assert config.for_target('cube').innovation_gate == config.innovation_gate
    with pytest.raises(ValueError, match='target'):
        config.for_target('destination')


@pytest.mark.parametrize('target', ['cube', 'tool'])
@pytest.mark.parametrize('value', [[], False, {'enabled': True}, {'typo': 1},
    {'innovation_gate': 0}, {'max_prediction_age': float('nan')},
    {'position_measurement_std': True}, {'linear_acceleration_noise': '0.1'}])
def test_invalid_target_tuning_rejected(target, value):
    with pytest.raises(ValueError, match=target):
        EKFConfig(**{target: value})


def test_controller_uses_separate_tracker_covariances():
    from types import SimpleNamespace
    import numpy as np
    from common.controller import ApproachController
    from common.controller_config import ControllerConfig
    config = EKFConfig(cube={'position_measurement_std': .01},
                       tool={'position_measurement_std': .02})
    kin = SimpleNamespace(end_effector_marker_id=1, end_effector_marker_size=.04)
    controller = ApproachController(None, kin, estimator_config=config,
                                    controller_config=ControllerConfig(mode='pure_pbvs'))
    cube = controller.estimator.update(0., np.eye(4), np.eye(4))
    tool = controller.tool_estimator.update(0., np.eye(4), np.eye(4))
    assert cube.covariance[0, 0] == pytest.approx(.01**2)
    assert tool.covariance[0, 0] == pytest.approx(.02**2)
