import copy
import json
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

import calibrate_camera as calibration


def test_removed_reference_mode_is_rejected_before_device_access(monkeypatch, capsys):
    def unexpected(*args, **kwargs):
        raise AssertionError('No device should be opened for a removed command')
    monkeypatch.setattr(calibration, 'RealSenseD435i', unexpected)
    with pytest.raises(SystemExit) as error:
        calibration.main(['reference'])
    assert error.value.code == 2
    assert 'invalid choice' in capsys.readouterr().err


@pytest.mark.parametrize('mode', ['collect', 'solve'])
def test_tool_marker_commands_remain_available(mode, capsys):
    with pytest.raises(SystemExit) as error:
        calibration.main([mode, '--help'])
    assert error.value.code == 0
    assert '--samples' in capsys.readouterr().out




@pytest.mark.parametrize('stage,limits', [
    ('validation', dict(position_rms_mm=8, position_max_mm=11, rotation_rms_deg=4, rotation_max_deg=6)),
    ('fit', dict(position_rms_mm=8, position_max_mm=10, rotation_rms_deg=4, rotation_max_deg=10)),
])
def test_acceptance_boundaries(stage, limits):
    assert calibration.acceptable(limits, stage)
    for metric in limits:
        for value in (limits[metric] + .001, float('nan'), float('inf'), -1):
            assert not calibration.acceptable(dict(limits, **{metric: value}), stage)


def test_acceptance_uses_distinct_stage_limits():
    metrics = dict(position_rms_mm=7, position_max_mm=10.5, rotation_rms_deg=3, rotation_max_deg=5)
    assert calibration.acceptable(metrics, 'validation')
    assert not calibration.acceptable(metrics, 'fit')
    metrics.update(position_max_mm=9, rotation_max_deg=8)
    assert not calibration.acceptable(metrics, 'validation')
    assert calibration.acceptable(metrics, 'fit')
from unittest.mock import Mock


def transform(rotation, translation):
    result = np.eye(4)
    result[:3, :3] = Rotation.from_rotvec(rotation).as_matrix()
    result[:3, 3] = translation
    return result


def test_duplicate_capture_retries_without_reusing_frames():
    fresh = object()
    camera = SimpleNamespace(capture=Mock(side_effect=[
        calibration.RepeatedFrameError('duplicate'), fresh]))
    assert calibration.capture_fresh(camera) is fresh
    assert camera.capture.call_count == 2


def test_duplicate_capture_has_bounded_retries():
    camera = SimpleNamespace(capture=Mock(side_effect=calibration.RepeatedFrameError('duplicate')))
    with pytest.raises(RuntimeError, match='5 consecutive'):
        calibration.capture_fresh(camera)
    assert camera.capture.call_count == 5


def test_other_camera_failures_are_not_retried():
    camera = SimpleNamespace(capture=Mock(side_effect=RuntimeError('restarted')))
    with pytest.raises(RuntimeError, match='restarted'):
        calibration.capture_fresh(camera)
    assert camera.capture.call_count == 1


@pytest.fixture
def dataset():
    config = calibration.HardwareConfig.load(calibration.DEFAULT_HARDWARE_CONFIG_PATH)
    meta = calibration.metadata(config)
    model = calibration.workspace(meta)
    x = transform([.2, -.3, .1], [.02, .01, .08])
    z = transform([.4, -.2, .6], [.5, -.3, .7])
    rng = np.random.default_rng(42)
    samples = []
    for q in rng.uniform(-2, 2, (24, 6)):
        b = np.linalg.inv(z) @ model.forward_kinematics(q) @ x
        samples.append(dict(joint_positions=q.tolist(), camera_from_marker=b.tolist()))
    return dict(version=1, metadata=meta, samples=samples), x, z


def test_recovers_both_transform_directions(dataset):
    data, x, z = dataset
    result = calibration.solve(data)
    assert result['passed']
    np.testing.assert_allclose(result['calibration']['tool_from_marker'], x, atol=1e-10)
    np.testing.assert_allclose(result['calibration']['world_from_camera'], z, atol=1e-10)
    assert len(result['validation_indices']) == 6


def test_small_measurement_noise(dataset):
    data, x, z = dataset
    rng = np.random.default_rng(7)
    for sample in data['samples']:
        b = np.array(sample['camera_from_marker'])
        b = b @ transform(rng.normal(0, .001, 3), rng.normal(0, .0003, 3))
        sample['camera_from_marker'] = b.tolist()
    result = calibration.solve(data)
    assert result['passed']
    assert calibration.pose_distance(np.array(result['calibration']['world_from_camera']), z)[0] < .002


def test_nonidentity_base_and_tool(dataset):
    data, x, z = dataset
    data['metadata']['world_from_base'] = transform([.2, .1, -.3], [.1, .2, .3]).tolist()
    data['metadata']['flange_from_tool'] = transform([.1, -.2, .4], [0, 0, .1]).tolist()
    model = calibration.workspace(data['metadata'])
    for sample in data['samples']:
        sample['camera_from_marker'] = (np.linalg.inv(z) @ model.forward_kinematics(
            sample['joint_positions']) @ x).tolist()
    result = calibration.solve(data)
    assert result['passed']
    np.testing.assert_allclose(result['calibration']['world_from_camera'], z, atol=1e-10)


def test_offline_cli(dataset, tmp_path):
    data, _, _ = dataset
    samples, output = tmp_path / 'samples.json', tmp_path / 'result.json'
    samples.write_text(json.dumps(data))
    assert calibration.main(['solve', '--samples', str(samples), '--output', str(output)]) == 0
    assert json.loads(output.read_text())['passed']
    assert calibration.main(['solve', '--samples', str(samples), '--output', str(samples)]) == 1


def test_append_preserves_samples_and_checks_compatibility(dataset, tmp_path):
    data, _, _ = dataset
    data['robot_host'] = 'robot'
    path = tmp_path / 'samples.json'
    original = json.dumps(data)
    path.write_text(original)
    resumed = calibration.collection_data(path, data['metadata'], 'robot', append=True)
    assert resumed == data
    with pytest.raises(ValueError, match='use --append'):
        calibration.collection_data(path, data['metadata'], 'robot')
    with pytest.raises(ValueError, match='robot host'):
        calibration.collection_data(path, data['metadata'], 'other', append=True)
    changed = dict(data['metadata'], marker_size=.05)
    with pytest.raises(ValueError, match='settings changed'):
        calibration.collection_data(path, changed, 'robot', append=True)
    assert path.read_text() == original


def test_append_requires_valid_existing_file(dataset, tmp_path):
    data, _, _ = dataset
    path = tmp_path / 'samples.json'
    with pytest.raises(ValueError, match='does not exist'):
        calibration.collection_data(path, data['metadata'], 'robot', append=True)
    assert calibration.collection_data(path, data['metadata'], 'robot')['samples'] == []
    path.write_text('{"version": 1, "samples": null}')
    with pytest.raises(ValueError, match='invalid'):
        calibration.collection_data(path, data['metadata'], 'robot', append=True)


def test_bad_held_out_pose_fails(dataset):
    data, _, _ = dataset
    data['samples'][0]['camera_from_marker'][0][3] += .1
    result = calibration.solve(data)
    assert not result['passed']
    assert result['validation_errors']['position_max_mm'] > 90


def test_insufficient_and_degenerate_data(dataset):
    data, _, _ = dataset
    short = copy.deepcopy(data)
    short['samples'] = short['samples'][:11]
    with pytest.raises(ValueError, match='at least 12'):
        calibration.solve(short)
    data['samples'] = [data['samples'][0]] * 20
    with pytest.raises(ValueError, match='orientation diversity'):
        calibration.solve(data)


def test_write_requires_pass_and_unchanged_settings(dataset, tmp_path):
    data, _, _ = dataset
    path = tmp_path / 'hardware_config.json'
    values = json.loads(calibration.DEFAULT_HARDWARE_CONFIG_PATH.read_text())
    values['robot']['api_path'] = calibration.HardwareConfig.load().robot['api_path']
    original = json.dumps(values)
    path.write_text(original)
    result = calibration.solve(data)
    rejected = dict(result, passed=False)
    with pytest.raises(ValueError, match='failed residual'):
        calibration.apply_result(path, rejected)
    assert path.read_text() == original
    backup = calibration.apply_result(path, result)
    assert backup.read_text() == original
    updated = json.loads(path.read_text())
    old = json.loads(original)
    assert updated['robot'] == old['robot']
    assert updated['calibration']['tool_from_grasp'] == old['calibration']['tool_from_grasp']
    assert updated['calibration']['world_from_camera'] == result['calibration']['world_from_camera']
    old['perception']['tool_marker_size'] *= 2
    path.write_text(json.dumps(old))
    with pytest.raises(ValueError, match='settings changed'):
        calibration.apply_result(path, result)


@pytest.mark.parametrize('moving,stale,duplicate', [(False, False, False), (True, False, False),
                                                  (False, True, False), (False, False, True)])
@pytest.mark.parametrize('check_tag_stability', [True, False])
def test_capture_is_read_only_and_rejects_bad_samples(dataset, moving, stale, duplicate, check_tag_stability):
    data, _, _ = dataset
    sample = data['samples'][0]
    joint = SimpleNamespace(timestamp=0 if stale else 1., positions=np.array(sample['joint_positions']),
                            velocities=np.full(6, .1 if moving else 0.))
    arm = SimpleNamespace(read_joint_state=lambda: joint)
    camera = SimpleNamespace(capture=lambda: SimpleNamespace(timestamp=1.))
    processor = SimpleNamespace(process_marker=lambda frame: SimpleNamespace(
        camera_from_marker=np.array(sample['camera_from_marker'])))
    args = (arm, camera, lambda frame: frame, processor, calibration.workspace(data['metadata']),
            [sample] if duplicate else [])
    if moving or stale or duplicate:
        with pytest.raises(ValueError):
            calibration.capture_pose(*args, clock=lambda: 1., check_tag_stability=check_tag_stability)
    else:
        captured = calibration.capture_pose(*args, clock=lambda: 1., check_tag_stability=check_tag_stability)
        assert captured['frames_averaged'] == 10
        np.testing.assert_allclose(captured['joint_positions'], sample['joint_positions'])


@pytest.mark.parametrize('rotation', [False, True])
def test_stability_check_is_opt_out_and_records_scatter(dataset, rotation):
    data, _, _ = dataset
    sample = data['samples'][0]
    joint = SimpleNamespace(timestamp=1., positions=np.array(sample['joint_positions']), velocities=np.zeros(6))
    arm = SimpleNamespace(read_joint_state=lambda: joint)
    camera = SimpleNamespace(capture=lambda: SimpleNamespace(timestamp=1.))
    poses = []
    for sign in [-1, 1] * 5:
        delta = transform([sign * .12, 0, 0] if rotation else [0, 0, 0],
                          [0, 0, 0] if rotation else [sign * .01, 0, 0])
        poses.append(SimpleNamespace(camera_from_marker=np.array(sample['camera_from_marker']) @ delta))
    processor = SimpleNamespace(process_marker=Mock(side_effect=poses))
    args = (arm, camera, lambda frame: frame, processor, calibration.workspace(data['metadata']), [])
    with pytest.raises(ValueError, match='tool-tag pose is unstable'):
        calibration.capture_pose(*args, clock=lambda: 1.)
    processor.process_marker = Mock(side_effect=poses)
    captured = calibration.capture_pose(*args, clock=lambda: 1., check_tag_stability=False)
    assert captured['tag_stability_checked'] is False
    assert captured['frames_averaged'] == 10
    metric = 'tag_rotation_max_deviation_deg' if rotation else 'tag_position_max_deviation_mm'
    assert captured[metric] > (3 if rotation else 5)
