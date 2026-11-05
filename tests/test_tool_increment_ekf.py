from dataclasses import replace

import numpy as np
from scipy.spatial.transform import Rotation

from common.estimator import EKFConfig
from common.tool_increment_ekf import ToolIncrementEKF
from test_tool_pose import tracker


def pose(x=0):
    p = np.eye(4)
    p[0, 3] = x
    return p


def test_bias_is_preserved_without_absolute_fk_fallback():
    e = ToolIncrementEKF(EKFConfig())
    e.step(0., pose(.1), pose(.2), 0.)
    result = e.step(.05, pose(.102))
    assert np.isclose(result[0, 3], .202)
    assert e.last_status == 'fk_predicted'
    result = e.step(.1, pose(.104), pose(2.), .1)
    assert np.isclose(result[0, 3], .204)
    assert not e.measurement_accepted
    assert np.linalg.eigvalsh(e.covariance).min() >= 0


def test_body_rotation_and_translation_preserve_alignment():
    e = ToolIncrementEKF(EKFConfig())
    initial = pose()
    initial[:3, :3] = Rotation.from_euler('z', 90, degrees=True).as_matrix()
    e.step(0., pose(), initial, 0.)
    delta = pose(.01)
    delta[:3, :3] = Rotation.from_euler('x', 15, degrees=True).as_matrix()
    np.testing.assert_allclose(e.step(.05, delta), initial @ delta, atol=1e-12)


def test_expiry_requires_explicit_reset_even_with_new_tag():
    e = ToolIncrementEKF(EKFConfig())
    np.testing.assert_allclose(e.step(0., pose()), pose())
    e.step(.1, pose(), pose(.01), .1)
    assert e.step(.5, pose(), pose(.02), .5) is None
    assert e.step(.6, pose(), pose(.02), .6) is None
    assert e.last_status == 'expired_reset_required'
    e.reset()
    assert e.step(.6, pose(), pose(.02), .6) is not None


def test_duplicate_measurement_does_not_correct_again():
    e = ToolIncrementEKF(EKFConfig())
    e.step(0., pose(), pose(.1), 0.)
    result = e.step(.05, pose(.01), pose(.1), 0.)
    assert np.isclose(result[0, 3], .11)
    assert e.measurement_timestamp == 0.


def test_uncertainty_limit_holds():
    e = ToolIncrementEKF(EKFConfig(fk_position_noise=.5, max_position_std=.02))
    e.step(0., pose(), pose(), 0.)
    assert e.step(.05, pose()) is None
    assert e.last_status == 'uncertain_reset_required'


def test_consistent_reacquisition_corrects_without_reset():
    e = ToolIncrementEKF(EKFConfig())
    e.step(0., pose(), pose(.1), 0.)
    e.step(.05, pose(.01))
    result = e.step(.1, pose(.02), pose(.121), .1)
    assert .12 < result[0, 3] < .121
    assert e.measurement_accepted
    assert e.last_status == 'updated'


def test_tracker_integrates_prediction_without_raw_fk_switch():
    t = tracker(enabled=True)
    c = replace(t.estimator.config, prediction_model='fk_increment')
    t.estimator.config = c
    t.increment_estimator = ToolIncrementEKF(c)
    assert np.isclose(t.refresh()[0, 3], .2)
    t.processor.process_marker.return_value = None
    t.interface.timestamp += .05
    t.interface.read_state.return_value.timestamp += .05
    t.interface.capture_rgbd.return_value.timestamp += .05
    assert np.isclose(t.refresh()[0, 3], .2)
    assert t.source == 'fk_predicted'
    assert t.interface.require_camera_for_motion


def test_cube_cannot_enable_increment_prediction():
    c = EKFConfig(prediction_model='fk_increment', tool={'prediction_model': 'fk_increment'})
    assert c.for_target('cube').prediction_model == 'constant_velocity'
    assert c.for_target('tool').prediction_model == 'fk_increment'


def test_fk_startup_propagates_and_does_not_claim_visual_measurement():
    e = ToolIncrementEKF(EKFConfig())
    np.testing.assert_allclose(e.step(0., pose(.1)), pose(.1))
    assert e.last_status == 'fk_initialized'
    assert not e.measurement_accepted
    assert e.measurement_timestamp is None
    assert e.last_camera_timestamp is None
    np.testing.assert_allclose(e.step(.05, pose(.102)), pose(.102))
    corrected = e.step(.1, pose(.104), pose(.105), .1)
    assert .104 < corrected[0, 3] < .105
    assert e.measurement_accepted


def test_fk_startup_still_expires_without_tag():
    e = ToolIncrementEKF(EKFConfig())
    e.step(0., pose(.1))
    assert e.step(.31, pose(.11)) is None
    assert e.last_status == 'expired_reset_required'


def test_first_tag_outlier_does_not_reset_fk_seed():
    e = ToolIncrementEKF(EKFConfig())
    e.step(0., pose(.1))
    result = e.step(.05, pose(.102), pose(2.), .05)
    assert np.isclose(result[0, 3], .102)
    assert not e.measurement_accepted


def test_visual_correction_is_bounded():
    c = EKFConfig(position_measurement_std=.08, innovation_gate=25.,
                  max_position_correction=.003)
    e = ToolIncrementEKF(c)
    e.step(0., pose())
    result = e.step(.05, pose(), pose(.1), .05)
    assert e.measurement_accepted
    assert 0 < result[0, 3] <= .003 + 1e-12


def test_skewed_tag_is_not_fused():
    t = tracker(enabled=True)
    c = replace(t.estimator.config, prediction_model='fk_increment', max_measurement_skew=.05)
    t.estimator.config = c
    t.increment_estimator = ToolIncrementEKF(c)
    frame = t.interface.capture_rgbd.return_value
    frame.timestamp = 9.9
    t.interface.timestamp = 10.
    result = t.select(frame, t.processor.process_marker(frame), 10.)
    assert np.isclose(result[0, 3], .1)
    assert t.source == 'fk_unverified'
    assert 'camera_state_skew' in t.diagnostic


def test_tracker_fk_initialization_respects_tag_only():
    for source in ('tag_with_fk_fallback', 'tag_only'):
        t = tracker(source, enabled=True)
        c = replace(t.estimator.config, prediction_model='fk_increment')
        t.estimator.config = c
        t.increment_estimator = ToolIncrementEKF(c)
        t.processor.process_marker.return_value = None
        result = t.refresh()
        if source == 'tag_only':
            assert result is None
        else:
            assert np.isclose(result[0, 3], .1)
            assert t.source == 'fk_unverified'
