import csv
from contextlib import nullcontext
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest

from simulation.estimation_telemetry import EstimationTelemetry
from simulation.simulate import Simulation


def test_true_estimate_covariance_and_missing_samples(tmp_path):
    sim = Simulation()
    data = sim.robot.data
    true_cube = data.body('cube').xpos.copy()
    estimate = np.eye(4)
    estimate[:3, 3] = true_cube + [.01, -.02, .03]
    covariance = np.eye(12) * .0004
    cube = SimpleNamespace(world_from_cube=estimate, covariance=covariance)
    telemetry = EstimationTelemetry()
    telemetry.record(data, cube_estimate=cube, cube_status='updated')
    truth, error, sigma, status = telemetry.samples[0]['cube']
    np.testing.assert_allclose(truth, true_cube)
    np.testing.assert_allclose(error[:3], [.01, -.02, .03])
    np.testing.assert_allclose(sigma, .02)
    assert status == 'updated'
    assert np.isnan(telemetry.samples[0]['tool'][1]).all()
    image, csv_path = telemetry.save(tmp_path)
    assert image.stat().st_size > 0
    assert (tmp_path / 'orientation_error.png').stat().st_size > 0
    with csv_path.open() as source:
        row = next(csv.DictReader(source))
    assert float(row['cube_error_x_m']) == np.float64(error[0])
    assert float(row['cube_estimate_x_m']) == np.float64(estimate[0, 3])
    assert float(row['cube_true_quaternion_w']) == 1.
    assert float(row['cube_estimate_quaternion_w']) == 1.
    assert row['tool_status'] == 'unavailable'
    assert np.isnan(float(row['tool_error_x_m']))


def test_mujoco_overlay_draws_both_and_clears_missing():
    sim = Simulation()
    model, data = sim.robot.model, sim.robot.data
    scene = mujoco.MjvScene(model, maxgeom=20)
    viewer = SimpleNamespace(user_scn=scene, lock=lambda: nullcontext())
    telemetry = EstimationTelemetry()
    pose = np.eye(4)
    cube = SimpleNamespace(world_from_cube=pose, covariance=np.eye(12)*.0001)
    telemetry.record(data, cube_estimate=cube, tool_pose=pose, tool_covariance=np.eye(6)*.0001)
    telemetry.draw(viewer)
    assert scene.ngeom == 12
    assert scene.geoms[0].type == mujoco.mjtGeom.mjGEOM_ELLIPSOID
    np.testing.assert_allclose(scene.geoms[0].size, [.02, .02, .02], atol=1e-6)
    telemetry.draw(viewer, uncertainty_scale=10.0)
    for index in (0, 6):
        np.testing.assert_allclose(scene.geoms[index].size, [.2, .2, .2], atol=1e-6)
        np.testing.assert_allclose(scene.geoms[index].pos, pose[:3, 3])
        assert '2 sigma x 10 (display only)' in scene.geoms[index + 1].label
    for kind in ('cube', 'tool'):
        np.testing.assert_allclose(telemetry.latest[kind][1], np.eye(6)*.0001)
        np.testing.assert_allclose(telemetry.samples[-1][kind][2], .01)
    telemetry.record(data)
    telemetry.draw(viewer)
    assert scene.ngeom == 0


def test_uncertainty_text_tracks_pose_scale_and_clears_without_covariance():
    sim = Simulation()
    viewer = SimpleNamespace(user_scn=mujoco.MjvScene(sim.robot.model, maxgeom=20),
                             lock=lambda: nullcontext())
    telemetry = EstimationTelemetry()
    pose = np.eye(4)
    for scale in (10.0, 2.5, 1.0):
        pose[:3, 3] += [.1, -.02, .03]
        telemetry.record(sim.robot.data, tool_pose=pose, tool_covariance=np.eye(6)*.0001)
        telemetry.draw(viewer, uncertainty_scale=scale)
        text = viewer.user_scn.geoms[1]
        assert text.type == mujoco.mjtGeom.mjGEOM_LABEL
        assert text.label == f'tool: 2 sigma x {scale:g} (display only)'
        np.testing.assert_allclose(text.pos, pose[:3, 3] + [0, 0, .02*scale + .012])
        assert not viewer.user_scn.geoms[0].label
    telemetry.record(sim.robot.data, tool_pose=np.eye(4), tool_status='fk')
    telemetry.draw(viewer)
    assert viewer.user_scn.ngeom == 4
    assert all(not viewer.user_scn.geoms[i].label for i in range(4))
    telemetry.record(sim.robot.data)
    telemetry.draw(viewer)
    assert viewer.user_scn.ngeom == 0


@pytest.mark.parametrize('scale', [0, -1, True, '10', None, float('nan'), float('inf')])
def test_overlay_rejects_invalid_scale(scale):
    with pytest.raises(ValueError, match='uncertainty_scale'):
        EstimationTelemetry().draw(None, uncertainty_scale=scale)


def test_stale_tool_estimate_is_not_reused():
    sim = Simulation()
    tracker = SimpleNamespace(pose=np.eye(4), source='fk_predicted',
        increment_estimator=SimpleNamespace(timestamp=sim.interface.timestamp - 1, covariance=np.eye(6)))
    sim.record_estimates(tracker=tracker)
    assert np.isnan(sim.estimation.samples[-1]['tool'][1]).all()


def test_actual_simulation_records_both_estimators():
    from common.controller_config import ControllerConfig
    from common.estimator import EKFConfig
    sim = Simulation(controller_config=ControllerConfig(mode='pure_pbvs',
        tool_pose_source='tag_with_fk_fallback'), estimator_config=EKFConfig.load())
    sim.run_controlled(100)
    assert any(np.isfinite(s['cube'][1]).all() for s in sim.estimation.samples)
    assert any(np.isfinite(s['tool'][1]).all() for s in sim.estimation.samples)
    sim.interface.close()


def test_unfiltered_pose_is_plotted_without_fabricated_covariance(tmp_path):
    sim = Simulation()
    telemetry = EstimationTelemetry()
    pose = np.eye(4)
    pose[:3, 3] = [.1, .2, .3]
    telemetry.record(sim.robot.data, cube_pose=pose, cube_status='raw_cube_tag',
                     tool_pose=pose, tool_status='fk')
    for kind in ('cube', 'tool'):
        assert np.isfinite(telemetry.samples[-1][kind][1]).all()
        assert np.isnan(telemetry.samples[-1][kind][2]).all()
        assert telemetry.latest[kind][1] is None
    image, csv_path = telemetry.save(tmp_path)
    assert image.exists()
    with csv_path.open() as source:
        row = next(csv.DictReader(source))
    assert row['cube_status'] == 'raw_cube_tag'
    assert row['tool_status'] == 'fk'
    assert np.isnan(float(row['tool_std_x_m']))
    viewer = SimpleNamespace(user_scn=mujoco.MjvScene(sim.robot.model, maxgeom=10), lock=lambda: nullcontext())
    telemetry.draw(viewer)
    assert viewer.user_scn.ngeom == 8
    assert all(viewer.user_scn.geoms[i].type != mujoco.mjtGeom.mjGEOM_ELLIPSOID for i in range(8))


def test_real_simulation_with_disabled_ekfs_logs_used_poses():
    from common.controller_config import ControllerConfig
    from common.estimator import EKFConfig
    sim = Simulation(controller_config=ControllerConfig(mode='pure_pbvs', tool_pose_source='fk'),
                     estimator_config=EKFConfig(enabled=False))
    sim.run_controlled(100)
    assert any(s['cube'][3] == 'raw_cube_tag' and np.isfinite(s['cube'][1]).all()
               and np.isnan(s['cube'][2]).all() for s in sim.estimation.samples)
    assert any(s['tool'][3] == 'fk' and np.isfinite(s['tool'][1]).all()
               and np.isnan(s['tool'][2]).all() for s in sim.estimation.samples)
    sim.interface.close()


def test_real_simulation_logs_raw_tool_tag_without_ekf():
    from common.controller_config import ControllerConfig
    from common.estimator import EKFConfig
    sim = Simulation(controller_config=ControllerConfig(mode='pure_pbvs',
        tool_pose_source='tag_with_fk_fallback'), estimator_config=EKFConfig(enabled=False))
    sim.run_controlled(100)
    assert any(s['tool'][3] == 'raw_tool_tag' and np.isfinite(s['tool'][1]).all()
               and np.isnan(s['tool'][2]).all() for s in sim.estimation.samples)
    sim.interface.close()


def test_ekf_rejection_plots_used_fk_not_rejected_prediction():
    sim = Simulation()
    pose = np.eye(4)
    pose[0, 3] = .2
    rejected = SimpleNamespace(timestamp=sim.interface.timestamp, measurement_accepted=False,
                               world_from_cube=np.eye(4), covariance=np.eye(12))
    controller = SimpleNamespace(latest_tool_estimate=rejected, tool_pose_source='fk_fallback',
        latest_tool_pose=pose, latest_tool_pose_timestamp=sim.interface.timestamp)
    sim.record_estimates(controller=controller)
    sample = sim.estimation.samples[-1]['tool']
    assert sample[3] == 'fk_fallback'
    np.testing.assert_allclose(sample[0] + sample[1][:3], pose[:3, 3])
    assert np.isnan(sample[2]).all()


def test_stale_control_result_tool_pose_is_not_reused():
    from common.estimator import EKFConfig
    sim = Simulation(estimator_config=EKFConfig(enabled=False))
    result = SimpleNamespace(pose_estimate=None, tool_pose=np.eye(4),
        tool_pose_timestamp=sim.interface.timestamp - 1, tool_pose_source='tag')
    sim.record_estimates(result=result)
    assert sim.estimation.samples[-1]['tool'][3] == 'unavailable'


def test_placement_transfer_records_raw_tag_pose_with_tool_ekf_off():
    from common.estimator import EKFConfig
    sim = Simulation(estimator_config=EKFConfig(enabled=False))
    pose = np.eye(4)
    pose[:3, 3] = [.2, .3, .4]
    action = SimpleNamespace(phase='transferring', latest_tool_pose=pose,
        latest_tool_pose_timestamp=sim.interface.timestamp, tool_pose_source='tag')
    sim.record_estimates(controller=action, action=action)
    truth, error, sigma, status = sim.estimation.samples[-1]['tool']
    assert status == 'raw_tool_tag'
    np.testing.assert_allclose(truth + error[:3], pose[:3, 3])
    assert np.isnan(sigma).all()


def test_pickup_records_latched_target_as_latched_not_live():
    from common.estimator import EKFConfig
    sim = Simulation(estimator_config=EKFConfig(enabled=False))
    latched = np.eye(4)
    latched[:3, 3] = [.1, .2, .3]
    pickup = SimpleNamespace(phase='descending', _world_from_cube=latched,
        latest_tool_pose=np.eye(4), latest_tool_pose_timestamp=sim.interface.timestamp)
    sim.record_estimates(action=pickup)
    sample = sim.estimation.samples[-1]
    assert sample['cube'][3] == 'latched_pick_target'
    assert sample['tool'][3] == 'fk'
    np.testing.assert_allclose(sample['cube'][0] + sample['cube'][1][:3], latched[:3, 3])
    assert np.isnan(sample['cube'][2]).all()
