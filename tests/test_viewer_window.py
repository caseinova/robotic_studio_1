from unittest.mock import Mock, patch

import pytest

from simulation.simulation_config import SimulationConfig
from simulation.viewer import launch_viewer, viewer_settings, window_rectangle


def test_defaults_are_centered_and_panels_hidden():
    settings = SimulationConfig().viewer
    assert not settings['show_left_ui'] and not settings['show_right_ui']
    assert window_rectangle(settings, (0, 30, 1920, 1000)) == (96, 80, 1728, 900)


def test_scene_starts_on_opposite_side_without_moving_perception_camera():
    import numpy as np
    from simulation.robot_model import RobotModel
    robot = RobotModel()
    assert robot.model.vis.global_.azimuth == pytest.approx(225)
    assert robot.model.vis.global_.elevation == pytest.approx(-18)
    np.testing.assert_allclose(robot.model.cam_pos[robot.camera_id], [.5852, .8864, .45])


def test_explicit_geometry_and_clamping():
    settings = viewer_settings(dict(width=1200, height=700, x=40, y=50))
    assert window_rectangle(settings, (0, 0, 1920, 1080)) == (40, 50, 1200, 700)
    assert window_rectangle(settings, (0, 0, 800, 600)) == (40, 50, 800, 600)


@pytest.mark.parametrize('settings', [None, {'unknown': 1}, {'width': False},
    {'width': 0}, {'show_left_ui': 1}, {'screen_fraction': float('nan')},
    {'screen_fraction': 1.1}, {'x': 2.5}])
def test_invalid_window_settings(settings):
    with pytest.raises(ValueError, match='viewer'):
        SimulationConfig(viewer=settings)


def test_launch_hides_panels_and_places_own_window():
    with patch('mujoco.viewer.launch_passive') as launch, patch('simulation.window_layout.place_window', return_value=True) as place:
        viewer = launch_viewer('model', 'data', {})
    launch.assert_called_once_with('model', 'data', show_left_ui=False, show_right_ui=False)
    assert viewer is launch.return_value
    viewer.cam.lookat.__setitem__.assert_called_once_with(slice(None), (0., .16, .23))
    assert viewer.cam.distance == 1.1
    assert viewer.cam.azimuth == 225.
    assert viewer.cam.elevation == -18.
    place.assert_called_once()


def test_geometry_failure_keeps_viewer_and_panels():
    with patch('mujoco.viewer.launch_passive') as launch, patch('simulation.window_layout.place_window', side_effect=RuntimeError('unsupported')):
        with pytest.warns(RuntimeWarning, match='geometry unchanged'):
            assert launch_viewer(None, None, {}) is launch.return_value
    launch.return_value.close.assert_not_called()


def test_no_geometry_calls_when_disabled():
    with patch('mujoco.viewer.launch_passive'), patch('simulation.window_layout.place_window') as place:
        launch_viewer(None, None, {'resize_window': False, 'show_left_ui': True})
    place.assert_not_called()


def test_interrupted_layout_closes_new_viewer():
    with patch('mujoco.viewer.launch_passive') as launch, patch('simulation.window_layout.place_window', side_effect=KeyboardInterrupt):
        with pytest.raises(KeyboardInterrupt):
            launch_viewer(None, None, {})
    launch.return_value.close.assert_called_once()
