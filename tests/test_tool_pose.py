from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from common.controller_config import ControllerConfig
from common.estimator import EKFConfig
from common.tool_pose import ToolPoseTracker


def tracker(source='tag_with_fk_fallback', enabled=False):
    fk = np.eye(4)
    fk[0, 3] = .1
    grasp = fk.copy()
    grasp[2, 3] = .195
    k = SimpleNamespace(end_effector_marker_id=1, end_effector_marker_size=.04,
        tool_from_end_effector_marker=np.eye(4), tool_pose=lambda: fk,
        grasp_pose=lambda: grasp, synchronize=Mock())
    frame = SimpleNamespace(timestamp=10., world_from_camera=np.eye(4))
    interface = SimpleNamespace(timestamp=10., capture_rgbd=Mock(return_value=frame),
        read_state=Mock(return_value=SimpleNamespace(timestamp=10.)))
    t = ToolPoseTracker(interface, k, ControllerConfig(tool_pose_source=source),
                       EKFConfig(enabled=enabled), .05)
    tag = np.eye(4)
    tag[0, 3] = .2
    t.processor.process_marker = Mock(return_value=SimpleNamespace(camera_from_marker=tag))
    return t


def test_tag_and_grasp_use_selected_pose():
    t = tracker()
    assert t.refresh()[0, 3] == pytest.approx(.2)
    np.testing.assert_allclose(t.grasp_pose()[:3, 3], [.2, 0, .195])
    assert t.source == 'tag'
    assert t.interface.require_camera_for_motion
    t.processor.process_marker.return_value = None
    assert t.refresh()[0, 3] == pytest.approx(.1)
    assert t.source == 'fk_fallback'
    assert not t.interface.require_camera_for_motion


def test_fk_does_not_capture_or_filter():
    t = tracker('fk', enabled=True)
    assert t.refresh()[0, 3] == pytest.approx(.1)
    t.interface.capture_rgbd.assert_not_called()
    t.processor.process_marker.assert_not_called()
    assert t.estimator.timestamp is None


def test_tag_only_never_reuses_missing_pose():
    t = tracker('tag_only')
    assert t.refresh() is not None
    t.processor.process_marker.return_value = None
    assert t.refresh() is None
    assert t.source == 'unavailable'


@pytest.mark.parametrize('source', ['tag_only', 'tag_with_fk_fallback'])
def test_tool_ekf_acceptance_and_rejection(source):
    t = tracker(source, enabled=True)
    filtered = np.eye(4)
    filtered[0, 3] = .15
    t.estimator.update = Mock(return_value=SimpleNamespace(
        measurement_accepted=True, camera_from_cube=filtered))
    assert t.refresh()[0, 3] == pytest.approx(.15)
    t.estimator.update.return_value.measurement_accepted = False
    result = t.refresh()
    if source == 'tag_only':
        assert result is None
    else:
        assert result[0, 3] == pytest.approx(.1)
        assert t.source == 'fk_fallback'


def test_stale_visual_pose_holds_instead_of_reusing():
    t = tracker()
    t.refresh()
    t.interface.timestamp = 11.
    t.interface.read_state.return_value.timestamp = 11.
    assert t.refresh() is None
    assert t.diagnostic == 'stale_tool_pose_measurement'
