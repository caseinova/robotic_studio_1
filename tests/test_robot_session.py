from contextlib import contextmanager
from types import SimpleNamespace
import time

import numpy as np
import pytest

from common.controller_config import ControllerConfig
from common.estimator import EKFConfig
from hardware_fakes import configured
from hardware.robot_session import RobotSession
from common.pickup import PickAction
from common.sequence_config import Destination, SequenceConfig


def destination(**changes):
    values = dict(marker_id=10, marker_size_m=.08, marker_from_cube=np.eye(4).tolist(),
                  collision_support_plane="table")
    values.update(changes)
    return Destination(**values)


def test_pick_tracking_diagnostics_keep_timing_and_source_changes():
    from common.controller import ControlResult
    from dataclasses import replace
    interface = SimpleNamespace(timestamp=10.)
    def capture():
        interface.timestamp += .01
        return SimpleNamespace(timestamp=interface.timestamp - .02)
    interface.capture_rgbd = capture
    result = ControlResult(None, np.zeros(6), np.zeros(6), "tracking", .01, .02,
                           np.nan, tool_pose_source="tag")
    def update(frame):
        interface.timestamp += .06
        return result
    s = SimpleNamespace(interface=interface, publish_preview=lambda frame: None,
        controller_config=SimpleNamespace(frame_max_age_intervals=4),
        settings=SimpleNamespace(dt=.05))
    action = PickAction.__new__(PickAction)
    action.session = s
    action.pickup = SimpleNamespace(phase="idle")
    action.approach = SimpleNamespace(update=update)
    action._last_update = action._last_source = None
    action._source_switches = 0
    assert not action.update()
    assert "capture_ms=10.0" in s.tracking_report
    assert "processing_ms=60.0" in s.tracking_report
    assert "frame_age_ms=80.0" in s.tracking_report
    result = replace(result, tool_pose_source="fk_fallback")
    action.update()
    assert "source_switches=1" in s.tracking_report
    assert "interval_ms=70.0" in s.tracking_report
    result = replace(result, status="target_lost_or_stale", diagnostic="stale_rgbd_frame",
                     tool_pose_source="")
    action.update()
    assert s.diagnostic == "stale_rgbd_frame"
    assert "source_switches=1" in s.tracking_report


@pytest.mark.parametrize("changes", [dict(marker_id=True), dict(marker_id=50),
    dict(marker_size_m=0), dict(approach_height_m=-1), dict(retreat_height_m=float('nan')),
    dict(marker_from_cube=np.zeros((4, 4)).tolist()), dict(collision_support_plane="")])
def test_invalid_destination(changes):
    with pytest.raises(ValueError):
        destination(**changes)


def test_sequence_config_and_preflight():
    assert SequenceConfig().destinations == {}
    with pytest.raises(ValueError, match="unique"):
        SequenceConfig(destinations={"a": destination(), "b": destination()})
    config = SequenceConfig(destinations={"b": destination()})
    hardware = configured()
    hardware.collision["links"] = [link for link in hardware.collision["links"]
                                   if link["frame"] != "tool"]
    hardware.collision["obstacles"] = []
    with pytest.raises(ValueError, match="unknown destination"):
        config.validate_destination("missing", hardware, hardware.build_kinematics())
    with pytest.raises(ValueError, match="tool-body"):
        config.validate_destination("b", hardware, hardware.build_kinematics())
    conflict = SequenceConfig(destinations={"b": destination(marker_id=1)})
    with pytest.raises(ValueError, match="overlaps"):
        conflict.validate_destination("b", hardware, hardware.build_kinematics())


def make_session(monkeypatch, **settings):
    calls = []
    interface = SimpleNamespace(
        prepare_motion=lambda: calls.append("prepare"),
        read_state=lambda: SimpleNamespace(gripper_width=.064),
        set_joint_velocities=lambda q: calls.append("stop"))

    @contextmanager
    def factory(config, clock):
        calls.append("open")
        try:
            yield interface
        finally:
            calls.append("close")

    robot = RobotSession(configured(), ControllerConfig(mode="pure_pbvs"), EKFConfig(),
        SequenceConfig(**settings), interface_factory=factory)
    monkeypatch.setattr(robot, "validate_destination", lambda name: destination())
    return robot, calls


def test_one_session_pick_then_place_and_idle_heartbeat(monkeypatch):
    robot, calls = make_session(monkeypatch)

    class Pick:
        def __init__(self, s):
            self.s = s

        def update(self):
            calls.append("pick")
            self.s.tool_from_cube = np.eye(4)
            return True

    class Place(Pick):
        def __init__(self, s, target):
            super().__init__(s)

        def update(self):
            assert self.s.tool_from_cube is not None
            calls.append("place")
            self.s.tool_from_cube = None
            return True

    monkeypatch.setattr("hardware.robot_session.PickAction", Pick)
    monkeypatch.setattr("common.placement.PlaceAction", Place)
    with robot:
        assert robot.servo_to_pick_up().holding_cube
        before = calls.count("stop")
        time.sleep(.12)
        assert calls.count("stop") > before
        assert not robot.move_to_destination_and_drop("b").holding_cube
    assert calls.count("open") == calls.count("close") == 1
    assert calls.index("pick") < calls.index("place") < calls.index("close")


def test_failure_stops_chain_and_closes_without_release(monkeypatch):
    robot, calls = make_session(monkeypatch)

    class Fails:
        def __init__(self, s):
            pass

        def update(self):
            raise RuntimeError("path blocked")

    monkeypatch.setattr("hardware.robot_session.PickAction", Fails)
    with pytest.raises(RuntimeError, match="path blocked"):
        with robot:
            robot.servo_to_pick_up()
            calls.append("should not run")
    assert "should not run" not in calls
    assert calls[-1] == "close"


def test_concurrent_actions_rejected_and_cancel_propagates(monkeypatch):
    robot, calls = make_session(monkeypatch)
    class Waiting:
        def __init__(self, s):
            pass
        def update(self):
            return False
    monkeypatch.setattr("hardware.robot_session.PickAction", Waiting)
    with robot:
        future = robot.start_servo_to_pick_up()
        with pytest.raises(RuntimeError, match="already active"):
            robot.start_servo_to_pick_up()
    with pytest.raises(RuntimeError, match="cancelled"):
        future.result()


def test_idle_expiry(monkeypatch):
    robot, calls = make_session(monkeypatch, idle_timeout_s=.03)
    robot.__enter__()
    robot._thread.join(timeout=2)
    with pytest.raises(RuntimeError, match="idle_timeout"):
        robot.close()
    assert calls[-1] == "close"


def test_place_without_pick_is_rejected(monkeypatch):
    robot, calls = make_session(monkeypatch)
    with pytest.raises(RuntimeError, match="successful pickup"):
        with robot:
            robot.move_to_destination_and_drop("b")


def test_preview_keeps_only_latest_frame(monkeypatch):
    robot, _ = make_session(monkeypatch)
    first, second = object(), object()
    robot.publish_preview(first)
    assert robot.latest_preview() is None
    robot.camera_view = True
    robot.publish_preview(first)
    robot.publish_preview(second)
    assert robot.latest_preview() is second


def test_exit_does_not_duplicate_same_action_failure(monkeypatch):
    robot, _ = make_session(monkeypatch)
    error = ValueError("destination gap")
    robot.error = error
    assert robot.__exit__(type(error), error, None) is False


def test_preview_renders_rgb_and_window_cancel(monkeypatch):
    import sys
    from unittest.mock import Mock
    from common.sequence_preview import SequencePreview
    cv = Mock()
    cv.waitKey.return_value = -1
    cv.getWindowProperty.return_value = 1
    cv.cvtColor.side_effect = lambda rgb, code: rgb.copy()
    monkeypatch.setitem(sys.modules, "cv2", cv)
    view = SequencePreview({"width": 64, "height": 48})
    frame = SimpleNamespace(rgb=np.zeros((48, 64, 3), dtype=np.uint8))
    robot = SimpleNamespace(latest_preview=lambda: frame, status="pickup_lifting")
    view.update(robot)
    assert cv.imshow.call_count == 2
    view.update(robot)
    assert cv.imshow.call_count == 2
    assert cv.cvtColor.call_count == 1
    robot.status = "pickup_descending"
    view.update(robot)
    assert cv.imshow.call_count == 3
    assert cv.cvtColor.call_count == 1
    frame = SimpleNamespace(rgb=np.zeros((48, 64, 3), dtype=np.uint8))
    view.update(robot)
    assert cv.cvtColor.call_count == 2
    assert cv.aruco.mock_calls == []
    np.testing.assert_array_equal(frame.rgb, 0)
    cv.waitKey.return_value = ord('q')
    with pytest.raises(RuntimeError, match="cancelled"):
        view.update(robot)
    view.close()
    cv.destroyWindow.assert_called_once()
