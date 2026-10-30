from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock, MagicMock

import numpy as np
import pytest

from camera_processing import TaggedCubeProcessor
from runtime_config import RuntimeConfig
import run_sequence


@pytest.mark.parametrize("override", [False, True])
def test_sequence_uses_shared_runtime_config_paths(monkeypatch, tmp_path, override):
    runtime = replace(RuntimeConfig.load(), hardware_config=tmp_path / "hardware.json",
                      controller_config=tmp_path / "controller.json",
                      estimator_config=tmp_path / "estimator.json", camera=False)
    load_runtime = Mock(return_value=runtime)
    monkeypatch.setattr(run_sequence.RuntimeConfig, "load", load_runtime)
    hardware, controller, estimator = Mock(), Mock(), Mock()
    for cls, loader in ((run_sequence.HardwareConfig, hardware),
                        (run_sequence.ControllerConfig, controller), (run_sequence.EKFConfig, estimator)):
        monkeypatch.setattr(cls, "load", loader)
    session = MagicMock()
    monkeypatch.setattr(run_sequence, "RobotSession", session)
    args = ["drop_zone_b", "--config", "custom-runtime.json", "--validate"]
    expected = runtime.hardware_config
    if override:
        expected = tmp_path / "override.json"
        args += ["--hardware-config", str(expected)]
    assert run_sequence.main(args) == 0
    load_runtime.assert_called_once_with("custom-runtime.json")
    hardware.assert_called_once_with(expected)
    controller.assert_called_once_with(runtime.controller_config)
    estimator.assert_called_once_with(runtime.estimator_config)
    assert session.call_args.kwargs["camera_view"] is False
    session.return_value.__enter__.assert_not_called()


def test_metric_rejection_diagnostics_report_face_ids_and_reset(monkeypatch):
    processor = TaggedCubeProcessor(marker_size=.04, cube_size=.064)
    corners = np.array([[[10., 10.], [30., 10.], [30., 30.], [10., 30.]]])
    processor.detector = SimpleNamespace(detectMarkers=lambda gray:
        ([corners, corners], np.array([[0], [2]]), []))
    processor._fallback_detector = SimpleNamespace(detectMarkers=lambda gray: ([], None, []))
    frame = SimpleNamespace(rgb=np.zeros((48, 64, 3), dtype=np.uint8), depth=np.ones((48, 64)),
        intrinsics=np.array([[50., 0, 32], [0, 50., 24], [0, 0, 1]]), world_from_camera=np.eye(4))
    edges = np.array([.05, .04, .05, .04])
    def reject(*args, **kwargs):
        processor.last_metric_edges = edges.copy()
        return processor._reject("tag_metric_size_inconsistent")
    monkeypatch.setattr(processor, "_extract", reject)
    assert processor.process(frame) is None
    assert [i for i, _ in processor.metric_size_rejections] == [0, 2]
    np.testing.assert_allclose(processor.metric_size_rejections[0][1], edges)
    processor.detector = SimpleNamespace(detectMarkers=lambda gray: ([], None, []))
    processor.process(frame)
    assert processor.metric_size_rejections == []
