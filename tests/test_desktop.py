"""Offline desktop and child-process tests. Never open hardware devices."""
from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import pytest

from desktop.profiles import ROOT, arguments, parse_json, read_profile, snapshot, write_profile


def test_profile_roundtrip_and_snapshot_isolation(tmp_path):
    docs = read_profile(ROOT / 'config')
    original = deepcopy(docs)
    write_profile(tmp_path / 'profile', docs)
    assert read_profile(tmp_path / 'profile') == docs
    directory = snapshot(docs, 'simulation', 'pick_place', 'drop_zone_b', tmp_path)
    assert docs == original
    runtime = json.loads((directory / 'runtime_config.json').read_text())
    assert runtime['backend'] == 'simulation'
    assert runtime['task'] == 'pick'
    assert (directory / runtime['controller_config']).is_file()
    assert Path(read_profile(directory)['hardware']['robot']['api_path']).is_absolute()
    module, args = arguments(directory, validate=True)
    assert module == 'run_sequence'
    assert args[-2:] == ['--', 'drop_zone_b']
    assert '--validate' in args


def test_invalid_profile_and_snapshot(tmp_path):
    docs = read_profile(ROOT / 'config')
    for backend, task, dest in [('x', 'pick', ''), ('hardware', 'bad', ''), ('simulation', 'pick_place', 'missing')]:
        with pytest.raises(ValueError):
            snapshot(docs, backend, task, dest, tmp_path)
    with pytest.raises(ValueError):
        parse_json('{"x": NaN}')
    docs['controller']['tool_clearance'] = float('inf')
    with pytest.raises(ValueError):
        write_profile(tmp_path / 'bad', docs)
    assert not (tmp_path / 'bad').exists()


def test_external_runtime_references_and_relative_api(tmp_path):
    docs = read_profile(ROOT / 'config')
    write_profile(tmp_path, docs)
    runtime = json.loads((tmp_path / 'runtime_config.json').read_text())
    runtime['controller_config'] = 'custom.json'
    (tmp_path / 'custom.json').write_text('{"mode":"pure_pbvs"}')
    (tmp_path / 'runtime_config.json').write_text(json.dumps(runtime))
    assert read_profile(tmp_path)['controller'] == {'mode': 'pure_pbvs'}


def test_worker_validation_and_preexisting_stop(tmp_path):
    docs = read_profile(ROOT / 'config')
    directory = snapshot(docs, 'simulation', 'pick_place', 'drop_zone_b', tmp_path)
    command = [sys.executable, '-m', 'desktop.worker', str(directory), '--validate']
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr + result.stdout
    assert 'valid' in result.stdout
    (directory / 'stop.request').touch()
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=30)
    assert result.returncode == 130


def test_gui_main_closes_cleanly(tmp_path):
    pytest.importorskip('PySide6')
    code = '''
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication
from desktop.window import main, ExperimentWindow
app = QApplication([])
def close():
    next(w for w in app.topLevelWidgets() if isinstance(w, ExperimentWindow)).close()
QTimer.singleShot(100, close)
raise SystemExit(main())
'''
    result = subprocess.run([sys.executable, '-c', code], cwd=ROOT,
                            env=dict(os.environ, QT_QPA_PLATFORM='offscreen'),
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr


@pytest.fixture(scope='module')
def app():
    pytest.importorskip('PySide6')
    os.environ['QT_QPA_PLATFORM'] = 'offscreen'
    from PySide6.QtWidgets import QApplication
    application = QApplication.instance() or QApplication([])
    yield application
    from PySide6.QtCore import QCoreApplication, QEvent
    for widget in application.topLevelWidgets():
        widget.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    application.processEvents()


def wait_for(app, condition, timeout=30):
    deadline = time.monotonic() + timeout
    while not condition() and time.monotonic() < deadline:
        app.processEvents()
        time.sleep(.01)
    assert condition(), 'Qt process did not complete'


def test_editor_roundtrip_and_nested_json(app, monkeypatch):
    from desktop import window
    docs = read_profile(ROOT / 'config')
    tree = window.ConfigTree(docs, 'simulation')
    assert tree.documents() == docs
    hardware = next(tree.topLevelItem(i) for i in range(tree.topLevelItemCount())
                    if tree.topLevelItem(i).text(0) == 'Hardware installation')
    collision = next(hardware.child(i) for i in range(hardware.childCount()) if hardware.child(i).text(0) == 'collision')
    value = deepcopy(docs['hardware']['collision'])
    value['links'] = []
    monkeypatch.setattr(window, 'edit_json', lambda *args: (True, value))
    tree.change_json(collision)
    assert tree.documents()['hardware']['collision']['links'] == []
    assert tree.documents()['controller'] == docs['controller']
    tree.close()


def test_gui_validates_and_runs_without_blocking(app, tmp_path):
    from desktop.window import ExperimentWindow
    w = ExperimentWindow(tmp_path)
    docs = w.panel.tree.documents()
    docs['runtime'].update(headless=True, steps=1)
    w.panel.tree.set_documents(docs)
    w.panel.task.setCurrentIndex(0)  # Observe: no rendering or hardware access.
    w.prepare(True)
    assert w.process is not None
    assert not w.tabs.isEnabled()
    wait_for(app, lambda: w.process is None)
    assert w.status.text() == 'Completed', w.logs.toPlainText()
    assert w.tabs.isEnabled()
    assert (w.directory / 'console.log').is_file()
    w.close()


def test_invalid_config_never_runs(app, tmp_path):
    from desktop.window import ExperimentWindow
    w = ExperimentWindow(tmp_path)
    docs = w.panel.tree.documents()
    docs['controller']['approach_linear_speed_m_s'] = -1
    w.panel.tree.set_documents(docs)
    w.prepare(True)
    wait_for(app, lambda: w.process is None)
    assert w.status.text().startswith('Failed')
    assert 'must be' in w.logs.toPlainText()
    w.close()


def test_stop_is_cooperative_and_keeps_window_locked(app, tmp_path):
    from desktop.window import ExperimentWindow
    w = ExperimentWindow(tmp_path)
    docs = w.panel.tree.documents()
    docs['runtime'].update(headless=True, steps=100000000)
    w.panel.tree.set_documents(docs)
    w.panel.task.setCurrentIndex(0)
    w.prepare(True)
    wait_for(app, lambda: w.process is None or not w.validating)
    assert w.process is not None, w.logs.toPlainText()
    w.stop()
    assert (w.directory / 'stop.request').exists()
    assert not w.start_button.isEnabled()
    wait_for(app, lambda: w.process is None)
    assert 'after stop request' in w.status.text()
    w.close()


def test_hardware_confirmation_defaults_to_no(app, tmp_path, monkeypatch):
    from desktop import window
    from PySide6.QtCore import QProcess
    w = window.ExperimentWindow(tmp_path)
    w.tabs.setCurrentIndex(1)
    w.validating = True
    w.run_after_validation = True
    w.process = QProcess(w)
    calls = []
    def confirm(*args):
        calls.append(args)
        return window.QMessageBox.StandardButton.No
    monkeypatch.setattr(window.QMessageBox, 'warning', confirm)
    monkeypatch.setattr(w, 'launch', lambda **kwargs: pytest.fail('Hardware must not launch'))
    w.finished(0, QProcess.ExitStatus.NormalExit)
    assert calls[0][-1] == window.QMessageBox.StandardButton.No
    assert w.status.text() == 'Hardware run cancelled'
    w.close()
