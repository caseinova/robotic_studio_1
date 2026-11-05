"""Qt configuration editor and nonblocking experiment process manager."""
from copy import deepcopy
import codecs
import json
import math
import re
from pathlib import Path
import sys

from PySide6.QtCore import QCoreApplication, QEvent, QProcess, QProcessEnvironment, Qt, QTimer, Signal
from PySide6.QtGui import QFont, QFontDatabase, QTextCursor, QValidator
from PySide6.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QDialog, QDialogButtonBox,
    QDoubleSpinBox, QFileDialog, QHBoxLayout, QLabel, QLineEdit, QMainWindow,
    QMessageBox, QPlainTextEdit, QPushButton, QSpinBox, QSplitter, QStyle,
    QTabWidget, QTreeWidget, QTreeWidgetItem, QVBoxLayout, QWidget,
)

from desktop.profiles import FILES, ROOT, parse_json, read_profile, snapshot, write_profile

TITLES = {
    'runtime': 'Run settings', 'controller': 'Approach & pickup',
    'sequence': 'Placement & destinations', 'estimator': 'Pose estimation',
    'simulation': 'Simulation scene', 'noise': 'Simulation noise',
    'hardware': 'Hardware installation',
}
ENUMS = {'mode': ('pure_pbvs', 'pbvs_mpc'),
         'tool_pose_source': ('tag_only', 'tag_with_fk_fallback', 'fk'),
         'prediction_model': ('constant_velocity', 'fk_increment')}


class RealInput(QDoubleSpinBox):
    def textFromValue(self, value):
        return format(value, '.12g')

    def validate(self, text, pos):
        text = text.strip()
        if re.fullmatch(r'[+-]?(?:(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d*)?|\.)?', text):
            try:
                value = float(text)
            except ValueError:
                return QValidator.State.Intermediate, text, pos
            if math.isfinite(value) and self.minimum() <= value <= self.maximum():
                return QValidator.State.Acceptable, text, pos
            return QValidator.State.Intermediate, text, pos
        return QValidator.State.Invalid, text, pos

    def valueFromText(self, text):
        try:
            return float(text.strip())
        except ValueError:
            return self.value()


def committed_numeric_value(widget, original, initial, initial_text):
    # Include text still being edited when a profile is saved or a run starts.
    if widget.value() == initial and widget.text() == initial_text:
        return original
    widget.interpretText()
    value = widget.value()
    return original if value == initial else value


def button(parent, text, icon, callback, tooltip=None):
    widget = QPushButton(parent.style().standardIcon(icon), text)
    widget.setToolTip(tooltip or text)
    widget.clicked.connect(callback)
    return widget


def edit_json(parent, title, value):
    dialog = QDialog(parent)
    dialog.setWindowTitle(title)
    dialog.resize(620, 480)
    layout = QVBoxLayout(dialog)
    editor = QPlainTextEdit(json.dumps(value, indent=2))
    editor.setFont(QFontDatabase.systemFont(QFontDatabase.SystemFont.FixedFont))
    layout.addWidget(editor)
    error = QLabel()
    error.setWordWrap(True)
    error.setStyleSheet('color: #b42318')
    layout.addWidget(error)
    controls = QDialogButtonBox(QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel)
    layout.addWidget(controls)
    result = []

    def accept():
        try:
            result.append(parse_json(editor.toPlainText()))
        except ValueError as exc:
            error.setText(str(exc))
            return
        dialog.accept()

    controls.accepted.connect(accept)
    controls.rejected.connect(dialog.reject)
    return (True, result[0]) if dialog.exec() == QDialog.DialogCode.Accepted else (False, value)


class ConfigTree(QTreeWidget):
    documentsChanged = Signal()

    def __init__(self, documents, backend):
        super().__init__()
        self.backend = backend
        self.setColumnCount(2)
        self.setHeaderLabels(['Setting', 'Value'])
        self.setColumnWidth(0, 350)
        self.setAlternatingRowColors(True)
        self.setUniformRowHeights(False)
        self.getters = {}
        self.set_documents(documents)

    def set_documents(self, documents):
        self.getters = {}
        self.clear()
        self.original = deepcopy(documents)
        order = ['runtime', 'controller', 'sequence', 'estimator']
        order += ['simulation', 'noise', 'hardware'] if self.backend == 'simulation' else ['hardware', 'simulation', 'noise']
        for name in order:
            item = self.add_value(self.invisibleRootItem(), name, documents[name], top=True)
            item.setText(0, TITLES[name])
            item.setExpanded(name in ('runtime', 'controller'))

    def add_value(self, parent, name, value, *, top=False):
        item = QTreeWidgetItem(parent, [name.replace('_', ' '), ''])
        item.setToolTip(0, name)
        if isinstance(value, dict):
            self.getters[id(item)] = lambda i=item: {
                i.child(n).data(0, Qt.ItemDataRole.UserRole): self.getters[id(i.child(n))]()
                for n in range(i.childCount())}
            row = QWidget()
            layout = QHBoxLayout(row)
            layout.setContentsMargins(0, 0, 4, 0)
            layout.addStretch()
            if top:
                load = button(self, '', QStyle.StandardPixmap.SP_DialogOpenButton,
                              lambda checked=False, key=name: self.load_document(key), 'Load configuration file')
                layout.addWidget(load)
            edit = button(self, '', QStyle.StandardPixmap.SP_FileDialogDetailedView,
                          lambda checked=False, i=item: self.change_json(i), 'Edit category as JSON')
            layout.addWidget(edit)
            self.setItemWidget(item, 1, row)
            for key, child in value.items():
                # These selectors are provided by the backend tabs and profile loader.
                if top and name == 'runtime' and (key in ('backend', 'task') or key.endswith('_config')):
                    continue
                self.add_value(item, key, child)
        elif isinstance(value, list) or value is None:
            self.getters[id(item)] = lambda v=deepcopy(value): v
            widget = button(self, self.summary(value), QStyle.StandardPixmap.SP_FileDialogDetailedView,
                            lambda checked=False, i=item: self.change_json(i), 'Edit JSON value, array, or matrix')
            self.setItemWidget(item, 1, widget)
        elif isinstance(value, bool):
            widget = QCheckBox()
            widget.setChecked(value)
            self.getters[id(item)] = widget.isChecked
            self.setItemWidget(item, 1, widget)
        elif isinstance(value, (int, float)):
            if isinstance(value, int) and -(2**31) < value < 2**31:
                widget = QSpinBox()
                widget.setRange(-(2**31) + 1, 2**31 - 1)
            else:
                widget = RealInput()
                widget.setDecimals(16)
                widget.setRange(-1e100, 1e100)
                widget.setSingleStep(.01)
            widget.setValue(value)
            widget.setKeyboardTracking(False)
            widget.setButtonSymbols(QDoubleSpinBox.ButtonSymbols.PlusMinus)
            initial = widget.value()
            initial_text = widget.text()
            self.getters[id(item)] = lambda w=widget, original=value, initial=initial, text=initial_text: (
                committed_numeric_value(w, original, initial, text))
            self.setItemWidget(item, 1, widget)
        else:
            if name in ENUMS and value in ENUMS[name]:
                widget = QComboBox()
                widget.addItems(ENUMS[name])
                widget.setCurrentText(value)
                getter = widget.currentText
            else:
                widget = QLineEdit(str(value))
                getter = widget.text
            self.getters[id(item)] = getter
            self.setItemWidget(item, 1, widget)
        item.setData(0, Qt.ItemDataRole.UserRole, name)
        return item

    @staticmethod
    def summary(value):
        if value is None:
            return 'null'
        return f'{len(value)} entries'

    def documents(self):
        values = {self.topLevelItem(n).data(0, Qt.ItemDataRole.UserRole):
                  self.getters[id(self.topLevelItem(n))]() for n in range(self.topLevelItemCount())}
        runtime = deepcopy(self.original['runtime'])
        runtime.update(values['runtime'])
        values['runtime'] = runtime
        return values

    def change_json(self, item):
        key = item.data(0, Qt.ItemDataRole.UserRole)
        current = self.getters[id(item)]()
        accepted, value = edit_json(self, key, current)
        if not accepted:
            return
        if isinstance(current, dict) and not isinstance(value, dict):
            QMessageBox.warning(self, 'Invalid category', 'Categories must remain JSON objects.')
            return
        documents = self.documents()
        keys = []
        node = item
        while node is not None:
            keys.append(node.data(0, Qt.ItemDataRole.UserRole))
            node = node.parent()
        keys.reverse()
        target = documents
        for key in keys[:-1]:
            target = target[key]
        if keys == ['runtime']:
            # Source references and backend/task belong to the profile/tab controls.
            documents['runtime'].update(value)
        else:
            target[keys[-1]] = value
        self.set_documents(documents)
        self.documentsChanged.emit()

    def load_document(self, name):
        filename, _ = QFileDialog.getOpenFileName(self, 'Load ' + TITLES[name], str(ROOT / 'config'), 'JSON (*.json)')
        if not filename:
            return
        try:
            if name == 'runtime':
                # Runtime references identify the rest of the profile.
                if Path(filename).name != 'runtime_config.json':
                    raise ValueError('Load a profile directory containing runtime_config.json')
                self.set_documents(read_profile(Path(filename).parent))
                self.documentsChanged.emit()
                return
            value = parse_json(Path(filename).read_text())
            if not isinstance(value, dict):
                raise ValueError('Expected a JSON object')
            if name == 'hardware' and value.get('robot', {}).get('api_path'):
                value['robot']['api_path'] = str((Path(filename).parent / value['robot']['api_path']).resolve())
            documents = self.documents()
            documents[name] = value
            self.set_documents(documents)
            self.documentsChanged.emit()
        except (OSError, ValueError, KeyError, TypeError) as exc:
            QMessageBox.warning(self, 'Load failed', str(exc))


class BackendPanel(QWidget):
    def __init__(self, backend):
        super().__init__()
        self.backend = backend
        layout = QVBoxLayout(self)
        row = QHBoxLayout()
        row.addWidget(QLabel('Experiment'))
        self.task = QComboBox()
        for label, key in [('Observe', 'observe'), ('Servo', 'servo'), ('Pick', 'pick'), ('Pick and place', 'pick_place')]:
            self.task.addItem(label, key)
        self.task.setCurrentIndex(0 if backend == 'hardware' else 2)
        row.addWidget(self.task)
        row.addWidget(QLabel('Destination'))
        self.destination = QComboBox()
        self.destination.setEditable(True)
        self.destination.setMinimumWidth(180)
        row.addWidget(self.destination)
        row.addStretch()
        layout.addLayout(row)
        self.tree = ConfigTree(read_profile(ROOT / 'config'), backend)
        layout.addWidget(self.tree)
        self.task.currentIndexChanged.connect(self.refresh_destinations)
        self.tree.documentsChanged.connect(self.refresh_destinations)
        self.refresh_destinations()

    def refresh_destinations(self):
        previous = self.destination.currentText()
        self.destination.clear()
        destinations = self.tree.documents()['sequence'].get('destinations', {})
        if isinstance(destinations, dict):
            self.destination.addItems(destinations)
        if previous:
            self.destination.setCurrentText(previous)
        self.destination.setEnabled(self.task.currentData() == 'pick_place')


class ExperimentWindow(QMainWindow):
    def __init__(self, runs_dir=None):
        super().__init__()
        self.setWindowTitle('Robotic Studio | Experiments')
        self.resize(1080, 820)
        self.runs_dir = Path(runs_dir or ROOT / 'experiment_runs')
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        self.process = None
        self.directory = None
        self.stop_requested = False
        self.run_after_validation = False
        self.log_file = None
        self.decoder = codecs.getincrementaldecoder('utf-8')(errors='replace')
        self.central = QWidget()
        self.setCentralWidget(self.central)
        layout = QVBoxLayout(self.central)
        toolbar = QHBoxLayout()
        self.load_button = button(self, 'Load profile', QStyle.StandardPixmap.SP_DialogOpenButton, self.load_profile)
        self.save_button = button(self, 'Save profile', QStyle.StandardPixmap.SP_DialogSaveButton, self.save_profile)
        toolbar.addWidget(self.load_button)
        toolbar.addWidget(self.save_button)
        toolbar.addStretch()
        self.validate_button = button(self, 'Validate', QStyle.StandardPixmap.SP_DialogApplyButton, lambda: self.prepare(False))
        self.start_button = button(self, 'Start', QStyle.StandardPixmap.SP_MediaPlay, lambda: self.prepare(True))
        self.stop_button = button(self, 'Stop', QStyle.StandardPixmap.SP_MediaStop, self.stop,
                                  'Request normal shutdown; not an emergency stop')
        self.stop_button.setEnabled(False)
        toolbar.addWidget(self.validate_button)
        toolbar.addWidget(self.start_button)
        toolbar.addWidget(self.stop_button)
        layout.addLayout(toolbar)
        splitter = QSplitter(Qt.Orientation.Vertical)
        self.tabs = QTabWidget()
        for backend in ('simulation', 'hardware'):
            self.tabs.addTab(BackendPanel(backend), backend.title())
        splitter.addWidget(self.tabs)
        output = QWidget()
        output_layout = QVBoxLayout(output)
        output_layout.setContentsMargins(0, 0, 0, 0)
        self.status = QLabel('Ready')
        self.status.setTextFormat(Qt.TextFormat.PlainText)
        output_layout.addWidget(self.status)
        self.logs = QPlainTextEdit()
        self.logs.setReadOnly(True)
        self.logs.setMaximumBlockCount(5000)
        self.logs.setFont(QFontDatabase.systemFont(QFontDatabase.SystemFont.FixedFont))
        output_layout.addWidget(self.logs)
        splitter.addWidget(output)
        splitter.setSizes([560, 200])
        layout.addWidget(splitter)
        self.stop_timer = QTimer(self)
        self.stop_timer.setSingleShot(True)
        self.stop_timer.timeout.connect(self.shutdown_pending)

    @property
    def panel(self):
        return self.tabs.currentWidget()

    def busy(self, value):
        for widget in (self.tabs, self.load_button, self.save_button, self.validate_button, self.start_button):
            widget.setEnabled(not value)
        self.stop_button.setEnabled(value)

    def load_profile(self):
        directory = QFileDialog.getExistingDirectory(self, 'Load profile', str(ROOT / 'config'))
        if not directory:
            return
        if QMessageBox.question(self, 'Replace draft?', 'Replace settings in the selected backend tab?',
                                defaultButton=QMessageBox.StandardButton.No) != QMessageBox.StandardButton.Yes:
            return
        try:
            self.panel.tree.set_documents(read_profile(directory))
            self.panel.refresh_destinations()
            self.status.setText('Loaded: ' + directory)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            QMessageBox.warning(self, 'Load failed', str(exc))

    def save_profile(self):
        directory = QFileDialog.getExistingDirectory(self, 'Save profile', str(ROOT))
        if not directory:
            return
        if any((Path(directory) / (name + '_config.json')).exists() for name in FILES):
            if QMessageBox.question(self, 'Overwrite profile?', 'Replace configuration files in this directory?',
                                    defaultButton=QMessageBox.StandardButton.No) != QMessageBox.StandardButton.Yes:
                return
        try:
            documents = self.panel.tree.documents()
            documents['runtime']['backend'] = self.panel.backend
            task = self.panel.task.currentData()
            documents['runtime']['task'] = 'pick' if task == 'pick_place' else task
            write_profile(directory, documents)
            self.status.setText('Saved: ' + directory)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            QMessageBox.warning(self, 'Save failed', str(exc))

    def prepare(self, run):
        if self.process is not None:
            return
        try:
            self.directory = snapshot(self.panel.tree.documents(), self.panel.backend,
                self.panel.task.currentData(), self.panel.destination.currentText(), self.runs_dir)
            self.log_file = (self.directory / 'console.log').open('a', encoding='utf-8')
        except (OSError, ValueError, KeyError, TypeError) as exc:
            QMessageBox.warning(self, 'Invalid configuration', str(exc))
            return
        self.logs.clear()
        self.append('Run directory: ' + str(self.directory) + '\n')
        self.stop_requested = False
        self.run_after_validation = run
        self.busy(True)
        self.launch(validate=True)

    def launch(self, *, validate):
        self.validating = validate
        self.status.setText('Validating' if validate else 'Running: ' + self.panel.backend)
        self.decoder.reset()
        process = QProcess(self)
        self.process = process
        process.setWorkingDirectory(str(ROOT))
        process.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        env = QProcessEnvironment.systemEnvironment()
        env.insert('PYTHONUNBUFFERED', '1')
        # The child uses OpenCV/MuJoCo, not the launcher's Qt plugin paths.
        for key in ('QT_QPA_PLATFORM_PLUGIN_PATH', 'QT_QPA_FONTDIR'):
            env.remove(key)
        process.setProcessEnvironment(env)
        process.readyReadStandardOutput.connect(self.read_output)
        process.finished.connect(self.finished)
        process.errorOccurred.connect(self.process_error)
        args = ['-u', '-m', 'desktop.worker', str(self.directory)]
        if validate:
            args.append('--validate')
        process.start(sys.executable, args)

    def append(self, text):
        cursor = self.logs.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        cursor.insertText(text)
        self.logs.setTextCursor(cursor)
        if self.log_file is not None:
            try:
                self.log_file.write(text)
                self.log_file.flush()
            except OSError:
                self.log_file.close()
                self.log_file = None
                self.logs.appendPlainText('Log file unavailable; output remains in this window.')

    def read_output(self):
        if self.process is not None:
            self.append(self.decoder.decode(bytes(self.process.readAllStandardOutput())))

    def process_error(self, error):
        if error == QProcess.ProcessError.FailedToStart:
            self.append('Failed to start: ' + self.process.errorString() + '\n')
            self.finish_run('Launch failed')

    def finished(self, code, exit_status):
        self.read_output()
        self.append(self.decoder.decode(b'', final=True))
        self.process.deleteLater()
        self.process = None
        succeeded = code == 0 and exit_status == QProcess.ExitStatus.NormalExit
        if self.validating and succeeded and self.run_after_validation and not self.stop_requested:
            if self.panel.backend == 'hardware':
                host = self.panel.tree.documents()['hardware'].get('robot', {}).get('host', 'custom adapter')
                answer = QMessageBox.warning(self, 'Confirm hardware operation',
                    f'Robot: {host}\nExperiment: {self.panel.task.currentText()}\n\n'
                    'This run connects to the real robot and camera and may move the arm or gripper.\n\n'
                    'Verify the workspace, installation calibration and limits on the pendant. '
                    'Keep the physical emergency stop accessible. GUI Stop is not an emergency stop.\n\n'
                    'Start hardware operation?', QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                    QMessageBox.StandardButton.No)
                if answer != QMessageBox.StandardButton.Yes:
                    self.finish_run('Hardware run cancelled')
                    return
            self.launch(validate=False)
            return
        if self.stop_requested:
            status = 'Process exited after stop request'
        elif succeeded:
            status = 'Validation passed' if self.validating else 'Completed'
        else:
            status = f'Failed (exit {code})'
        if self.panel.backend == 'hardware' and not self.validating:
            status += ' | Verify robot state on pendant'
        self.finish_run(status)

    def finish_run(self, status):
        self.stop_timer.stop()
        if self.process is not None:
            self.process.deleteLater()
            self.process = None
        self.status.setText(status)
        self.append('\n' + status + '\n')
        if self.log_file is not None:
            self.log_file.close()
            self.log_file = None
        self.busy(False)

    def stop(self):
        if self.process is None or self.stop_requested:
            return
        try:
            (self.directory / 'stop.request').touch()
        except OSError as exc:
            QMessageBox.critical(self, 'Stop request failed', str(exc) + '\nUse the robot pendant to stop hardware.')
            return
        self.stop_requested = True
        self.run_after_validation = False
        self.stop_button.setEnabled(False)
        self.status.setText('Shutdown requested; waiting for process cleanup')
        self.stop_timer.start(10000)

    def shutdown_pending(self):
        self.status.setText('Shutdown unconfirmed. Check the robot pendant; the process is still active.')
        self.append('\nShutdown has not completed. No forced termination was sent.\n')

    def closeEvent(self, event):
        if self.process is not None:
            event.ignore()
            if QMessageBox.question(self, 'Experiment active', 'Request shutdown? The window stays open until the process exits.',
                                    defaultButton=QMessageBox.StandardButton.No) == QMessageBox.StandardButton.Yes:
                self.stop()
        else:
            for index in range(self.tabs.count()):
                tree = self.tabs.widget(index).tree
                tree.getters.clear()
                tree.clear()
            event.accept()


def main():
    app = QApplication.instance() or QApplication(sys.argv)
    app.setApplicationName('Robotic Studio')
    app.setStyle('Fusion')
    app.setFont(QFont('Sans Serif', 10))
    app.setStyleSheet('''
        QMainWindow, QWidget { background: #f4f6f7; color: #202629; }
        QTreeWidget, QPlainTextEdit, QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {
            background: white; selection-background-color: #d5eee6; selection-color: #202629;
        }
        QTreeWidget { alternate-background-color: #f4f7f8; }
        QPushButton { padding: 5px 10px; }
        QTabBar::tab { padding: 8px 24px; }
        QTabBar::tab:selected { border-bottom: 3px solid #16836c; }
        QHeaderView::section { background: #e5eaed; padding: 6px; border: 0; }
    ''')
    window = ExperimentWindow()
    window.show()
    result = app.exec()
    window.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    return result
