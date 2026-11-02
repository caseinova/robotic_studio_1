"""Configuration bundles and immutable experiment snapshots."""
from copy import deepcopy
from datetime import datetime
import json
import os
from pathlib import Path
import tempfile

ROOT = Path(__file__).resolve().parents[1]
FILES = ('runtime', 'controller', 'sequence', 'estimator', 'simulation', 'noise', 'hardware')
TASKS = ('observe', 'servo', 'pick', 'pick_place')


def parse_json(text):
    def reject(value):
        raise ValueError(f'Non-finite JSON value: {value}')
    return json.loads(text, parse_constant=reject)


def read_profile(directory):
    directory = Path(directory).resolve()
    runtime = parse_json((directory / 'runtime_config.json').read_text())
    if not isinstance(runtime, dict):
        raise ValueError('Runtime configuration must be a JSON object')
    runtime.setdefault('annotations', True)
    documents = {'runtime': runtime}
    for name in FILES[1:]:
        path = directory / runtime.get(name + '_config', name + '_config.json')
        document = parse_json(path.read_text())
        if not isinstance(document, dict):
            raise ValueError(f'{path}: expected a JSON object')
        # Installation paths are relative to the hardware file, not runtime.json.
        if name == 'hardware' and document.get('robot', {}).get('api_path'):
            document['robot']['api_path'] = str((path.parent / document['robot']['api_path']).resolve())
        documents[name] = document
    return documents


def write_profile(directory, documents):
    """Write a self-contained settings bundle (assets remain external)."""
    directory = Path(directory)
    values = deepcopy(documents)
    for name in FILES:
        if not isinstance(values[name], dict):
            raise ValueError(f'{name}: expected a JSON object')
    for name in FILES[1:]:
        if name != 'sequence':
            values['runtime'][name + '_config'] = name + '_config.json'
    encoded = {name: json.dumps(values[name], indent=2, allow_nan=False) + '\n' for name in FILES}
    directory.mkdir(parents=True, exist_ok=True)
    for name, text in encoded.items():
        target = directory / (name + '_config.json')
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=directory, delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(text)
        try:
            os.replace(temporary, target)
        finally:
            temporary.unlink(missing_ok=True)


def snapshot(documents, backend, task, destination, runs_dir):
    if backend not in ('simulation', 'hardware') or task not in TASKS:
        raise ValueError('Unknown backend or experiment')
    if task == 'pick_place' and destination not in documents['sequence'].get('destinations', {}):
        raise ValueError('Select a configured destination')
    directory = Path(tempfile.mkdtemp(prefix=datetime.now().strftime('%Y%m%d-%H%M%S-'), dir=runs_dir))
    values = deepcopy(documents)
    values['runtime']['backend'] = backend
    values['runtime']['task'] = 'pick' if task == 'pick_place' else task
    write_profile(directory, values)
    request = {'backend': backend, 'task': task, 'destination': destination}
    (directory / 'request.json').write_text(json.dumps(request, indent=2), encoding='utf-8')
    return directory


def arguments(directory, validate=False):
    directory = Path(directory).resolve()
    request = parse_json((directory / 'request.json').read_text())
    if request['backend'] not in ('simulation', 'hardware') or request['task'] not in TASKS:
        raise ValueError('Invalid experiment request')
    args = ['--config', str(directory / 'runtime_config.json'), '--backend', request['backend']]
    if request['task'] == 'pick_place':
        # Positional names beginning with '-' must not become CLI flags.
        args += ['--sequence-config', str(directory / 'sequence_config.json')]
        if validate:
            args += ['--validate']
        return 'run_sequence', args + ['--', request['destination']]
    args += ['--task', request['task']]
    if validate:
        args += ['--validate']
    return 'run', args
