import json

import pytest

from common.runtime_config import DEFAULT_RUNTIME_CONFIG_PATH, RuntimeConfig


@pytest.mark.parametrize('enabled', [True, False])
def test_annotations_load_and_override(tmp_path, enabled):
    values = json.loads(DEFAULT_RUNTIME_CONFIG_PATH.read_text())
    values['annotations'] = enabled
    path = tmp_path / 'runtime.json'
    path.write_text(json.dumps(values))
    config = RuntimeConfig.load(path)
    assert config.annotations is enabled
    assert config.override(annotations=not enabled).annotations is not enabled


def test_legacy_runtime_defaults_to_annotations(tmp_path):
    values = json.loads(DEFAULT_RUNTIME_CONFIG_PATH.read_text())
    values.pop('annotations')
    path = tmp_path / 'runtime.json'
    path.write_text(json.dumps(values))
    assert RuntimeConfig.load(path).annotations is True


@pytest.mark.parametrize('value', [0, 1, 'false', None])
def test_annotations_requires_boolean(value):
    from dataclasses import replace
    with pytest.raises(ValueError, match='annotations must be boolean'):
        replace(RuntimeConfig.load(), annotations=value)
