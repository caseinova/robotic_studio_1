import copy
from unittest.mock import Mock, patch
import xmlrpc.client

import pytest
import pycurl

from hardware_onrobot_xmlrpc import OnRobotXMLRPC, validate_settings


SETTINGS = dict(host='192.168.0.192', port=41414, options={'gripper_id': 0},
                model='RG2', force_n=20., max_width_m=.11, timeout_s=.1,
                use_device_fingertip_offset=True)


def transport(value, status=200):
    handle = Mock()
    options = {}
    handle.setopt.side_effect = lambda key, value: options.__setitem__(key, value)
    handle.getinfo.return_value = status
    handle.perform.side_effect = lambda: options[pycurl.WRITEFUNCTION](
        xmlrpc.client.dumps((value,), methodresponse=True, allow_none=True).encode())
    return handle, options


def test_read_units_timestamp_and_bounded_wire_request():
    handle, options = transport(80.)
    with patch.object(pycurl, 'Curl', return_value=handle):
        gripper = OnRobotXMLRPC(SETTINGS, clock=lambda: 12.)
        sample = gripper.read_width()
    assert sample.width == .08 and sample.timestamp == 12.
    assert xmlrpc.client.loads(options[pycurl.POSTFIELDS]) == ((0,), 'rg_get_width')
    assert options[pycurl.URL] == 'http://192.168.0.192:41414/'
    assert options[pycurl.TIMEOUT_MS] == options[pycurl.CONNECTTIMEOUT_MS] == 100
    handle.close.assert_called_once()


@pytest.mark.parametrize('ack', [None, True, 1])
def test_command_conversion_and_close_has_no_actuation(ack):
    handle, options = transport(ack)
    with patch.object(pycurl, 'Curl', return_value=handle) as factory:
        gripper = OnRobotXMLRPC(SETTINGS)
        gripper.command_width(.08)
        assert xmlrpc.client.loads(options[pycurl.POSTFIELDS]) == ((0, 80., 20.), 'rg_grip')
        gripper.close()
        factory.assert_called_once()


@pytest.mark.parametrize('value', [True, '80', float('nan'), -1, 200])
def test_bad_width_rejected(value):
    handle, _ = transport(value)
    with patch.object(pycurl, 'Curl', return_value=handle):
        gripper = OnRobotXMLRPC(SETTINGS)
        with pytest.raises(RuntimeError):
            gripper.read_width()


def test_timeout_does_not_retry_command():
    handle, _ = transport(True)
    handle.perform.side_effect = pycurl.error(28, 'timeout')
    with patch.object(pycurl, 'Curl', return_value=handle) as factory:
        gripper = OnRobotXMLRPC(SETTINGS)
        with pytest.raises(RuntimeError, match='not retried'):
            gripper.command_width(.05)
        with pytest.raises(RuntimeError, match='closed'):
            gripper.command_width(.05)
        factory.assert_called_once()
        handle.close.assert_called_once()


@pytest.mark.parametrize('ack', [False, 0, 'ok'])
def test_negative_or_unknown_ack_rejected(ack):
    handle, _ = transport(ack)
    with patch.object(pycurl, 'Curl', return_value=handle):
        with pytest.raises(RuntimeError, match='acknowledgment'):
            OnRobotXMLRPC(SETTINGS).command_width(.05)


@pytest.mark.parametrize('key,value', [('force_n', 41), ('max_width_m', .12),
    ('timeout_s', 0), ('port', 0), ('options', {'gripper_id': True})])
def test_settings_validation(key, value):
    config = copy.deepcopy(SETTINGS)
    config[key] = value
    with pytest.raises(ValueError):
        validate_settings(config)
