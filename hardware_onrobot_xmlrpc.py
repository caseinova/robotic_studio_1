"""OnRobot RG XML-RPC adapter matching the supplied onRobot.gripper.RG API.

Uses one bounded HTTP request per operation, without actuator-write retries.
"""

from io import BytesIO
import math
import time
import xmlrpc.client

from hardware_devices import WidthSample


def validate_settings(config):
    if not isinstance(config.get('host'), str) or not config['host'].strip():
        raise ValueError('XML-RPC gripper.host must be set')
    if any(c in config['host'] for c in '/:@?#'):
        raise ValueError('XML-RPC gripper.host must be a hostname or IPv4 address, not a URL')
    if type(config.get('port')) is not int or not 1 <= config['port'] <= 65535:
        raise ValueError('XML-RPC gripper.port must be in 1..65535')
    options = config.get('options', {})
    if not isinstance(options, dict) or set(options) != {'gripper_id'}:
        raise ValueError('XML-RPC gripper.options requires only gripper_id')
    if type(options['gripper_id']) is not int or not 0 <= options['gripper_id'] <= 2147483647:
        raise ValueError('gripper_id must be a nonnegative XML-RPC integer')
    if config.get('model') not in ('RG2', 'RG6'):
        raise ValueError('XML-RPC gripper model must be RG2 or RG6')
    for key in ('timeout_s', 'force_n', 'max_width_m'):
        v = config.get(key)
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v <= 0:
            raise ValueError(f'gripper.{key} must be finite and positive')
    max_force, max_width = (40, .11) if config['model'] == 'RG2' else (120, .16)
    if config['force_n'] > max_force or config['max_width_m'] > max_width:
        raise ValueError('gripper force/opening exceeds selected model limits')
    if config.get('use_device_fingertip_offset') is not True:
        raise ValueError('XML-RPC uses rg_get_width; raw fingertip-offset selection is unsupported')


class OnRobotXMLRPC:
    def __init__(self, config, clock=time.monotonic):
        validate_settings(config)
        import pycurl
        self.curl = pycurl
        self.clock = clock
        self.url = f"http://{config['host']}:{config['port']}/"
        self.gripper_id = config['options']['gripper_id']
        self.timeout_ms = max(1, math.ceil(config['timeout_s'] * 1000))
        self.max_width = config['max_width_m']
        self.force = config['force_n']
        self._closed = False

    def _request(self, method, *args):
        if self._closed:
            raise RuntimeError('XML-RPC gripper is closed')
        body = xmlrpc.client.dumps((self.gripper_id, *args), methodname=method).encode('utf-8')
        response = BytesIO()
        curl = self.curl.Curl()
        def receive(chunk):
            if response.tell() + len(chunk) > 65536:
                raise RuntimeError('gripper response exceeds 64 KiB')
            return response.write(chunk)
        try:
            curl.setopt(self.curl.URL, self.url)
            curl.setopt(self.curl.HTTPHEADER, ['Content-Type: application/x-www-form-urlencoded'])
            curl.setopt(self.curl.POSTFIELDS, body)
            curl.setopt(self.curl.WRITEFUNCTION, receive)
            curl.setopt(self.curl.CONNECTTIMEOUT_MS, self.timeout_ms)
            curl.setopt(self.curl.TIMEOUT_MS, self.timeout_ms)
            curl.setopt(self.curl.NOSIGNAL, 1)
            curl.setopt(self.curl.PROXY, '')
            curl.perform()
            status = curl.getinfo(self.curl.RESPONSE_CODE)
            if status != 200:
                raise RuntimeError(f'HTTP status {status}')
            values, _ = xmlrpc.client.loads(response.getvalue())
            if len(values) != 1:
                raise ValueError('expected one XML-RPC response value')
            return values[0]
        except Exception as error:
            self._closed = True
            raise RuntimeError(f'OnRobot {method} failed (not retried): {error}') from error
        finally:
            curl.close()

    def read_width(self):
        timestamp = self.clock()  # Start of read, not delivery time after network latency.
        value = self._request('rg_get_width')
        if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value):
            self._closed = True
            raise RuntimeError('invalid OnRobot measured width')
        width = value / 1000.
        if not 0 <= width <= self.max_width + .0002:
            self._closed = True
            raise RuntimeError('OnRobot measured width outside configured physical opening')
        return WidthSample(timestamp, min(width, self.max_width))

    def command_width(self, width):
        if isinstance(width, bool) or not isinstance(width, (float, int)) or not math.isfinite(width) or not 0 <= width <= self.max_width:
            raise ValueError('gripper command outside configured opening in meters')
        result = self._request('rg_grip', float(width * 1000), float(self.force))
        # The wrist URCap returns integer 0; retain the other supported replies.
        # Acknowledgment is not proof of completed physical movement.
        if result is not None and not (result is True or type(result) is int and result in (0, 1)):
            self._closed = True
            raise RuntimeError(f'OnRobot rg_grip returned unexpected acknowledgment: {result!r}; not retried')

    def close(self):
        self._closed = True  # Each request owns its connection; no finger command on cleanup.


def create_gripper(config, clock):
    return OnRobotXMLRPC(config, clock)
