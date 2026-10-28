"""Device transports used by HardwareInterface; optional SDKs import lazily.

UR RTDE API: https://sdurobotics.gitlab.io/ur_rtde/pages/reference/api.html
OnRobot RG2/RG6 register map: OnRobot user manual, Modbus TCP section.
The D435i camera uses hardware_realsense; other cameras can supply a factory.
"""

from dataclasses import dataclass
import importlib
import socket
import struct
import threading
import time

import numpy as np


@dataclass(frozen=True)
class JointSample:
    timestamp: float
    positions: np.ndarray
    velocities: np.ndarray


@dataclass(frozen=True)
class WidthSample:
    timestamp: float
    width: float


@dataclass(frozen=True)
class CameraSample:
    """Registered RGB uint8 and raw optical-Z depth, before rectification.

    timestamp is acquisition time mapped to the supplied monotonic clock,
    not the time an old buffered frame is delivered. Depth is in the units
    specified by camera.depth_scale; RGB/depth already share the RGB optics.
    """

    timestamp: float
    rgb: np.ndarray
    depth: np.ndarray


def custom_device(settings, clock):
    module, name = settings["factory"].split(":", 1)
    factory = getattr(importlib.import_module(module), name, None)
    if not callable(factory):
        raise ValueError(f"adapter factory is not callable: {settings['factory']}")
    return factory(config=settings, clock=clock)


class URRTDE:
    """CB-series receive connection and a 125 Hz velocity-command worker.

    Only the worker owns RTDEControlInterface (the SDK is not thread safe).
    Command age is checked independently of the slower vision/MPC loop.
    An on-robot SDK watchdog also stops control if worker communication dies.
    Control startup is explicit via prepare_motion or the first motion command.
    """

    def __init__(self, settings, control, clock=time.monotonic):
        self.clock, self.control, self.settings = clock, control, settings
        self._connect_receive()
        self._read_lock, self._command_lock = threading.Lock(), threading.Lock()
        self._stop_event = threading.Event()
        self._ready_event = threading.Event()
        self._thread = None
        self._error = None
        self._worker_phase = "not_started"
        self._command = None
        self._anchor = None
        self._closed = False

    def _connect_receive(self):
        try:
            import rtde_receive
            import rtde_control
        except ImportError as error:
            raise RuntimeError("UR3 hardware requires ur-rtde; see requirements-hardware.txt") from error
        self._control_type = rtde_control.RTDEControlInterface
        self._receive = rtde_receive.RTDEReceiveInterface(self.settings["host"], self.settings["frequency_hz"])

    def _joint_values(self):
        return self._receive.getActualQ(), self._receive.getActualQd()

    def _connect_command(self):
        return self._control_type(self.settings["host"], self.settings["frequency_hz"])

    def _watchdog_transport(self, sender):
        return sender

    def _send_velocity(self, sender, velocity, period):
        return sender.speedJ(velocity.tolist(), self.control["acceleration_limit"], period)

    def _close_command(self, sender):
        for action in (lambda: sender.speedStop(self.control["acceleration_limit"]),
                       sender.stopScript, sender.disconnect):
            try:
                action()
            except Exception:
                pass  # Preserve the original worker failure.

    def _close_receive(self):
        self._receive.disconnect()

    def read_joint_state(self):
        if self._closed or self._error is not None:
            raise RuntimeError(f"UR transport unavailable: {self._error or 'closed'}")
        with self._read_lock:
            if not self._receive.isConnected():
                raise RuntimeError("UR receive connection lost")
            if self._receive.getSafetyMode() not in (1, 2):
                raise RuntimeError("UR safety mode does not permit motion")
            # Detect a packet changing while its individual fields are read.
            for _ in range(5):
                before = self._receive.getTimestamp()
                positions, velocities = self._joint_values()
                q = np.asarray(positions, dtype=float)
                dq = np.asarray(velocities, dtype=float)
                after = self._receive.getTimestamp()
                if before == after:
                    break
            else:
                raise RuntimeError("could not acquire a coherent UR joint snapshot")
            if not np.isfinite(after) or q.shape != (6,) or dq.shape != (6,) or not np.all(np.isfinite([q, dq])):
                raise RuntimeError("invalid UR joint snapshot")
            if self._anchor is None:
                # One sample period of conservative clock-mapping uncertainty.
                self._anchor = (after, self.clock() - 1 / self.settings["frequency_hz"])
            if after < self._anchor[0]:
                raise RuntimeError("UR clock restarted; reconnect the hardware session")
            timestamp = self._anchor[1] + after - self._anchor[0]
            return JointSample(timestamp, q.copy(), dq.copy())

    def _queue(self, mode, target):
        if self._closed or self._stop_event.is_set() or self._error is not None:
            raise RuntimeError(f"UR command transport stopped: {self._error or 'closed'}")
        with self._command_lock:
            self._command = (mode, np.array(target, dtype=float, copy=True), self.clock())
        if self._thread is None:
            self._thread = threading.Thread(target=self._run, name="ur3-command-worker", daemon=True)
            self._thread.start()

    def command_velocity(self, target):
        self._queue("velocity", target)

    def command_position(self, target):
        self._queue("position", target)

    def prepare_motion(self, timeout=10.):
        """Open control on its owning worker, without queuing a motion target."""
        if self._closed or self._stop_event.is_set():
            raise RuntimeError("UR command transport stopped")
        if self._thread is None:
            self._thread = threading.Thread(target=self._run, name="ur3-command-worker", daemon=True)
            self._thread.start()
        if not self._ready_event.wait(timeout):
            self._stop_event.set()
            raise TimeoutError("UR command connection startup timed out")
        if self._error is not None or self._stop_event.is_set():
            raise RuntimeError(f"UR command startup failed: {self._error or 'stopped'}")

    def _run(self):
        rtde = None
        connection_started = self.clock()
        sent_commands = 0
        try:
            self._worker_phase = "connecting_control"
            rtde = self._connect_command()
            connection_elapsed = self.clock() - connection_started
            watchdog = self._watchdog_transport(rtde)
            self._worker_phase = "enabling_watchdog"
            if not watchdog.setWatchdog(1 / self.control["command_timeout"]):
                raise RuntimeError("UR command watchdog could not be enabled")
            self._ready_event.set()
            period = 1 / self.settings["frequency_hz"]
            while not self._stop_event.is_set():
                self._worker_phase = "checking_command"
                started = self.clock()
                with self._command_lock:
                    command = self._command
                if command is None:
                    # Keep the watchdog alive before the first target, never replay
                    # or fabricate a command while waiting for fresh vision/state.
                    if not watchdog.kickWatchdog():
                        raise RuntimeError("UR idle watchdog kick failed")
                    self._stop_event.wait(period)
                    continue
                mode, target, timestamp = command
                if started - timestamp > self.control["command_timeout"]:
                    raise RuntimeError("UR command expired: upper-level control stopped updating; "
                        f"command_age_ms={(started - timestamp) * 1000:.1f}, "
                        f"timeout_ms={self.control['command_timeout'] * 1000:.1f}, "
                        f"connect_ms={connection_elapsed * 1000:.1f}, sent_commands={sent_commands}")
                self._worker_phase = "reading_joint_state"
                measured = self.read_joint_state()
                if not 0 <= self.clock() - measured.timestamp <= self.control["state_max_age"]:
                    raise RuntimeError("UR joint stream is stale")
                velocity = target if mode == "velocity" else np.clip(
                    5 * (target - measured.positions), -self.control["velocity_limit"], self.control["velocity_limit"])
                lower = np.asarray(self.settings["joint_lower"])
                upper = np.asarray(self.settings["joint_upper"])
                if np.any(measured.positions < lower) or np.any(measured.positions > upper):
                    raise RuntimeError("UR measured joints are outside configured position limits")
                # Bound stopping distance too, not just one worker interval.
                acceleration = self.control["acceleration_limit"]
                braking_lower = -np.sqrt(2 * acceleration * np.maximum(measured.positions - lower, 0))
                braking_upper = np.sqrt(2 * acceleration * np.maximum(upper - measured.positions, 0))
                velocity = np.clip(velocity, np.maximum(braking_lower, -self.control["velocity_limit"]),
                                   np.minimum(braking_upper, self.control["velocity_limit"]))
                self._worker_phase = "kicking_watchdog"
                if not watchdog.kickWatchdog():
                    raise RuntimeError("UR rejected the watchdog kick")
                self._worker_phase = "sending_velocity"
                if not self._send_velocity(rtde, velocity, period):
                    raise RuntimeError("UR rejected the velocity command")
                sent_commands += 1
                self._stop_event.wait(max(0., period - (self.clock() - started)))
        except Exception as error:
            self._error = error
        finally:
            self._stop_event.set()
            self._ready_event.set()
            if rtde is not None:
                self._worker_phase = "closing_control"
                try:
                    self._close_command(rtde)
                except Exception as error:
                    if self._error is None:
                        self._error = error
            self._worker_phase = "stopped"

    def stop(self):
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=self.control["command_timeout"] + 1)
            if self._thread.is_alive():
                raise RuntimeError(f"UR command worker did not stop (phase={self._worker_phase}); "
                                   "stop completion unconfirmed; verify robot stopped on pendant")
        if self._error is not None:
            raise RuntimeError(f"UR command worker failed: {self._error}") from self._error

    def close(self):
        if not self._closed:
            try:
                self.stop()
            finally:
                self._closed = True
                self._close_receive()


class ModbusTCP:
    """Small synchronous FC3/FC16 client with bounded reads and checked echoes."""

    def __init__(self, host, port, unit_id, timeout):
        self.socket = socket.create_connection((host, port), timeout=timeout)
        self.socket.settimeout(timeout)
        self.unit_id, self.transaction = unit_id, 0
        self._lock = threading.Lock()

    def _read(self, count):
        data = bytearray()
        while len(data) < count:
            part = self.socket.recv(count - len(data))
            if not part:
                raise RuntimeError("OnRobot Modbus connection closed")
            data.extend(part)
        return bytes(data)

    def _request(self, pdu):
        with self._lock:
            self.transaction = (self.transaction + 1) & 65535
            header = struct.pack(">HHHB", self.transaction, 0, len(pdu) + 1, self.unit_id)
            try:
                self.socket.sendall(header + pdu)
                transaction, protocol, length, unit = struct.unpack(">HHHB", self._read(7))
                if (transaction != self.transaction or protocol != 0 or unit != self.unit_id
                        or not 2 <= length <= 254):
                    raise RuntimeError("invalid OnRobot Modbus response header")
                response = self._read(length - 1)
                if response[0] != pdu[0]:
                    raise RuntimeError(f"OnRobot Modbus exception/function mismatch: {response.hex()}")
                return response
            except Exception:
                self.close()  # Never reuse an ambiguous stream or replay a failed actuator write.
                raise

    def read_register(self, address):
        response = self._request(struct.pack(">BHH", 3, address, 1))
        if len(response) != 4 or response[1] != 2:
            raise RuntimeError("invalid OnRobot register response")
        return struct.unpack(">H", response[2:])[0]

    def write_registers(self, address, values):
        count = len(values)
        response = self._request(struct.pack(">BHHB", 16, address, count, count * 2)
                                 + struct.pack(">" + "H" * count, *values))
        if response != struct.pack(">BHH", 16, address, count):
            raise RuntimeError("OnRobot register write was not acknowledged")

    def close(self):
        self.socket.close()


class OnRobotRG:
    """RG2/RG6 Compute Box Modbus TCP, units converted to meters/Newtons.

    Configuration selects model and unit ID; no activation, homing, power
    cycling, or fingertip-offset writes are performed. Existing device
    fingertip calibration is honored when use_device_fingertip_offset=true.
    """

    def __init__(self, settings, clock=time.monotonic):
        self.settings, self.clock = settings, clock
        self._bus = ModbusTCP(settings["host"], settings["port"], settings["unit_id"], settings["timeout_s"])

    def _status(self):
        status = self._bus.read_register(268)
        if status & 0x7c:
            raise RuntimeError(f"OnRobot safety switch/error flags are set: {status:#x}")
        return status

    def read_width(self):
        timestamp = self.clock()  # Conservative acquisition time before bounded I/O.
        self._status()
        value = self._bus.read_register(275 if self.settings["use_device_fingertip_offset"] else 267)
        if value >= 32768:
            value -= 65536
        return WidthSample(timestamp, value * .0001)

    def command_width(self, width):
        if self._status() & 1:
            raise RuntimeError("OnRobot is busy and cannot accept a new width command")
        self._bus.write_registers(0, [round(self.settings["force_n"] * 10), round(width * 10000),
                                     16 if self.settings["use_device_fingertip_offset"] else 1])

    def close(self):
        self._bus.close()  # Disconnect only: do not release a held object.
