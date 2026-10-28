import struct
import sys
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from hardware_devices import ModbusTCP, OnRobotRG, URRTDE
from hardware_fakes import Clock, configured


class PacketSocket:
    """Return one byte at a time to exercise TCP fragmentation."""
    def __init__(self, response):
        self.response = bytearray(response)
        self.sent = b''
        self.closed = False

    def settimeout(self, timeout):
        pass

    def sendall(self, message):
        self.sent = message

    def recv(self, count):
        data = self.response[:1]
        del self.response[:1]
        return bytes(data)

    def close(self):
        self.closed = True


class HardwareDeviceTests(unittest.TestCase):
    def test_slow_prepare_sends_no_target_then_accepts_fresh_command(self):
        config, clock = configured(), Clock()
        receiver, sender = Mock(), Mock()
        receiver.isConnected.return_value = True
        receiver.getSafetyMode.return_value = 1
        receiver.getTimestamp.side_effect = lambda: clock() + 1000
        receiver.getActualQ.return_value = receiver.getActualQd.return_value = [0.] * 6
        sent = threading.Event()

        def connect():
            clock.sleep(2.)
            return sender

        with patch.object(URRTDE, '_connect_receive', lambda arm: setattr(arm, '_receive', receiver)), \
             patch.object(URRTDE, '_connect_command', side_effect=connect):
            arm = URRTDE(config.robot, config.control, clock)
            arm.prepare_motion()
            self.assertIsNone(arm._command)
            sender.speedJ.assert_not_called()
            sender.setWatchdog.assert_called_once_with(1 / config.control['command_timeout'])

            def speed(*args):
                arm._stop_event.set()
                sent.set()
                return True

            sender.speedJ.side_effect = speed
            arm.command_velocity(np.zeros(6))
            self.assertTrue(sent.wait(1))
            arm.close()
            self.assertIsNone(arm._error)

    def test_prepare_propagates_connection_failure_without_waiting_for_timeout(self):
        config = configured()
        with patch.object(URRTDE, '_connect_receive', lambda arm: setattr(arm, '_receive', Mock())), \
             patch.object(URRTDE, '_connect_command', side_effect=RuntimeError('offline')):
            arm = URRTDE(config.robot, config.control)
            with self.assertRaisesRegex(RuntimeError, 'startup failed.*offline'):
                arm.prepare_motion(timeout=1)
            with self.assertRaisesRegex(RuntimeError, 'offline'):
                arm.close()
            arm._receive.disconnect.assert_called_once()

    def test_modbus_handles_split_packets_and_validates_transaction(self):
        for transaction, accepted in ((1, True), (2, False)):
            sock = PacketSocket(struct.pack('>HHHBBBH', transaction, 0, 5, 65, 3, 2, 1000))
            with patch('hardware_devices.socket.create_connection', return_value=sock):
                bus = ModbusTCP('offline', 502, 65, .1)
                if accepted:
                    self.assertEqual(bus.read_register(275), 1000)
                    self.assertEqual(sock.sent[7:], struct.pack('>BHH', 3, 275, 1))
                else:
                    with self.assertRaisesRegex(RuntimeError, 'header'):
                        bus.read_register(275)
                    self.assertTrue(sock.closed)

    def test_modbus_write_checks_acknowledged_address_and_count(self):
        pdu = struct.pack('>BHH', 16, 0, 3)
        sock = PacketSocket(struct.pack('>HHHB', 1, 0, len(pdu) + 1, 65) + pdu)
        with patch('hardware_devices.socket.create_connection', return_value=sock):
            bus = ModbusTCP('offline', 502, 65, .1)
            bus.write_registers(0, [200, 1000, 16])
        self.assertEqual(sock.sent[7:], struct.pack('>BHHBHHH', 16, 0, 3, 6, 200, 1000, 16))

    def test_onrobot_uses_measured_width_and_atomic_force_width_control(self):
        bus = Mock()
        bus.read_register.side_effect = [0, 950, 0]
        with patch('hardware_devices.ModbusTCP', return_value=bus):
            device = OnRobotRG(configured().gripper, Clock())
        self.assertAlmostEqual(device.read_width().width, .095)
        device.command_width(.08)
        bus.write_registers.assert_called_once_with(0, [200, 800, 16])
        device.close()
        # Cleanup never opens the fingers or power-cycles the device.
        self.assertEqual(bus.write_registers.call_count, 1)
        bus.close.assert_called_once()

    def test_onrobot_busy_and_safety_flags_do_not_silently_accept_commands(self):
        for status in (1, 4, 8, 16, 32, 64):
            bus = Mock()
            bus.read_register.return_value = status
            with patch('hardware_devices.ModbusTCP', return_value=bus):
                device = OnRobotRG(configured().gripper, Clock())
            with self.subTest(status=status), self.assertRaises(RuntimeError):
                device.command_width(.04)
            bus.write_registers.assert_not_called()

    def stop_test_device(self, phase, alive):
        device = URRTDE.__new__(URRTDE)
        device.control = {'command_timeout': .25}
        device._stop_event = threading.Event()
        device._thread = Mock()
        device._thread.is_alive.side_effect = alive
        device._worker_phase = phase
        device._error = None
        return device

    def test_stop_waits_for_thread_unwinding_after_cleanup(self):
        device = self.stop_test_device('stopped', [True, False])
        device.stop()
        self.assertTrue(device._stop_event.is_set())
        self.assertEqual(device._thread.join.call_count, 2)
        device._thread.join.assert_called_with(timeout=.1)

    def test_stopped_phase_does_not_hide_live_worker(self):
        device = self.stop_test_device('stopped', [True, True])
        with self.assertRaisesRegex(RuntimeError, 'stop completion unconfirmed'):
            device.stop()

    def test_blocked_control_cleanup_still_fails(self):
        device = self.stop_test_device('closing_control', [True, True])
        with self.assertRaisesRegex(RuntimeError, 'phase=closing_control'):
            device.stop()
        device._thread.join.assert_called_once_with(timeout=1.25)

    def test_cleanup_error_survives_thread_termination(self):
        device = self.stop_test_device('stopped', [True, False])
        device._error = RuntimeError('control close failed')
        with self.assertRaisesRegex(RuntimeError, 'control close failed'):
            device.stop()

    def test_rtde_worker_expires_command_enables_robot_watchdog_and_stops(self):
        clock = Clock()
        config = configured()
        config.control['command_timeout'] = .025
        receiver = Mock()
        receiver.isConnected.return_value = True
        receiver.getSafetyMode.return_value = 1
        receiver.getTimestamp.side_effect = lambda: clock() + 1000
        receiver.getActualQ.return_value = [0.] * 6
        receiver.getActualQd.return_value = [0.] * 6
        sender = Mock()
        sender.setWatchdog.return_value = sender.kickWatchdog.return_value = True

        def speed(target, acceleration, period):
            self.assertLessEqual(max(abs(v) for v in target), .15)
            clock.sleep(period)
            return True

        sender.speedJ.side_effect = speed
        with patch.dict(sys.modules, {
            'rtde_receive': SimpleNamespace(RTDEReceiveInterface=Mock(return_value=receiver)),
            'rtde_control': SimpleNamespace(RTDEControlInterface=Mock(return_value=sender)),
        }):
            device = URRTDE(config.robot, config.control, clock)
        # Receive-only connection uploads no script / sends no motion.
        sender.speedJ.assert_not_called()
        sample = device.read_joint_state()
        self.assertAlmostEqual(clock() - sample.timestamp, .008)
        device.command_position(np.ones(6))
        device._thread.join(timeout=2)
        self.assertFalse(device._thread.is_alive())
        self.assertIn('expired', str(device._error))
        sender.setWatchdog.assert_called_once_with(40.)
        sender.speedStop.assert_called_once()
        sender.stopScript.assert_called_once()
        with self.assertRaises(RuntimeError):
            device.close()
        receiver.disconnect.assert_called_once()


if __name__ == '__main__':
    unittest.main()
