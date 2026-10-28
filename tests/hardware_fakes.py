"""Offline device fixtures. Nothing here opens a socket or SDK."""

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from hardware_config import HardwareConfig
from hardware_kinematics import UR3DH
from hardware_devices import CameraSample, JointSample, WidthSample


class Clock:
    def __init__(self):
        self.now = 100.

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


def configured():
    values = json.loads((Path(__file__).resolve().parents[1] / 'hardware_config.json').read_text())
    # Retain the legacy CB-series fixture independently of the installed arm selection.
    values['robot'].update(driver='ur_rtde', frequency_hz=125, dh=asdict(UR3DH()),
                           host='127.0.0.1', joint_lower=[-6.] * 6, joint_upper=[6.] * 6)
    values['gripper'].update(driver='onrobot_rg', host='127.0.0.1')
    values['camera'].update(driver='custom', factory='hardware_fakes:camera_factory', width=64, height=48,
                            intrinsics=[[50., 0, 31.5], [0, 50., 23.5], [0, 0, 1]])
    for key in values['calibration']:
        values['calibration'][key] = np.eye(4).tolist()
    return HardwareConfig(**values)


class Robot:
    def __init__(self, clock):
        self.clock, self.calls = clock, []
        self.age = 0
        self.q = np.zeros(6)
        self.closed = self.stopped = False

    def read_joint_state(self):
        return JointSample(self.clock() - self.age, self.q.copy(), np.zeros(6))

    def command_velocity(self, qd):
        self.calls.append(('velocity', np.array(qd)))

    def command_position(self, q):
        self.calls.append(('position', np.array(q)))

    def stop(self):
        self.stopped = True

    def close(self):
        self.closed = True


class Camera:
    def __init__(self, settings, clock):
        self.clock, self.settings = clock, settings
        self.closed = False
        self.age = 0

    def capture(self):
        h, w = self.settings['height'], self.settings['width']
        return CameraSample(self.clock() - self.age, np.full((h, w, 3), 120, dtype=np.uint8),
                            np.full((h, w), 1000, dtype=np.uint16))

    def close(self):
        self.closed = True


class Gripper:
    def __init__(self, clock):
        self.clock, self.width = clock, .1
        self.age = 0
        self.closed = False
        self.calls = []

    def read_width(self):
        return WidthSample(self.clock() - self.age, self.width)

    def command_width(self, width):
        self.calls.append(width)
        self.width = width

    def close(self):
        self.closed = True


def camera_factory(config, clock):
    return Camera(config, clock)
