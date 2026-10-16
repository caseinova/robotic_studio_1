"""D435i SDK contract checks; no camera, network or installed SDK required."""

from types import SimpleNamespace
import unittest

import numpy as np

from hardware_fakes import Clock, Robot, Gripper, configured
from hardware_realsense import RealSenseD435i
from interface import HardwareInterface


class Frame:
    def __init__(self, data, timestamp, number, domain='global'):
        self.data, self.timestamp, self.number, self.domain = data, timestamp, number, domain

    def get_data(self): return self.data
    def get_timestamp(self): return self.timestamp * 1000
    def get_frame_number(self): return self.number
    def get_frame_timestamp_domain(self): return self.domain


class FakeSDK:
    def __init__(self, settings, clock):
        self.settings, self.clock = settings, clock
        self.serial_selected, self.streams, self.sensor_options = None, [], []
        self.started = self.stopped = self.number = 0
        self.override_frame = None
        self.wall_jump = 0
        self.name, self.domain = 'Intel RealSense D435I', 'global'
        self.scale = .001
        self.skew, self.age = .002, .02
        self.stream = SimpleNamespace(color='color', depth='depth')
        self.format = SimpleNamespace(rgb8='rgb8', z16='z16')
        self.camera_info = SimpleNamespace(name='name', serial_number='serial')
        self.option = SimpleNamespace(global_time_enabled='global_time', frames_queue_size='queue')
        self.timestamp_domain = SimpleNamespace(global_time='global')
        self.distortion = SimpleNamespace(none='none', brown_conrady='brown')
        self.intr = SimpleNamespace(width=settings['width'], height=settings['height'], fx=50., fy=51.,
                                    ppx=31.5, ppy=23.5, model='brown', coeffs=[.01, 0, 0, 0, 0])
        sensor = SimpleNamespace(supports=lambda option: True,
                                 set_option=lambda option, value: self.sensor_options.append((option, value)))
        self.device = SimpleNamespace(get_info=lambda key: self.name if key == 'name' else 'SERIAL',
            query_sensors=lambda: [sensor], first_depth_sensor=lambda: SimpleNamespace(get_depth_scale=lambda: self.scale))
        self.devices = [self.device]
        self.profile = SimpleNamespace(get_device=lambda: self.device,
            get_stream=lambda stream: SimpleNamespace(as_video_stream_profile=lambda:
                SimpleNamespace(get_intrinsics=lambda: self.intr)))

    def wall_clock(self): return 1000 + self.clock.now - 100 + self.wall_jump
    def context(self): return SimpleNamespace(query_devices=lambda: self.devices)
    def pipeline(self): return self
    def pipeline_wrapper(self, pipeline): return pipeline

    def config(self):
        return SimpleNamespace(enable_device=lambda serial: setattr(self, 'serial_selected', serial),
            enable_stream=lambda *args: self.streams.append(args), resolve=lambda wrapper: self.profile)

    def start(self, config):
        self.started += 1
        return self.profile

    def stop(self): self.stopped += 1
    def align(self, target): return SimpleNamespace(process=lambda frames: frames)

    def wait_for_frames(self, timeout):
        if self.override_frame is not None:
            return self.override_frame
        self.number += 1
        shape = (self.settings['height'], self.settings['width'])
        color = Frame(np.full((*shape, 3), [20, 40, 80], dtype=np.uint8),
                      self.wall_clock() - self.age, self.number, self.domain)
        depth = Frame(np.full(shape, 1000, dtype=np.uint16),
                      self.wall_clock() - self.age - self.skew, self.number, self.domain)
        self.last_frames = SimpleNamespace(get_color_frame=lambda: color, get_depth_frame=lambda: depth)
        return self.last_frames


class RealSenseTests(unittest.TestCase):
    def setUp(self):
        self.config, self.clock = configured(), Clock()
        self.config.camera['factory'] = 'hardware_realsense:create_camera'
        self.config.camera['options']['warmup_frames'] = 0
        self.sdk = FakeSDK(self.config.camera, self.clock)

    def camera(self):
        return RealSenseD435i(self.config.camera, self.clock, sdk=self.sdk, wall_clock=self.sdk.wall_clock)

    def test_streams_rgb_and_depth_alignment_owned_arrays_and_metric_interface(self):
        camera = self.camera()
        self.assertEqual({stream[0] for stream in self.sdk.streams}, {'color', 'depth'})
        self.assertIn(('global_time', 1), self.sdk.sensor_options)
        sample = camera.capture()
        self.assertAlmostEqual(self.clock() - sample.timestamp, .027)
        self.sdk.last_frames.get_color_frame().data[:] = 0
        np.testing.assert_array_equal(sample.rgb[0, 0], [20, 40, 80])
        self.clock.sleep(.01)
        interface = HardwareInterface(self.config, Robot(self.clock), camera, Gripper(self.clock), clock=self.clock)
        frame = interface.capture_rgbd()
        np.testing.assert_allclose(frame.depth, 1.)
        interface.close()
        self.assertEqual(self.sdk.stopped, 1)

    def test_device_serial_selection_is_explicit_with_multiple_cameras(self):
        self.sdk.devices *= 2
        with self.assertRaisesRegex(RuntimeError, 'multiple'):
            self.camera()
        self.config.camera['options']['serial'] = 'SERIAL'
        camera = self.camera()
        self.assertEqual(self.sdk.serial_selected, 'SERIAL')
        camera.close()

    def test_hardware_clock_is_not_restamped_as_delivery_time(self):
        camera = self.camera()
        self.sdk.domain = 'hardware'
        with self.assertRaisesRegex(RuntimeError, 'global acquisition'):
            camera.capture()
        camera.close()

    def test_duplicate_frames_and_excessive_color_depth_skew_are_rejected(self):
        camera = self.camera()
        camera.capture()
        self.sdk.override_frame = self.sdk.last_frames
        self.clock.sleep(.02)
        with self.assertRaisesRegex(RuntimeError, 'repeated'):
            camera.capture()
        self.sdk.override_frame = None
        self.sdk.skew = .1
        with self.assertRaisesRegex(RuntimeError, 'skew'):
            camera.capture()
        camera.close()

    def test_old_acquisition_timestamp_remains_old(self):
        camera = self.camera()
        self.sdk.age = .2
        sample = camera.capture()
        self.assertGreater(self.clock() - sample.timestamp, .2)
        interface = HardwareInterface(self.config, Robot(self.clock), camera, Gripper(self.clock), clock=self.clock)
        with self.assertRaisesRegex(RuntimeError, 'stale'):
            interface.capture_rgbd()
        interface.close()

    def test_clock_jump_future_stamp_and_missing_depth_are_rejected(self):
        camera = self.camera()
        self.sdk.wall_jump = .1
        with self.assertRaisesRegex(RuntimeError, 'wall clock'):
            camera.capture()
        self.sdk.wall_jump, self.sdk.age = 0, -.02
        with self.assertRaisesRegex(RuntimeError, 'future'):
            camera.capture()
        self.sdk.override_frame = SimpleNamespace(get_color_frame=lambda: object(), get_depth_frame=lambda: None)
        with self.assertRaisesRegex(RuntimeError, 'complete'):
            camera.capture()
        camera.close()

    def test_bad_depth_scale_or_profile_cleans_up_pipeline(self):
        self.sdk.scale = .002
        with self.assertRaisesRegex(ValueError, 'depth_scale'):
            self.camera()
        self.assertEqual(self.sdk.stopped, 1)
        self.sdk.scale, self.sdk.intr.width = .001, 1
        with self.assertRaisesRegex(RuntimeError, 'resolution'):
            self.camera()
        self.assertEqual(self.sdk.stopped, 2)

    def test_calibration_report_has_intrinsics_and_depth_scale_without_extrinsics(self):
        camera = self.camera()
        report = camera.factory_calibration()
        self.assertEqual(report['camera']['intrinsics'], [[50., 0, 31.5], [0, 51., 23.5], [0, 0, 1]])
        self.assertEqual(report['camera']['depth_scale'], .001)
        self.assertNotIn('world_from_camera', report['camera'])
        self.sdk.intr.model = 'unsupported'
        self.assertIsNone(camera.factory_calibration()['camera']['distortion'])
        camera.close()
        camera.close()
        self.assertEqual(self.sdk.stopped, 1)


if __name__ == '__main__':
    unittest.main()
