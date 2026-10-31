from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from common.controller_config import ControllerConfig
from hardware.hardware_config import HardwareConfig
from common.runtime_config import RuntimeConfig, DEFAULT_RUNTIME_CONFIG_PATH
from hardware_fakes import configured


class HardwareConfigTests(unittest.TestCase):
    def test_frame_budget_uses_servo_multiplier_not_state_multiplier(self):
        config = configured()
        config.collision['links'] = []
        config.control['frame_max_age'] = .2
        controller = ControllerConfig(mode='pure_pbvs', frame_max_age_intervals=4)
        config.validate_ready(controller, task='servo')
        config.validate_ready(ControllerConfig(mode='pure_pbvs'), task='observe')
        with self.assertRaisesRegex(ValueError, 'controller frame budget'):
            config.validate_ready(ControllerConfig(mode='pure_pbvs'), task='servo')
        with self.assertRaisesRegex(ValueError, 'link_capsules'):
            config.validate_ready(controller, task='pick')
        config.control['state_max_age'] = .101
        with self.assertRaisesRegex(ValueError, 'state_max_age'):
            config.validate_ready(controller)
        config.control['state_max_age'] = .1
        config.control['frame_max_age'] = .251
        with self.assertRaisesRegex(ValueError, 'command_timeout'):
            config.validate_schema()

    def test_missing_intrinsics_reported_without_connecting(self):
        config = configured()
        config = replace(config, camera={**config.camera, "intrinsics": None})
        with self.assertRaisesRegex(ValueError, 'camera.intrinsics'):
            config.validate_ready(ControllerConfig(mode='pure_pbvs'))

    def test_complete_calibration_builds_ur3_and_perception(self):
        config = configured()
        config.validate_ready(ControllerConfig(mode='pure_pbvs'))
        np.testing.assert_allclose(config.build_kinematics().tool_pose()[:3, 3], [-.4569, -.19425, .06655])
        self.assertEqual(set(config.build_processor().cube_from_markers), {0, 2, 3, 4, 5, 6})
        self.assertEqual(config.build_mpc_config().velocity_limit, .15)

    def test_bad_transforms_ids_intrinsics_and_limits_rejected(self):
        for section, key, value in [('calibration', 'world_from_camera', np.zeros((4, 4)).tolist()),
                                    ('camera', 'intrinsics', [[0] * 3] * 3),
                                    ('robot', 'joint_upper', [-7] * 6),
                                    ('perception', 'tool_marker_id', 0),
                                    ('perception', 'reference_marker_id', 49)]:
            config = configured()
            getattr(config, section)[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                config.validate_ready(ControllerConfig(mode='pure_pbvs'))

    def test_mpc_and_pickup_require_explicit_collision_geometry(self):
        config = configured()
        config.collision['links'] = []
        for mode, task in [('pbvs_mpc', 'servo'), ('pure_pbvs', 'pick')]:
            with self.subTest(mode=mode, task=task), self.assertRaisesRegex(ValueError, 'link_capsules'):
                config.validate_ready(ControllerConfig(mode=mode), task=task)
        config.collision['links'] = [dict(name='test_link', frame=1, start=[0, 0, 0], end=[0, 0, .1], radius=.03)]
        config.validate_ready(ControllerConfig(mode='pbvs_mpc'))

    def test_onrobot_model_force_and_stroke_are_configurable_and_checked(self):
        config = configured()
        config.gripper.update(model='RG6', force_n=50, max_width_m=.15)
        config.validate_ready(ControllerConfig(mode='pure_pbvs'))
        config.gripper['model'] = 'RG2'
        with self.assertRaises(ValueError):
            config.validate_ready(ControllerConfig(mode='pure_pbvs'))
        config.gripper['model'] = '2FG7'
        with self.assertRaisesRegex(ValueError, 'RG2/RG6'):
            config.validate_ready(ControllerConfig(mode='pure_pbvs'))

    def test_camera_can_supply_depth_in_meters(self):
        config = configured()
        config.camera['depth_scale'] = 1.
        config.validate_ready(ControllerConfig(mode='pure_pbvs'))
        config.camera['depth_scale'] = 0
        with self.assertRaises(ValueError):
            config.validate_ready(ControllerConfig(mode='pure_pbvs'))

    def test_unknown_keys_and_unsafe_timing_fail_validation(self):
        config = configured()
        config.robot['misspelled_limit'] = 1
        with self.assertRaisesRegex(ValueError, 'unknown robot'):
            config.validate_schema()
        config = configured()
        config.control['command_timeout'] = .01
        with self.assertRaises(ValueError):
            config.validate_schema()

    def test_configurable_cube_reference_is_used(self):
        config = configured()
        config.perception['reference_marker_id'] = 2
        config.validate_ready(ControllerConfig(mode='pure_pbvs'))
        self.assertEqual(config.build_processor().reference_marker_id, 2)

    def test_runtime_paths_resolve_relative_to_file_not_working_directory(self):
        values = json.loads(DEFAULT_RUNTIME_CONFIG_PATH.read_text())
        with tempfile.TemporaryDirectory() as folder:
            file = Path(folder) / 'runtime.json'
            file.write_text(json.dumps(values))
            config = RuntimeConfig.load(file)
            self.assertEqual(config.hardware_config, Path(folder).resolve() / 'hardware_config.json')
            self.assertEqual(config.override(backend='hardware').backend, 'hardware')
            with self.assertRaises(ValueError):
                replace(config, backend='real')


if __name__ == '__main__':
    unittest.main()
