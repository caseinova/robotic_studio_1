from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch

import mujoco
import numpy as np
import pytest

from common.camera_processing import CameraProcessor
from common.controller_config import ControllerConfig
from common.sequence_config import SequenceConfig, Destination
from simulation.simulate import Simulation
from simulation.simulation_config import SimulationConfig
from simulation.simulation_session import SimulationSession, SimulationPlacementGuard, SimulationPlacementKinematics
from simulation.noise_config import NoiseConfig
from common.estimator import EKFConfig


def settings():
    target = np.eye(4)
    target[2, 3] = .04
    return SequenceConfig(destinations={'drop_zone_b': Destination(7, .08, target, 'table',
        approach_height_m=.05, retreat_height_m=.05, max_drop_height_m=.02)})


def test_destination_config_requires_floor_pose():
    for positions in ({'b': [0, 0, .1]}, {'b': [0, float('nan'), 0]}, {'b': [True, 0, 0]}):
        with pytest.raises(ValueError):
            SimulationConfig(destination_positions=positions)


def test_scene_detects_destination_from_rendered_pixels():
    with SimulationSession(Simulation(), settings()) as session:
        frame = session.interface.capture_rgbd()
        processor = CameraProcessor(marker_id=7, marker_size=.08)
        tag = processor.process_marker(frame)
        assert tag is not None, processor.last_status
        world = frame.world_from_camera @ tag.camera_from_marker
        np.testing.assert_allclose(world[:3, 3], [.12, .32, .0001], atol=.002)
        assert session.simulation.robot.model.joint('cube_free_joint').type == mujoco.mjtJoint.mjJNT_FREE


def test_simulation_prediction_never_moves_physics_cube():
    with SimulationSession(Simulation(), settings()) as session:
        robot = session.simulation.robot
        kin = SimulationPlacementKinematics(robot, 'table')
        before_q = robot.data.qpos.copy()
        before_size = robot.model.geom_size.copy()
        mount = np.eye(4)
        mount[2, 3] = .195
        guard = SimulationPlacementGuard(kin, .064, .015, mount, 'table')
        guard.released_pose = np.eye(4)
        guard.released_pose[:3, 3] = [.12, .32, .04]
        guard.released_half_sizes += .001
        guard.check(robot.joint_positions)
        np.testing.assert_array_equal(robot.data.qpos, before_q)
        np.testing.assert_array_equal(robot.model.geom_size, before_size)


def test_guard_blocks_payload_below_table():
    with SimulationSession(Simulation(), settings()) as session:
        robot = session.simulation.robot
        kin = SimulationPlacementKinematics(robot, 'table')
        guard = SimulationPlacementGuard(kin, .064, .015, np.eye(4), 'table')
        guard.released_pose = np.eye(4)
        guard.released_pose[:3, 3] = [.12, .32, .01]
        with pytest.raises(RuntimeError, match='placement_path_blocked'):
            guard.check(robot.joint_positions)


def test_actions_chain_and_failure_stops_without_release():
    with SimulationSession(Simulation(), settings()) as session:
        calls = []
        class Pick:
            def __init__(self, s, **kwargs):
                self.s = s
                self.count = 0
                assert kwargs['processor'].cube_size == .064
            def update(self):
                self.count += 1
                calls.append('pick')
                if self.count == 2:
                    self.s.tool_from_cube = np.eye(4)
                    return True
                return False
        class Place:
            def __init__(self, s, destination, **kwargs):
                self.s = s
                assert kwargs['guard_factory'] is SimulationPlacementGuard
            def update(self):
                calls.append('place')
                self.s.tool_from_cube = None
                return True
        with patch('simulation.simulation_session.PickAction', Pick), patch('simulation.simulation_session.PlaceAction', Place):
            assert session.servo_to_pick_up().holding_cube
            assert session.interface.timestamp >= session.settings.dt
            assert not session.move_to_destination_and_drop('drop_zone_b').holding_cube
        assert calls == ['pick', 'pick', 'place']
        with pytest.raises(RuntimeError, match='pick up'):
            session.move_to_destination_and_drop('drop_zone_b')
        session.tool_from_cube = np.eye(4)
        action = SimpleNamespace(update=lambda: (_ for _ in ()).throw(RuntimeError('blocked')))
        with patch.object(session.interface, 'open_gripper') as release:
            with pytest.raises(RuntimeError, match='blocked'):
                session._run(action, 'place')
            release.assert_not_called()
        assert session.failed


def test_action_timeout_is_simulation_time_and_blocks_reuse():
    config = replace(settings(), action_timeout_s=.1)
    with SimulationSession(Simulation(), config) as session:
        with pytest.raises(TimeoutError):
            session._run(SimpleNamespace(update=lambda: False), 'test')
        assert session.interface.timestamp >= .1
        assert session.failed


def test_physical_pick_and_place_with_rendered_destination():
    simulation = Simulation(
        controller_config=ControllerConfig(mode='pure_pbvs', tool_clearance=.25,
            grasp_height_offset_m=.04, pickup_dwell_seconds=2., frame_max_age_intervals=4.),
        estimator_config=EKFConfig.load(), noise_config=NoiseConfig.load())
    config = replace(settings(), cartesian_speed_m_s=.03, angular_speed_rad_s=.3,
                     transfer_timeout_s=30.)
    with SimulationSession(simulation, config) as session:
        assert session.servo_to_pick_up().holding_cube
        assert simulation.robot.data.body('cube').xpos[2] > .055
        assert not session.move_to_destination_and_drop('drop_zone_b').holding_cube
        simulation.step(500)
        placed = simulation.robot.data.body('cube').xpos.copy()
        np.testing.assert_allclose(placed[:2], [.12, .32], atol=.01)
        assert placed[2] == pytest.approx(.032, abs=.002)
        assert simulation.interface.gripper_width > .099
