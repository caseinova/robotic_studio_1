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


def test_fused_tool_tracker_is_shared_across_sequence_actions():
    from common.pickup import PickAction
    from common.placement import PlaceAction
    from common.camera_processing import TaggedCubeProcessor
    config = EKFConfig(tool_enabled=True, tool={'prediction_model': 'fk_increment'})
    simulation = Simulation(estimator_config=config,
                            controller_config=ControllerConfig(tool_pose_source='tag_with_fk_fallback'))
    with SimulationSession(simulation, settings()) as session:
        pick = PickAction(session, processor=TaggedCubeProcessor())
        tracker = pick.approach.tool_tracker
        assert tracker is not None
        assert pick.pickup.tool_tracker is tracker
        session.tool_from_cube = np.eye(4)
        session.kinematics = SimulationPlacementKinematics(simulation.robot, 'table')
        place = PlaceAction(session, settings().destinations['drop_zone_b'],
                            guard_factory=SimulationPlacementGuard,
                            frame_max_age=.2, acceleration_limit=.5)
        assert place.tool_tracker is tracker
        assert tracker.kinematics is session.kinematics


@pytest.mark.parametrize('marker_visible,expected', [(True, 'raw_tool_tag'), (False, 'fk_fallback')])
def test_sequence_logs_control_result_pose_with_tool_ekf_off(marker_visible, expected):
    from common.pickup import PickAction
    from common.camera_processing import TaggedCubeProcessor
    simulation = Simulation(controller_config=ControllerConfig(mode='pure_pbvs',
        tool_pose_source='tag_with_fk_fallback'), estimator_config=EKFConfig(enabled=False),
        noise_config=NoiseConfig(enabled=False))
    with SimulationSession(simulation, settings()) as session:
        action = PickAction(session, processor=TaggedCubeProcessor(
            cube_size=session.cube_size, marker_size=simulation.robot.cube_marker_size))
        marker = SimpleNamespace(camera_from_marker=np.eye(4)) if marker_visible else None
        original_update = action.update
        count = [0]

        def update():
            original_update()
            count[0] += 1
            # Telemetry must not depend on this mutable controller cache.
            action.approach.latest_tool_pose = None
            return count[0] == 3

        with patch.object(action.approach.tool_processor, 'process_marker', return_value=marker), \
             patch.object(action, 'update', side_effect=update):
            session._run(action, 'test')
        samples = simulation.estimation.samples
        assert len(samples) == 3
        assert [sample['tool'][3] for sample in samples] == [expected] * 3
        assert all(np.isfinite(sample['tool'][1]).all() and np.isnan(sample['tool'][2]).all()
                   for sample in samples)
        assert action.last_approach_result.tool_pose is not None


@pytest.mark.parametrize('camera_view', [True, False])
@pytest.mark.parametrize('annotations', [True, False])
def test_pick_preview_reuses_control_observation(camera_view, annotations):
    from common.pickup import PickAction
    from common.camera_processing import TaggedCubeProcessor
    with SimulationSession(Simulation(), settings()) as session:
        session.camera_view = camera_view
        session.annotations = annotations
        processor = TaggedCubeProcessor(cube_size=.064, marker_size=.04)
        action = PickAction(session, processor=processor)
        frame = session.interface.capture_rgbd()
        original = frame.rgb.copy()
        with patch.object(session.interface, 'capture_rgbd', return_value=frame) as capture, \
             patch.object(processor, 'process', wraps=processor.process) as detect, \
             patch.object(processor, 'annotate', wraps=processor.annotate) as annotate:
            action.update()
        capture.assert_called_once()
        detect.assert_called_once()
        assert annotate.call_count == int(camera_view and annotations)
        preview = session.latest_preview()
        assert preview.timestamp == frame.timestamp
        if camera_view and annotations:
            assert preview is not frame
            assert np.any(preview.rgb != original)
        else:
            assert preview is frame
        np.testing.assert_array_equal(frame.rgb, original)


def test_sequence_camera_preview_closes_on_cancellation():
    from common.runtime_config import RuntimeConfig
    from simulation.simulation_session import run_simulation_sequence
    runtime = RuntimeConfig.load().override(camera=True, headless=False)
    with patch('simulation.viewer.launch_viewer') as viewer, \
         patch('common.sequence_preview.SequencePreview') as preview:
        viewer.return_value.__enter__.return_value.is_running.return_value = True
        preview.return_value.update.side_effect = RuntimeError('sequence cancelled from camera window')
        with pytest.raises(RuntimeError, match='cancelled'):
            run_simulation_sequence(runtime, settings(), 'drop_zone_b')
        preview.assert_called_once()
        preview.return_value.update.assert_called_once()
        preview.return_value.close.assert_called_once()


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
