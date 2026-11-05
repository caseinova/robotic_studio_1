import json
from unittest.mock import patch

import pytest
import numpy as np

from common.controller import CubePickupController
from common.controller_config import ControllerConfig
from common.pickup import PickAction
from common.camera_processing import TaggedCubeProcessor
from common.sequence_config import SequenceConfig, Destination
from simulation.simulate import Simulation
from simulation.simulation_config import SimulationConfig
from simulation.simulation_session import SimulationSession


@pytest.mark.parametrize('name', ['approach_collision_clearance_m', 'placement_collision_clearance_m',
                                 'mpc_clearance_buffer_m'])
@pytest.mark.parametrize('value', [-1, True, None, float('nan'), float('inf')])
def test_invalid_other_clearances(name, value):
    with pytest.raises(ValueError, match=name):
        SimulationConfig(**{name: value})


def test_independent_simulation_margins():
    config = SimulationConfig(approach_collision_clearance_m=.021,
        pickup_collision_clearance_m=.012, placement_collision_clearance_m=.018,
        mpc_clearance_buffer_m=.004)
    simulation = Simulation(config=config, controller_config=ControllerConfig(mode='pbvs_mpc'))
    with SimulationSession(simulation, SequenceConfig()) as session:
        action = PickAction(session, processor=TaggedCubeProcessor())
        assert action.approach.mpc.config.clearance == .021
        assert action.approach.mpc.config.clearance_buffer == .004
        assert action.pickup.clearance == .012
        from common.placement import PlaceAction
        from simulation.simulation_session import SimulationPlacementGuard, SimulationPlacementKinematics
        session.tool_from_cube = np.eye(4)
        destination = Destination(7, .08, np.eye(4), 'table')
        session.kinematics = SimulationPlacementKinematics(simulation.robot, 'table')
        place = PlaceAction(session, destination, guard_factory=SimulationPlacementGuard,
                            frame_max_age=.2, acceleration_limit=.5)
        assert place.guard.clearance == .018
        assert place.transfer_mpc.config.clearance == .018
        assert place.transfer_mpc.config.clearance_buffer == .004


def test_zero_buffer_allowed_but_zero_collision_margin_rejected():
    assert SimulationConfig(mpc_clearance_buffer_m=0).build_mpc_config().clearance_buffer == 0
    for name in ('approach_collision_clearance_m', 'placement_collision_clearance_m'):
        with pytest.raises(ValueError, match=name):
            SimulationConfig(**{name: 0})


@pytest.mark.parametrize('value', [0, -1, True, None, '0.01', float('nan'), float('inf')])
def test_invalid_pickup_clearance(value):
    with pytest.raises(ValueError, match='pickup_collision_clearance_m'):
        SimulationConfig(pickup_collision_clearance_m=value)


def test_load_pickup_clearance_and_legacy_default(tmp_path):
    path = tmp_path / 'simulation.json'
    path.write_text('{}')
    assert SimulationConfig.load(path).pickup_collision_clearance_m == .015
    path.write_text(json.dumps({'pickup_collision_clearance_m': .01}))
    assert SimulationConfig.load(path).pickup_collision_clearance_m == .01


def test_standalone_pickup_receives_clearance():
    simulation = Simulation(pickup=True, config=SimulationConfig(pickup_collision_clearance_m=.01))
    with patch('common.controller.CubePickupController', wraps=CubePickupController) as factory:
        simulation.run_controlled(0)
    assert factory.call_args.kwargs['clearance'] == .01


def test_sequence_pickup_receives_clearance_without_changing_other_margins():
    simulation = Simulation(config=SimulationConfig(pickup_collision_clearance_m=.01))
    with SimulationSession(simulation, SequenceConfig()) as session:
        action = PickAction(session, processor=TaggedCubeProcessor())
        assert action.pickup.clearance == .01
        assert session.settings.clearance == .015
