import json
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from controller import PBVS, CubePickupController
from controller_config import ControllerConfig
from robot_session import PickAction
from hardware_fakes import configured
from estimator import EKFConfig


FIELDS = ("approach_linear_speed_m_s", "approach_angular_speed_rad_s",
          "pickup_linear_speed_m_s", "pickup_angular_speed_rad_s")


@pytest.mark.parametrize("value", [-1, True, 1.5, "3", None])
def test_invalid_reapproach_count(value):
    with pytest.raises(ValueError, match="pickup_reapproach_attempts"):
        ControllerConfig(pickup_reapproach_attempts=value)


@pytest.mark.parametrize("field", FIELDS)
@pytest.mark.parametrize("value", [0, -1, True, float('nan'), float('inf'), "0.1"])
def test_invalid_speed_config(field, value):
    with pytest.raises(ValueError, match=field):
        ControllerConfig(**{field: value})


def test_config_load_defaults_and_overrides(tmp_path):
    path = tmp_path / "controller.json"
    path.write_text("{}")
    config = ControllerConfig.load(path)
    assert [getattr(config, f) for f in FIELDS] == [.06, .3, .03, .3]
    values = dict(zip(FIELDS, [.02, .1, .01, .08]))
    path.write_text(json.dumps(values))
    assert all(getattr(ControllerConfig.load(path), k) == v for k, v in values.items())


def test_pbvs_limits_translation_and_rotation_independently():
    pbvs = PBVS(np.eye(4), linear_speed=.02, angular_speed=.08)
    target = np.eye(4)
    target[:3, 3] = [.2, .1, .3]
    target[:3, :3] = Rotation.from_euler('x', .5).as_matrix()
    command, _ = pbvs.reference(target, np.eye(6))
    assert np.linalg.norm(command[:3]) == pytest.approx(.02 / (1 + pbvs.damping**2))
    assert np.linalg.norm(command[3:]) == pytest.approx(.08 / (1 + pbvs.damping**2))


@pytest.mark.parametrize("mode", ["pure_pbvs", "pbvs_mpc"])
def test_sequence_wires_all_configured_speeds(mode):
    h = configured()
    config = ControllerConfig(mode=mode, **dict(zip(FIELDS, [.02, .1, .01, .08])))
    session = SimpleNamespace(settings=h.build_mpc_config(), controller_config=config,
        interface=Mock(), kinematics=h.build_kinematics(), hardware=h,
        estimator_config=EKFConfig(), cube_size=.064)
    action = PickAction(session)
    assert action.approach.pbvs.linear_speed == .02
    assert action.approach.pbvs.angular_speed == .1
    assert action.pickup.speed == .01
    assert action.pickup.angular_speed == .08


def test_pickup_preserves_separate_xy_z_limits_and_configures_rotation():
    current = np.eye(4)
    current[2, 3] = .4
    state = SimpleNamespace(timestamp=10., joint_positions=np.zeros(6),
                            joint_velocities=np.zeros(6), gripper_width=.12)
    interface = Mock(timestamp=10., gripper_max_width=.12)
    interface.read_state.return_value = state
    kin = Mock(limits=(np.full(6, -10.), np.full(6, 10.)))
    kin.pickup_collision_indices.return_value = []
    kin.tool_pose.return_value = current
    kin.grasp_pose.return_value = current
    kin.tool_jacobian.return_value = np.eye(6)
    kin.distances.return_value = np.array([])
    pickup = CubePickupController(interface, kin, speed=.01, angular_speed=.08)
    cube = np.eye(4)
    cube[2, 3] = .2
    pickup.start(SimpleNamespace(timestamp=10., world_from_cube=cube))
    pickup.update()
    pickup._descent_pose[0, 3] += .1
    pickup._descent_pose[:3, :3] = Rotation.from_euler('x', .5).as_matrix()
    result = pickup.update()
    assert result.status == "pickup_descending"
    assert np.linalg.norm(result.command_velocity[:2]) == pytest.approx(.01 / 1.0001)
    assert abs(result.command_velocity[2]) == pytest.approx(.01 / 1.0001)
    assert np.linalg.norm(result.command_velocity[3:]) == pytest.approx(.08 / 1.0001)
