from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from common.controller import CubePickupController


def setup_pickup(angles=(20, 35, 10), local_grasp=(0, 0, .1), dwell=0):
    cube = np.eye(4)
    cube[:3, :3] = Rotation.from_euler('xyz', angles, degrees=True).as_matrix()
    cube[:3, 3] = [.1, .2, .15]
    grasp = np.eye(4)
    grasp[:3, 3] = cube[:3, 3] + cube[:3, :3] @ local_grasp
    tool = grasp.copy()
    tool[2, 3] -= .195
    state = SimpleNamespace(timestamp=10., joint_positions=np.zeros(6),
                            joint_velocities=np.zeros(6), gripper_width=.12)
    interface = Mock(timestamp=10., gripper_max_width=.12)
    interface.read_state.return_value = state
    kin = Mock(limits=(np.full(6, -6.), np.full(6, 6.)))
    kin.tool_pose.return_value = tool
    kin.grasp_pose.return_value = grasp
    kin.tool_jacobian.return_value = np.eye(6)
    kin.pickup_collision_indices.return_value = [0]
    kin.distances.return_value = np.array([1.])
    kin.collision_pair_name.return_value = "obstacle"
    pickup = CubePickupController(interface, kin, cube_size=.064,
        grasp_height_offset_m=.01, pickup_dwell_seconds=dwell)
    observation = SimpleNamespace(timestamp=10., world_from_cube=cube)
    return pickup, interface, kin, observation, state


@pytest.mark.parametrize("angles", [(0, 0, 0), (20, 35, 10), (0, 90, 0)])
def test_cube_normal_descent_target_and_velocity(angles):
    pickup, interface, kin, obs, state = setup_pickup(angles)
    entry = kin.tool_pose().copy()
    pickup.start(obs)
    assert pickup.phase == "opening"
    rotation = obs.world_from_cube[:3, :3]
    displacement = pickup._descent_pose[:3, 3] - entry[:3, 3]
    np.testing.assert_allclose(displacement, rotation @ [0, 0, -.09], atol=1e-12)
    np.testing.assert_array_equal(pickup._descent_pose[:3, :3], entry[:3, :3])
    projected_grasp = kin.grasp_pose()[:3, 3] + displacement
    np.testing.assert_allclose(rotation.T @ (projected_grasp-obs.world_from_cube[:3, 3]),
                               [0, 0, .01], atol=1e-12)
    pickup.update()
    result = pickup.update()
    assert result.status == "pickup_descending"
    np.testing.assert_allclose(result.command_velocity[:3], rotation @ [0, 0, -.03/1.0001], atol=1e-12)


def test_world_centered_but_cube_off_center_is_rejected():
    pickup, interface, kin, obs, state = setup_pickup()
    kin.grasp_pose.return_value[:3, 3] = obs.world_from_cube[:3, 3] + [0, 0, .1]
    pickup.start(obs)
    assert pickup.phase == "failed"
    assert "tool_not_centered_over_cube" in pickup.diagnostic
    assert "frame=cube" in pickup.diagnostic
    interface.open_gripper.assert_not_called()


def test_tilted_dwell_uses_cube_xy_and_resets_outside_radius():
    pickup, interface, kin, obs, state = setup_pickup(dwell=1.)
    pickup.start(obs)
    pickup.update()
    assert pickup.phase == "dwelling"
    for now, x, phase in ((10., 0, "dwelling"), (10.6, .005, "dwelling"),
                          (10.7, .02, "dwelling"), (10.8, .005, "dwelling"),
                          (11.5, 0, "dwelling"), (11.9, 0, "descending")):
        interface.timestamp = state.timestamp = now
        kin.grasp_pose.return_value[:3, 3] = obs.world_from_cube[:3, 3] + obs.world_from_cube[:3, :3] @ [x, 0, .1]
        pickup.update()
        assert pickup.phase == phase


def test_tilted_path_still_checks_collisions():
    pickup, interface, kin, obs, state = setup_pickup()
    pickup.start(obs)
    pickup.update()
    kin.distances.return_value = np.array([-.01])
    result = pickup.update()
    assert result.status == "pickup_failed"
    assert "pickup_path_blocked" in result.diagnostic
    np.testing.assert_array_equal(interface.set_joint_velocities.call_args.args[0], np.zeros(6))


def test_nonrigid_cube_pose_cannot_define_descent_axes():
    pickup, interface, kin, obs, state = setup_pickup()
    obs.world_from_cube[:3, :3] *= 2
    pickup.start(obs)
    assert pickup.diagnostic == "invalid_pickup_target_or_state"
    interface.open_gripper.assert_not_called()
