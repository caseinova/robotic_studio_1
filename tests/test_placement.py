from types import SimpleNamespace

import numpy as np
import pytest

from hardware_kinematics import LinkCapsule, PlaneObstacle, SphereObstacle
from placement import PlaceAction, PlacementCollisionGuard, cube_plane_gap, vertical_drop_height
from sequence_config import Destination, SequenceConfig


def pose(z=0):
    result = np.eye(4)
    result[2, 3] = z
    return result


class Kinematics:
    def __init__(self):
        self.q = np.array([0., 0., .4, 0., 0., 0.])
        self.limits = (np.full(6, -6.), np.full(6, 6.))
        self.links = ()
        self.obstacles = (PlaneObstacle("table", (0, 0, 1), 0.),)
        self.gripper_width = .064
        self._frames = None

    def synchronize(self, state):
        self.q = state.joint_positions.copy()
        self.gripper_width = state.gripper_width

    def tool_pose(self):
        result = np.eye(4)
        result[:3, 3] = self.q[:3]
        return result

    def distances(self, q):
        self.q = q.copy()
        return np.array([.5])

    def collision_pair_name(self, i):
        return "arm/table"

    def tool_jacobian(self, q):
        self.q = q.copy()
        return np.eye(6)

    def _capsules(self, frames):
        return [(link.start, link.end, link.radius) for link in self.links]

    def set_obstacles(self, obstacles):
        self.obstacles = obstacles


def test_payload_plane_uses_oriented_cube_extent():
    plane = PlaneObstacle("table", (0, 0, 1), 0.)
    assert cube_plane_gap(pose(.032), .064, plane) == pytest.approx(0.)
    tilted = pose(.032)
    from scipy.spatial.transform import Rotation
    tilted[:3, :3] = Rotation.from_euler("x", 45, degrees=True).as_matrix()
    assert cube_plane_gap(tilted, .064, plane) < 0


def test_vertical_fall_uses_gravity_not_tilted_normal():
    from scipy.spatial.transform import Rotation
    tag = pose(.1)
    tag[:3, :3] = Rotation.from_euler("x", 20, degrees=True).as_matrix()
    normal = tag[:3, 2]
    plane = PlaneObstacle("tag", normal, float(normal @ tag[:3, 3]))
    cube = tag @ pose(.04)
    drop = vertical_drop_height(cube, .064, plane)
    assert drop == pytest.approx(.008 / normal[2])
    cube[2, 3] -= drop
    assert cube_plane_gap(cube, .064, plane) == pytest.approx(0.)
    with pytest.raises(ValueError, match="face upward"):
        vertical_drop_height(cube, .064, PlaneObstacle("bad", (0, 0, -1), 0))


def test_payload_collisions_and_support_contact():
    k = Kinematics()
    q = k.q.copy()
    q[2] = .032
    guard = PlacementCollisionGuard(k, .064, .02, np.eye(4), "table")
    guard.check(q)
    q[2] = .03
    with pytest.raises(RuntimeError, match="payload / table"):
        guard.check(q)
    q[2] = .032
    k.obstacles += (SphereObstacle("object", (0, 0, .032), .01),)
    with pytest.raises(RuntimeError, match="payload / object"):
        guard.check(q)


def test_only_finger_payload_contact_is_ignored():
    k = Kinematics()
    k.links = (LinkCapsule("palm", "tool", (0, 0, .4), (0, 0, .4), .01),)
    guard = PlacementCollisionGuard(k, .064, .02, np.eye(4), "table")
    with pytest.raises(RuntimeError, match="payload / palm"):
        guard.check(k.q)
    k.links = (LinkCapsule("finger", "tool", (0, 0, .4), (0, 0, .4), .01, contact_group="finger"),)
    guard.check(k.q)
    guard.released_pose = pose(.4)
    with pytest.raises(RuntimeError, match="payload / finger"):
        guard.check(k.q)


class Interface:
    def __init__(self, k):
        self.k = k
        self.timestamp = 100.
        self.q = k.q.copy()
        self.velocity = np.zeros(6)
        self.width = .064
        self.gripper_max_width = .1005
        self.open_count = 0

    def read_state(self):
        return SimpleNamespace(joint_positions=self.q.copy(), joint_velocities=self.velocity.copy(),
                               gripper_width=self.width, timestamp=self.timestamp)

    def capture_rgbd(self):
        return SimpleNamespace(world_from_camera=np.eye(4))

    def set_joint_velocities(self, command):
        self.velocity = command.copy()

    def open_gripper(self):
        self.open_count += 1
        self.width = self.gripper_max_width

    def step(self):
        self.q += .05 * self.velocity
        self.timestamp += .05


def make_action(monkeypatch):
    k = Kinematics()
    interface = Interface(k)
    s = SimpleNamespace(interface=interface, kinematics=k, cube_size=.064,
        tool_from_cube=pose(-.2), settings=SimpleNamespace(dt=.05, clearance=.02, velocity_limit=.15),
        hardware=SimpleNamespace(control={"acceleration_limit": .5, "frame_max_age": .2}),
        controller_config=SimpleNamespace(frame_max_age_intervals=4),
        config=SequenceConfig(phase_timeout_s=40))
    target = pose(.032)
    target[0, 3] = .1
    dest = Destination(10, .08, target.tolist(), "table", .02, .02)
    action = PlaceAction(s, dest)
    monkeypatch.setattr(action.processor, "process_marker", lambda frame:
        SimpleNamespace(camera_from_marker=np.eye(4), timestamp=interface.timestamp))
    return action, s


def test_place_phases_release_once_then_retreat(monkeypatch):
    action, s = make_action(monkeypatch)
    phases = []
    for _ in range(1000):
        phases.append(action.phase)
        done = action.update()
        s.interface.step()
        if done:
            break
    assert done
    assert list(dict.fromkeys(phases)) == ["acquiring", "raising", "transferring", "descending", "releasing", "retreating"]
    assert s.interface.open_count == 1
    assert s.tool_from_cube is None
    assert len(s.kinematics.obstacles) == 2
    assert s.interface.q[0] == pytest.approx(.1, abs=.002)


def test_elevated_tag_controls_target_and_release_not_table(monkeypatch):
    action, s = make_action(monkeypatch)
    table = s.kinematics.obstacles[0]
    monkeypatch.setattr(action.processor, "process_marker", lambda frame:
        SimpleNamespace(camera_from_marker=pose(.05), timestamp=s.interface.timestamp))
    action.update()
    assert action.placement_plane.offset == pytest.approx(.05)
    assert action.targets[-1][2, 3] == pytest.approx(.282)
    assert s.kinematics.obstacles == (table,)
    for _ in range(1000):
        done = action.update()
        s.interface.step()
        if done:
            break
    assert done
    assert s.interface.open_count == 1
    assert table.offset == 0
    assert action.guard.released_pose[2, 3] > .08


def test_tag_plane_does_not_override_collision_table(monkeypatch):
    action, s = make_action(monkeypatch)
    monkeypatch.setattr(action.processor, "process_marker", lambda frame:
        SimpleNamespace(camera_from_marker=pose(-.02), timestamp=s.interface.timestamp))
    assert not action.update()
    assert "intersects configured collision table" in s.diagnostic
    assert action.phase == "acquiring"
    assert not action.targets
    assert s.interface.open_count == 0
    assert np.all(s.interface.velocity == 0)


def test_tilted_tag_plane_in_world_coordinates(monkeypatch):
    from scipy.spatial.transform import Rotation
    action, s = make_action(monkeypatch)
    world_camera = pose(.12)
    world_camera[:3, :3] = Rotation.from_euler("y", 12, degrees=True).as_matrix()
    camera_tag = pose(.03)
    monkeypatch.setattr(s.interface, "capture_rgbd", lambda:
        SimpleNamespace(world_from_camera=world_camera))
    monkeypatch.setattr(action.processor, "process_marker", lambda frame:
        SimpleNamespace(camera_from_marker=camera_tag, timestamp=s.interface.timestamp))
    action.update()
    world_tag = world_camera @ camera_tag
    np.testing.assert_allclose(action.placement_plane.normal, world_tag[:3, 2])
    assert action.placement_plane.offset == pytest.approx(world_tag[:3, 2] @ world_tag[:3, 3])
    np.testing.assert_allclose(action.targets[-1] @ s.tool_from_cube,
                               world_tag @ action.destination.marker_from_cube)


def test_blocked_place_never_releases(monkeypatch):
    action, s = make_action(monkeypatch)
    action.update()
    monkeypatch.setattr(s.kinematics, "distances", lambda q: np.array([-.01]))
    with pytest.raises(RuntimeError, match="path_blocked"):
        action.update()
    assert s.interface.open_count == 0


def test_destination_is_latched_for_entire_placement(monkeypatch):
    action, s = make_action(monkeypatch)
    calls = []
    tag = pose()
    def detect(frame):
        calls.append(s.interface.timestamp)
        return SimpleNamespace(camera_from_marker=tag.copy(), timestamp=s.interface.timestamp)
    monkeypatch.setattr(action.processor, "process_marker", detect)
    action.update()
    plane = action.placement_plane
    tag[0, 3] = .02
    for _ in range(1000):
        s.interface.step()
        done = action.update()
        if done:
            break
    assert done
    assert len(calls) == 1
    assert action.placement_plane is plane
    assert s.interface.q[0] == pytest.approx(.1, abs=.002)
    assert s.interface.open_count == 1


@pytest.mark.parametrize("stale", [False, True])
def test_invalid_acquisition_waits_without_resetting_timeout(monkeypatch, stale):
    action, s = make_action(monkeypatch)
    started = action.started
    s.interface.step()
    monkeypatch.setattr(action.processor, "process_marker", lambda frame:
        SimpleNamespace(camera_from_marker=pose(), timestamp=0.) if stale else None)
    action.update()
    np.testing.assert_array_equal(s.interface.velocity, np.zeros(6))
    assert action.targets == []
    assert action.started == started
    assert s.interface.open_count == 0


@pytest.mark.parametrize("gap,accepted", [(.0123, True), (.025, False), (-.005, False)])
def test_configurable_small_drop(monkeypatch, gap, accepted):
    from dataclasses import replace
    action, s = make_action(monkeypatch)
    target = action.destination.marker_from_cube.copy()
    target[2, 3] = s.cube_size / 2 + gap
    action.destination = replace(action.destination, marker_from_cube=target, max_drop_height_m=.02)
    if not accepted:
        assert not action.update()
        assert "drop-height" in s.diagnostic
        assert action.phase == "acquiring"
        assert s.interface.open_count == 0
        return
    for _ in range(1000):
        done = action.update()
        s.interface.step()
        if done:
            break
    assert done
    assert s.interface.open_count == 1
    assert action.guard.released_half_sizes[2] > s.cube_size / 2


def test_invalid_acquisition_then_valid_target(monkeypatch):
    action, s = make_action(monkeypatch)
    started = action.started
    tag = pose(-.1)
    monkeypatch.setattr(action.processor, "process_marker", lambda frame:
        SimpleNamespace(camera_from_marker=tag.copy(), timestamp=s.interface.timestamp))
    s.interface.step()
    assert not action.update()
    assert "waiting for first valid target" in s.diagnostic
    assert action.targets == []
    assert action.placement_plane is None
    assert action.started == started
    tag[:] = pose()
    tag[0, 3] = .01
    s.interface.step()
    action.update()
    assert s.diagnostic == ""
    assert action.targets[-1][0, 3] == pytest.approx(.11)
    assert s.interface.open_count == 0


def test_initial_invalid_target_keeps_timeout(monkeypatch):
    action, s = make_action(monkeypatch)
    monkeypatch.setattr(action.processor, "process_marker", lambda frame:
        SimpleNamespace(camera_from_marker=pose(-.1), timestamp=s.interface.timestamp))
    action.update()
    assert "waiting for first valid target" in s.diagnostic
    s.interface.timestamp += s.config.phase_timeout_s + 1
    with pytest.raises(TimeoutError):
        action.update()


@pytest.mark.parametrize("width", [0., .1005])
def test_grasp_loss_before_transfer_never_releases(monkeypatch, width):
    action, s = make_action(monkeypatch)
    s.interface.width = width
    with pytest.raises(RuntimeError, match="grasp_lost"):
        action.update()
    assert s.interface.open_count == 0


def test_joint_limit_failure_before_command(monkeypatch):
    action, s = make_action(monkeypatch)
    s.interface.q[2] = .24
    action.update()
    upper = np.full(6, 6.)
    upper[2] = .239
    s.kinematics.limits = (np.full(6, -6.), upper)
    with pytest.raises(RuntimeError, match="joint_limit"):
        action.update()
    assert np.all(s.interface.velocity == 0)


@pytest.mark.parametrize("current_z,expected_z", [(.4, .4), (.24, .252)])
def test_raise_only_to_required_approach_height(monkeypatch, current_z, expected_z):
    action, s = make_action(monkeypatch)
    s.interface.q[2] = current_z
    action.update()
    assert action.targets[0][2, 3] == pytest.approx(expected_z)
    assert action.targets[1][2, 3] == pytest.approx(expected_z)
    assert action.targets[0][2, 3] >= current_z


def test_raising_timeout_contains_last_progress(monkeypatch):
    action, s = make_action(monkeypatch)
    s.interface.q[2] = .24
    action.update()
    action.update()
    assert "current_z_mm=240.0" in s.diagnostic
    assert "target_z_mm=252.0" in s.diagnostic
    s.interface.timestamp += s.config.phase_timeout_s + 1
    with pytest.raises(TimeoutError, match="raising.*position_error_mm=.*commanded_max_joint_speed"):
        action.update()
    assert s.interface.open_count == 0


def test_transfer_timeout_override_is_bounded(monkeypatch):
    action, s = make_action(monkeypatch)
    s.config = SequenceConfig(phase_timeout_s=20, transfer_timeout_s=30)
    action.update()
    action.targets.pop(0)
    action._phase("transferring")
    s.interface.timestamp += 21
    assert not action.update()
    s.interface.timestamp += 10
    with pytest.raises(TimeoutError, match="transferring; limit=30 s"):
        action.update()


@pytest.mark.parametrize("phase", ["acquiring", "raising", "descending", "releasing", "retreating"])
def test_transfer_override_leaves_other_phase_limits_unchanged(monkeypatch, phase):
    action, s = make_action(monkeypatch)
    s.config = SequenceConfig(phase_timeout_s=20, transfer_timeout_s=30)
    action._phase(phase)
    s.interface.timestamp += 21
    with pytest.raises(TimeoutError, match=f"{phase}; limit=20 s"):
        action.update()


@pytest.mark.parametrize("value", [0, -1, True, float('nan'), float('inf')])
def test_invalid_transfer_timeout(value):
    with pytest.raises(ValueError, match="transfer_timeout_s"):
        SequenceConfig(transfer_timeout_s=value)


def test_missing_transfer_override_preserves_shared_timeout(monkeypatch):
    action, s = make_action(monkeypatch)
    assert s.config.transfer_timeout_s is None
    action._phase("transferring")
    s.interface.timestamp += s.config.phase_timeout_s + 1
    with pytest.raises(TimeoutError, match="transferring; limit=40 s"):
        action.update()


def test_release_timeout_does_not_repeat_open_command(monkeypatch):
    action, s = make_action(monkeypatch)
    def stuck():
        s.interface.open_count += 1
    monkeypatch.setattr(s.interface, "open_gripper", stuck)
    for _ in range(1000):
        action.update()
        s.interface.step()
        if action.phase == "releasing":
            break
    assert action.phase == "releasing"
    s.interface.timestamp += 41
    with pytest.raises(TimeoutError, match="releasing"):
        action.update()
    assert s.interface.open_count == 1


def test_missing_and_stale_destination_do_not_move(monkeypatch):
    action, s = make_action(monkeypatch)
    monkeypatch.setattr(action.processor, "process_marker", lambda frame: None)
    assert not action.update()
    assert action.phase == "acquiring"
    monkeypatch.setattr(action.processor, "process_marker", lambda frame:
        SimpleNamespace(camera_from_marker=np.eye(4), timestamp=0.))
    assert not action.update()
    assert np.all(s.interface.velocity == 0)
    s.interface.timestamp += 41
    with pytest.raises(TimeoutError, match="acquiring"):
        action.update()
    assert s.interface.open_count == 0
