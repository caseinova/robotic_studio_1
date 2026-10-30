# Pick and Place Sessions

The existing `run.py --task pick` is unchanged. The new API keeps the same
camera, arm, and gripper connections open across successive actions:

```python
from controller_config import ControllerConfig
from estimator import EKFConfig
from hardware_config import HardwareConfig
from robot_session import RobotSession

robot = RobotSession(HardwareConfig.load(), ControllerConfig.load(), EKFConfig.load())
robot.validate_destination("drop_zone_b")  # Offline check BEFORE picking up.
with robot:
    robot.servo_to_pick_up()
    robot.move_to_destination_and_drop("drop_zone_b")
```

Nonblocking equivalents are `start_servo_to_pick_up()` and
`start_move_to_destination_and_drop(name)`. Each returns a Future:
`done()` polls, `result()` waits and raises on failure. Only one action may
run at once; the hardware owner thread advances each action's `update()`.
Use `close()` or leave the context to cancel. Cancelling a Future does not
stop an already running robot action. An exception or timeout aborts the
session; it never automatically opens the gripper as error recovery.

Between actions the session sends zero joint velocity and checks state and
gripper feedback. It expires after `idle_timeout_s`; it is not an indefinite
payload-holding facility. The existing robot command watchdog, freshness,
speed, joint-limit and cycle-time checks remain active. Blocked I/O does not
receive an independent background motion heartbeat.

## Destination Configuration

`config/sequence_config.json` has an intentionally empty `destinations` map.
Add measured destinations. This is an EXAMPLE only for a 64 mm cube on a flat
surface, with an 80 mm tag ID 10, cube center 150 mm along marker X:

```json
"drop_zone_b": {
  "marker_id": 10,
  "marker_size_m": 0.08,
  "marker_from_cube": [
    [1, 0, 0, 0.150],
    [0, 1, 0, 0],
    [0, 0, 1, 0.032],
    [0, 0, 0, 1]
  ],
  "collision_support_plane": "table",
  "approach_height_m": 0.10,
  "retreat_height_m": 0.10
}
```

Transforms map cube coordinates into marker coordinates. The translation is
the desired cube CENTER, not the flange or finger tips. The rotation defines
the deposited cube's orientation. IDs use DICT_4X4_50 and must not overlap
cube/tool IDs. The tag is a destination, not a camera-calibration reference.
Existing camera-to-base and tool calibration remain required.

Before hardware placement, configure measured tool-body and at least two
finger capsules (`frame: "tool"`, fingers `contact_group: "finger"`) in
hardware collision settings, plus the named horizontal table plane and all
other relevant obstacles. Opening-dependent finger capsules must cover their
full travel. Bare-arm capsules alone are rejected. These checks cannot verify
that your measurements actually enclose the physical robot or environment.

```bash
python run_sequence.py drop_zone_b --validate
python run_sequence.py drop_zone_b
```

Both launchers resolve hardware, controller and estimator paths from the same
`config/runtime_config.json`. The sequence runner also accepts `--config` and
`--hardware-config`, like `run.py`, and prints the resolved paths at startup.
Its preview obeys the runtime camera/headless flags. Metric-size rejections in
`Tracking:` reports include the cube face IDs, measured edge lengths in mm,
expected marker size, and the unchanged 10% tolerance.

The sequence runner shows a live camera window by default, including during
latched pickup and placement phases. Use `--headless` to disable it. Q, Escape,
or closing the window cancels the session without automatically releasing the
cube. Rendering runs on the main thread; device access remains on the session
thread. Display-only capture still counts against the control cycle budget.
The display shows the live camera image and phase label only. It performs no
additional ArUco detection or marker annotation; control-side detection is unchanged.
The Python API opts in with `RobotSession(..., camera_view=True)` and a
main-thread `SequencePreview` consumer. `--validate` creates no window/devices.

## Motion and Limitations

`transfer_timeout_s` overrides `phase_timeout_s` only for the horizontal/rotating
transfer phase. The installed config allows 30 s for transfer and 20 s for
other phases. Omit it or use null to retain the shared phase timeout.
The overall action and hardware run budgets still apply; speed limits,
collision checks, freshness checks and transport shutdown are unchanged.

Pickup reuses approach, dwell, descend, close, and lift. At closing it records
`tool_from_cube` from FK and the latched visual cube pose. This assumes no
unmeasured cube motion during gripping; width is only a grasp heuristic.

After pickup, placement waits for one valid, fresh destination observation.
Missing, stale, or geometrically invalid candidates produce diagnostics and
are retried with zero velocity under the existing phase/action timeouts.
The accepted pose and tag plane are then fixed throughout raising, transfer,
descent, release and retreat. There is no periodic destination update or EKF.
Travel tool height is `max(current_tool_z, destination_tool_z + approach_height_m)`:
raising never lowers the arm and does not add another lift when already high
enough. This height is not proof of an obstacle-free route; path checks still
apply. Placement diagnostics report remaining pose error, tool/target Z and
joint speeds; phase timeouts include the last motion sample.
Keep the tag, camera and placement surface stationary after acquisition.
The live display does not update the motion target. Actual collision
and release checks still apply. The old destination_update_interval_cycles
configuration key has been removed.

This is local Cartesian control, not a global path planner. Current and
intermediate predicted joint poses are checked against arm/tool geometry,
an oriented cube payload, and obstacles. The cube may contact only its named
support plane; finger/payload contact is intentional. Arm/table clearances
are not relaxed. Cube-versus-box obstacle checks are conservative and may
reject feasible paths. Sampled checks are not continuous swept-volume or
stopping-distance certification. A blocked path stops rather than rerouting.

The detected destination tag's center and +Z normal define the local placement
plane, latched from the same fresh observation as the target. This plane is
not inserted into the workspace collision map. `collision_support_plane: "table"`
names the separate configured collision table, not the placement-height
reference. Do not remove that geometry: arm, gripper and carried-cube collision
checks continue to use it. A target below that table still fails, even if it
is valid relative to the tag; resolve conflicting calibration/geometry first.

Release allows a vertical world-Z fall of up to the destination's
`max_drop_height_m` before the estimated cube contacts the tag plane. The tag
must face upward. For a flat tag and an aligned 64 mm cube, marker-relative
Z=0.032 means resting on the tag plane, and Z=0.040 means an 8 mm drop,
regardless of the tag's world height. Tilt is accounted for using the oriented
cube's extent and the plane normal; gravity remains along world -Z.
The default remains 0.003 m; the current
`drop_zone_b` explicitly allows 0.020 m (20 mm). This is a configurable release
allowance, not a physical safety rating. Below-table targets remain rejected,
and robot/table collision checks remain unchanged. The post-release obstacle
encloses the vertical fall from release to support; bounce, rotation, and
sliding are not modeled, so verify the actual final cube position before reuse.
Opening the gripper is verified by width only, not independent visual or
force confirmation that the cube detached. The deposited cube is added to
the obstacle map after retreat. Success of these software checks is not
hardware safety certification; validate geometry and motions under supervision.
