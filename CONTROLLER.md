# Cube approach controller

Robot measurements and commands go through `RobotInterface` in `interface.py`.
The controller never reads live `RobotModel.data`. The responsibilities are:

| File | Responsibility |
| --- | --- |
| `interface.py` | Robot state, clock, RGBD acquisition, arm/gripper commands, resource cleanup |
| `robot_state.py` | Backend-independent `RobotState` and `RGBDFrame` data |
| `kinematics.py` | MuJoCo prediction workspace for Jacobians and collision distances |
| `controller.py` | Camera processing, PBVS, and optional MPC |
| `simulate.py` | Physics stepping, experiment scene configuration, and viewer windows |

The controller reads one `RobotState` per update. The prediction workspace
is synchronized from measured arm positions and gripper width; it does not
copy the simulator's state. Finger positions are modeled symmetrically from
the measured width and remain fixed during arm prediction. MuJoCo remains a
prediction library when controlling hardware; no live physics simulation is
required by the controller.

```python
from controller import ApproachController
from interface import MujocoInterface
from kinematics import MujocoKinematics
from robot_model import RobotModel

robot = RobotModel()
with MujocoInterface(robot) as interface:
    controller = ApproachController(interface, MujocoKinematics(robot))
    result = controller.update()  # Reads camera/state and sends arm velocity.
    # In simulation, simulate.py advances physics between updates.
```

For hardware, implement a `RobotInterface` subclass and use it in place of
`MujocoInterface`. Implement `read_state()`, `timestamp`, `capture_rgbd()`,
`set_joint_positions()`, `set_joint_velocities()`, `set_gripper_width()`,
`gripper_max_width`, and `close()`. Convenience properties and open/close
gripper commands are inherited. `read_state()` returns independent arrays
from one measured snapshot; its timestamp records measurement time.
`timestamp` returns the current monotonic clock. RGBD acquisition timestamps
must use that same clock, including device-to-host clock conversion where
needed. Both stale state and stale frames stop arm commands.

The hardware camera adapter supplies registered, rectified RGB/depth,
metric optical-axis depth, calibrated intrinsics, and `world_from_camera`
in the prediction model's world frame. Keep the prediction model's camera,
tool/gripper geometry, joint limits, and known obstacle map calibrated to
the real installation. Those are model/configuration inputs; controller and
perception code can remain unchanged. Unknown obstacles are still not mapped
automatically from perception. Automatic gripper closure is not part of the
current approach controller.

Select the controller in `controller_config.json`:

```json
{"mode": "pbvs_mpc", "tool_clearance": 0.18}
```

Use `"pure_pbvs"` for direct PBVS joint-velocity commands with basic speed
and joint-angle bounds, or `"pbvs_mpc"` for MPC reference tracking with
acceleration and obstacle-clearance constraints. Pure PBVS does not run MPC
or check obstacle/self-collision clearance; its reported clearance is `nan`
(not evaluated). Both modes stop on lost/stale tags and at the approach goal.
The default is `pbvs_mpc`. Restart after changing the file. Use
`--controller-config path/to/controller.json` to choose another file. The
selection is used whenever `--servo` enables control, in both viewer and
headless runs. Python callers can pass `controller_config=ControllerConfig(
mode="pure_pbvs")` to `Simulation` or `ApproachController`.

`tool_clearance` sets the final position of the tool flange (`attachment_site`)
relative to the detected tag: centered over the tag, along its outward normal,
at the configured distance in meters. It is not the fingertip-to-cube gap.
The default is 0.18 m; the minimum is 0.16 m to keep the full tag visible to
the offset wrist camera. Both controller modes use this target, including
hardware callers. `--tool-clearance` temporarily overrides the controller
file's value; explicit Python `tool_clearance=` arguments do the same.

Edit `simulation_config.json` for each simulation experiment. Its first
setting, `cube_position`, sets the cube center `[x, y, z]` in world meters.
For the upright 10 cm cube, `z = 0.05` places its bottom on the floor.
`cube_quaternion` sets world orientation in `[w, x, y, z]` order; the loader
normalizes it. The cube's tag
moves with it. Restart the simulation after editing the file.

The CLI automatically loads this file independently of the current working
directory. Use `--config path/to/experiment.json` to select another file.
These settings are applied
only to the simulation's model instance, without modifying scene.xml or the
hardware interface. Python callers can use
`Simulation(config=SimulationConfig.load("experiment.json"))`; `Simulation()`
uses built-in defaults so automated tests do not depend on your experiment file.

Run both the robot overview and annotated camera view:

```powershell
venv/Scripts/python.exe simulate.py --servo
```

Run without windows (stops early when the target is reached):

```powershell
venv/Scripts/python.exe simulate.py --headless --servo --steps 6000
```

The default target puts `attachment_site` directly above the detected tag,
18 cm from the tagged face. `--tool-clearance` changes this distance; its
minimum is 16 cm because the wrist camera's 6 cm lateral offset must keep
the whole tag visible. There is no grasping or lifting yet.

`CameraProcessor` estimates marker pose from RGB and registered depth.
`PBVS` computes a camera-frame pose error, proportional camera twist, and
damped Jacobian inverse to obtain a joint-velocity reference in rad/s.
`JointSpaceMPC` optimizes eight future joint-velocity commands at 50 ms
intervals and applies only the first command before solving again.

The kinematic prediction is `q[k+1] = q[k] + dt*u[k]`. The objective is
`sum(w_tracking*||u[k]-u_pbvs||^2 + w_smoothing*||u[k]-u[k-1]||^2)`.
Hard constraints impose joint-angle bounds, maximum joint velocity and
acceleration, and at least 2 cm signed separation between collision geoms.
An extra 3 mm planning buffer helps absorb linearization/tracking errors.
Geometry is linearized at horizon midpoints/endpoints, then checked again
with nonlinear distances at four samples per interval. Invalid plans are
relinearized or rejected, and failed optimization or lost/stale tags command
zero velocity. `ControlResult` exposes the reference, command, errors,
status, observation, and minimum predicted clearance.

Known scene collision geometry supplies the obstacle map, including the
floor, cube, any added collision-enabled environment geoms, and nonadjacent
robot links. Same-body/adjacent links and the invariant shoulder-floor
mounting gap are excluded. Visual-only meshes and the tag are excluded.
Obstacle positions are model inputs; perception supplies the target pose,
not a complete map of unknown obstacles.

`RobotInterface.set_joint_velocities()` is the portable command API.
`MujocoInterface` switches the existing affine actuators into velocity
servos and supplies bias-force compensation before every simulation step.
Position commands restore their original gains. A hardware adapter must
provide its own velocity transport, calibrated model, and obstacle map.

This is local kinematic MPC: it may stop at a blocked path rather than find
a global detour. Sampled predicted clearance does not guarantee continuous
collision avoidance under arbitrary tracking errors or moving obstacles.
The simulator may run slower than real time when rendering/optimization
exceeds the control period. MPC parameters are in `MPCConfig`.

Checks: `venv/Scripts/python.exe -m unittest discover -s tests -v`.

The tool flange carries a generic parallel two-finger gripper with a maximum
inner opening of 120 mm. The fingers have collision geometry, friction, and
separate position actuators with 30 N force limits. It starts fully open.
The portable interface controls the opening in meters, independently of the
six arm joint commands:

```python
from simulate import Simulation

simulation = Simulation()
simulation.interface.close_gripper()
simulation.step(500)  # Advance physics to move the fingers.
print(simulation.interface.gripper_width)
simulation.interface.open_gripper()
simulation.step(500)
simulation.interface.set_gripper_width(0.10)
```

`RobotModel.grasp_pose` exposes the world pose of `grasp_site`, between the
fingertips, with +Z along the approach direction. The approach controller
continues to target the flange's `attachment_site`; it does not yet execute
a grasp. MPC includes the gripper's collision geometry against obstacles
and nonadjacent robot links, while finger opening remains separately commanded.
The scene cube remains fixed, so lifting requires a later scene change.
