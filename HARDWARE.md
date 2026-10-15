# UR3 calculation backend

`hardware_kinematics.py` provides `UR3Kinematics`, implementing the same
`Kinematics` contract as `MujocoKinematics`. It needs NumPy, not MuJoCo,
`robot_model.py`, XML, rendering, or a running physics simulation.

`interface.py` remains responsible for sensor acquisition, timestamps, and
robot/gripper communication. Geometry calculations belong in the kinematics
backend so measured robot states and hypothetical MPC states stay separate.
No hardware transport or hardware-mode launcher is implemented by this change.

## Calculations

| Method | Output |
| --- | --- |
| `synchronize(state)` | Sets measured joint angles and gripper opening in the calculation workspace |
| `forward_kinematics(q)` | World-from-tool transform without changing synchronized state |
| `tool_pose()` | World-from-tool transform at the workspace joint angles |
| `grasp_pose()` | World-from-grasp-center transform, using the configured tool-to-grasp offset |
| `tool_jacobian(q, local=False)` | 6×6 geometric Jacobian, linear rows then angular rows, in world axes |
| `tool_jacobian(q, local=True)` | Same Jacobian expressed in tool axes, as used by PBVS |
| `distances(q, gradients=True)` | Pair clearances and their derivatives with respect to the six joint angles |
| `pickup_collision_indices()` | Constraint indices excluding intended finger/target contacts only |

Standard DH transforms use `Rz(q + offset) Tz(d) Tx(a) Rx(alpha)`.
`UR3DH()` contains the nominal **UR3 CB-series**, not UR3e, parameters published
by [Universal Robots](https://www.universal-robots.com/articles/ur/application-installation/dh-parameters-for-calculations-of-kinematics-and-dynamics/).
Factory-calibrated values can replace all four arrays (`a`, `d`, `alpha`,
`offsets`). Joint order matches `RobotInterface`: shoulder pan, shoulder lift,
elbow, wrist 1, wrist 2, wrist 3. All angles are radians and lengths meters.

## Calibration inputs

The constructor requires explicit joint position limits and these transforms:

- `world_from_base`: installation of the DH base in the common world frame.
- `flange_from_tool`: selected tool frame relative to DH frame 6. This also
  resolves any difference between the hardware SDK's flange convention and DH.
- `tool_from_grasp`: grasp-center frame relative to the selected tool frame.
- `tool_from_marker`: measured tool-tag frame relative to that tool frame,
  required when using the tool tag. It may be omitted for FK-only operation.

As elsewhere in the project, `A_from_B` maps coordinates in B into A. Camera
frames supplied by the hardware interface must use this same world frame.
No camera calibration, tag mounting, link radius, or UR5e gripper dimension
is silently substituted for hardware calibration. Missing tag calibration
raises an explicit error if a detected tool tag is used.

The following factory accepts a calibration dictionary supplied by the
application; it does not read a config file or connect to hardware:

```python
from controller import ApproachController
from controller_config import ControllerConfig
from estimator import EKFConfig
from hardware_kinematics import UR3Kinematics

def make_controller(interface, calibration, link_capsules, obstacles):
    kinematics = UR3Kinematics(
        **calibration,
        link_capsules=link_capsules,
        obstacles=obstacles,
    )
    return ApproachController(
        interface,
        kinematics,
        controller_config=ControllerConfig(mode="pbvs_mpc"),
        estimator_config=EKFConfig.load(),
    )
```

`calibration` must supply the required constructor inputs above. It can also
override `dh`, `marker_id`, and `marker_size`. The controller's existing
`MPCConfig` supplies operating velocity/acceleration limits and clearance;
joint position limits are supplied to `UR3Kinematics`. Robot dynamics and
torque calculations are not part of this kinematic MPC.

## Collision geometry

Configure physical link/gripper envelopes as `LinkCapsule` objects, including
the base if it should participate in self-collision checks. Each capsule's
endpoints are expressed in DH frame 0..6 or the calibrated `"tool"` frame.
Do not assume a segment between DH origins encloses a physical link.

Environment shapes are `SphereObstacle`, oriented `BoxObstacle`, and
`PlaneObstacle`. A plane's allowed side is `normal · point >= offset`.
Per-obstacle `exclude_links` can describe deliberate base/table mounting.
Robot self pairs skip identical/adjacent attachment frames; additional
`excluded_self_pairs` must be explicit named pairs.

Gripper capsules can use `opening_axis` to shift by measured inner opening.
For example, `(0, .5, 0)` and `(0, -.5, 0)` describe symmetric fingers along
tool Y. Opening remains fixed across arm-only MPC predictions, matching the
simulation backend.

Distances subtract capsule radii, so positive means separated and negative
means collision. Sphere, plane, and capsule clearances are analytic. Box
clearance uses exact segment-to-solid-box distance outside the box. When a
capsule centerline intersects a box, the returned `-radius` flags collision;
it is not an estimate of penetration depth. Gradients are central finite
differences and may be nonsmooth when closest features change.

MPC/pickup raises an error if no link capsules are configured, rather than
treating missing geometry as an obstacle-free robot. Supplied shapes define
the checked coverage; the backend cannot verify that they enclose the actual
robot. No obstacles are automatically extracted from RGB-D segmentation.

The pickup controller allows a link marked `contact_group="finger"` to
touch an obstacle marked `contact_group="grasp_target"`. Palm and arm
constraints remain active. Normal MPC keeps all configured pairs, including
finger/target pairs, preserving existing approach behavior.

Use `set_obstacles()` between control updates to replace the supplied map.
Reset the MPC plan after changing it; reconstruct an active pickup controller
if pair membership/order changes because it caches its allowed-contact indices.

These calculations and tests are offline. Individual-arm calibration,
physical collision envelopes, and the hardware transport are still needed
before using this backend on the real UR3. Finite-difference collision
gradients also need timing checks against the chosen control interval.
