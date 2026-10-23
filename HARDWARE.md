# UR arm hardware backend

## Supplied UR3e API

The checked-in configuration selects `robot.driver = "ur3_rg2_rosbridge"`.
`hardware_rosbridge.RosbridgeArm` uses `RealUR3e` from the sibling
`UR3_RG2_rosbridge` repository. `robot.api_path` resolves relative to the
hardware configuration file; the default is `../UR3_RG2_rosbridge`.
Keep both repositories together, including the integration changes to the
supplied backend (relative package import and cleanup on connection failure).
Validation checks the path without importing the RTDE SDK or opening devices.

The 20 Hz controller queues velocity commands to a 100 Hz worker. The worker
uses the API's paced `set_joint_velocity()` and accounts for time already
spent in that call, so it does not add a second full control period.
Position holds are converted to bounded proportional joint velocities.
The same configured velocity and acceleration limits are passed to both
layers, avoiding an unexpected second velocity cap from API defaults.

A read-only API instance supplies feedback. A separate command instance is
created inside the worker on the first queued command and closed by that
worker. Timestamp/safety checks and SDK watchdog operations use the API's
`rtde_r`/`rtde_c` handles because its public interface does not expose them.
Coherent snapshots, command expiry, braking bounds, and cleanup remain active.
`observe` never enables arm control. Camera and gripper remain independent
RealSense and OnRobot adapters; the supplied API controls only the arm.

The checked-in DH values now describe nominal UR3e geometry matching the
supplied model. `UR3DH()` still defaults to CB-series UR3 for explicit legacy
callers, and `robot.driver = "ur_rtde"` retains the original direct transport.
Robot controller `base` and model `base_link` differ by a pi rotation about Z.
The supplied model's `tool0` matches nominal DH frame 6; the physical active
TCP, gripper, camera, and tag mounting still require installation calibration.
No identity calibration or joint operating limits are inferred automatically.

`hardware_kinematics.py` provides `UR3Kinematics`, implementing the same
`Kinematics` contract as `MujocoKinematics`. It needs NumPy, not MuJoCo,
`robot_model.py`, XML, rendering, or a running physics simulation.

`interface.py` remains responsible for sensor acquisition, timestamps, and
robot/gripper communication. Geometry calculations belong in the kinematics
backend so measured robot states and hypothetical MPC states stay separate.
`run.py` selects simulation or hardware from `runtime_config.json`. Hardware
uses `HardwareInterface`, the supplied UR3e API (or direct UR RTDE), and configurable OnRobot
RG2/RG6 Compute Box Modbus TCP. The camera is a RealSense D435i, selected
through `hardware_realsense:create_camera`. Any component can use a custom adapter.

## Configure and launch

1. Keep `runtime_config.json` as the shared entry point. `backend` selects
   `simulation` or `hardware`; `task` selects `observe`, `servo`, or `pick`.
   Controller selection (`pure_pbvs` / `pbvs_mpc`) stays in `controller_config.json`.
2. Fill the `null` values in `hardware_config.json`, including joint limits,
   robot/Compute Box addresses, camera intrinsics and installation transforms.
   Matrices named `A_from_B` map B coordinates into A. Do not substitute identity
   matrices unless those frames really coincide.
3. The D435i adapter is already selected in `camera.factory`. Confirm serial,
   resolution, distortion coefficients, and `depth_scale` (meters per raw unit).
   Obtain device intrinsics with the camera-only command below, or enter your
   measured OpenCV calibration. Camera-to-robot extrinsics must be calibrated separately.
4. Verify RG2 vs RG6, unit ID, gripping force and actual maximum finger opening.
   RG2 is the template selection, not automatic hardware identification.
5. For MPC or pickup, populate physical link/gripper envelopes and known
   obstacles under `collision`. No guessed UR3 link radii are installed.

```powershell
# Existing simulation: no hardware dependencies or connections.
py run.py --validate
py run.py --headless --steps 25

# Configuration-only preflight; no device constructors or commands.
py run.py --backend hardware --validate

# After configuration: read camera/encoders, without commanding movement.
py run.py --backend hardware --task observe

# Or set backend=hardware and task=servo/pick in runtime_config.json:
py run.py
```

The hardware preflight intentionally fails on the checked-in placeholders.
File paths in the runtime JSON are relative to that file, not the working
directory. `--config`, `--hardware-config`, `--backend`, `--task`, `--steps`,
and `--headless` can override selection. `simulate.py` remains available.

Install `requirements-hardware.txt` for the hardware path (it does not require
MuJoCo). No SDK dependencies or network connections are needed for `--validate`.
UR RTDE requires compatible CB-series robot software/setup; its normal control
script is uploaded on the first command. It does not reuse an unrelated URCap
program. Robot power/mode, payload, and installation settings stay on the robot.

## RealSense D435i setup

Install the hardware dependencies, then inspect the connected camera:

```powershell
py -m pip install -r requirements-hardware.txt
py hardware_realsense.py
```

The inspection command opens only the camera and prints its factory intrinsics,
distortion coefficients, and depth scale. Copy these into `hardware_config.json`;
it does not write calibration files or connect to the robot or gripper. If the
SDK distortion model cannot be represented by the OpenCV model, the report leaves
`distortion` null: provide an OpenCV calibration instead. Calibrate
`world_from_camera` separately for the installed camera position.

### Fixed-camera and tool-tag calibration

RGB exposure controls are optional entries in `camera.options`:

```json
"rgb_auto_exposure": null,
"rgb_exposure": null,
"rgb_gain": null,
"rgb_brightness": null
```

Null or omitted values leave that control unchanged. Set `rgb_auto_exposure`
to `true` for automatic exposure, or `false` for manual operation. Providing
`rgb_exposure` or `rgb_gain` also disables auto-exposure; combining either with
explicit `rgb_auto_exposure: true` is rejected. Brightness may be adjusted with
either mode. Values use the RGB sensor's native SDK units, not normalized
percentages; the adapter checks the connected device's range and step and
reports them for invalid values. Do not assume RGB exposure units match depth
exposure units. Controls apply only to the color sensor, before warmup, whenever
the camera is opened. Restart collection after changing configuration.

For washed-out marker edges, lower manual exposure first, then gain as needed.
No manual exposure value is preselected because suitable values depend on
lighting. These settings do not alter intrinsics or calibration transforms.

From the workspace root, close RealSense Viewer and run:

```bash
venv/bin/python robotic_studio_1/calibrate_camera.py collect --robot-ip ROBOT_IP
venv/bin/python robotic_studio_1/calibrate_camera.py solve
venv/bin/python robotic_studio_1/calibrate_camera.py solve --write-config
```

The collector only reads the robot; it never sends motion or gripper commands.
Keep the camera stationary and the configured tool tag rigidly mounted. Verify
the printed black square matches `perception.tool_marker_size` (currently 40 mm).
Position the arm manually using the robot's normal controls, stop, and press
Space in the camera window. Collect 15-25 distinct visible poses, varying both
position and tilt around at least two axes, then press Q. At least 12 poses are
required. Moving, stale, unstable, or near-duplicate captures are rejected.
Each accepted capture is saved immediately. Existing sample files are never
overwritten: use `collect --samples another_session.json` for a new session,
and pass the same `--samples` option to `solve`.

The solver estimates `world_from_camera` and `tool_from_marker` jointly using
the configured DH kinematics. With identity `world_from_base`, world is the
controller base. If `flange_from_tool` is unset, this workflow explicitly defines
the tool frame as the nominal DH flange and saves identity for that transform.
An existing tool transform is preserved. This is a frame definition, not a
measurement of the gripper TCP; `tool_from_grasp` still needs separate setup.

Solving is offline and writes `camera_calibration_result.json`; it changes no
configuration unless `--write-config` is supplied. Every fourth pose is held
out for validation before fitting all poses. Both fits must satisfy position
RMS <= 8 mm, position maximum <= 15 mm, rotation RMS <= 3 degrees, and rotation
maximum <= 5 degrees. Applying also checks that camera, tag, and kinematics
settings have not changed since collection, and backs up the original config.
These are fit-quality checks, not a hardware safety certification. Check the
result against independently measured positions before enabling motion; nominal
DH parameters and camera intrinsic errors can introduce systematic errors.

The adapter defaults to 640 by 480 RGB and aligned depth at 30 Hz. Set
`camera.options.serial` when more than one RealSense is connected. Acquisition
uses global capture timestamps and rejects duplicate, incomplete, or excessively
unsynchronized frames. The D435i IMU is not used by the pose estimator.

## Device adapter contracts

An adapter factory is called as `factory(config=component_settings, clock=clock)`.
`clock()` returns the session's monotonic time. Factories must clean up their
own partially constructed resources if construction raises. Successfully
created adapters are owned and closed by the runtime.

For additional cameras, the adapter contract is:

```python
from hardware_devices import CameraSample

class CameraAdapter:
    def capture(self):
        # Read from the selected SDK with bounded blocking and map the SDK's
        # acquisition timestamp onto the supplied monotonic clock.
        return CameraSample(acquisition_time, rgb_uint8, aligned_raw_depth)

    def close(self):
        # Release the SDK pipeline/device.
        ...

def create_camera(config, clock):
    # Construct CameraAdapter using your camera SDK and config["options"].
    ...
```

The D435i implementation already supplies this contract. Other camera factories
must return an adapter object. RGB must be RGB order, with depth already registered
to RGB optics and representing optical-axis Z. Both arrays must match configured
resolution. `HardwareInterface` rectifies both using the configured OpenCV
distortion model (linear RGB, nearest-neighbor depth), scales depth to meters,
and attaches the calibrated `world_from_camera` and K. For an already rectified
SDK stream, supply its rectified K and zero distortion to avoid double correction.
`depth_scale=1` supports depth already expressed in meters.

Never stamp an old buffered frame with its delivery time. Map device acquisition
time into `clock()` once, retaining age/latency. Repeated or stale camera samples
are rejected. Camera acquisition must have a bounded SDK timeout so exit and
error handling can complete. Other camera adapters must meet this same contract.

Custom robot adapters implement `read_joint_state()` returning
`JointSample(timestamp, positions, velocities)`, `command_velocity(qd)`,
`command_position(q)`, `stop()`, and `close()`. Custom grippers implement
`read_width()` returning `WidthSample(timestamp, width_m)`,
`command_width(width_m)`, and `close()`. All samples use the same acquisition
clock. Custom robot adapters must enforce command expiry/stop behavior like
the built-in worker; `close()` must not open a gripper carrying an object.

## Built-in transport and timing

The [UR RTDE adapter](https://sdurobotics.gitlab.io/ur_rtde/pages/reference/api.html)
reads joint positions, velocities, device timestamps and safety mode. It checks
packet timestamps around the reads to avoid mixing packet generations. The
robot clock maps to host monotonic time with a fixed anchor and one conservative
sample period; repeated packets retain their old timestamp rather than becoming
fresh. An implausible mapping or stale stream stops the session.

The UR3 CB-series worker runs at configurable frequency up to 125 Hz and owns
all control SDK calls. Vision/MPC updates its pending command independently.
The worker checks command/state age, operating velocity limits and joint-limit
braking bounds, and uses the configured acceleration with `speedJ`. Position
targets use bounded proportional velocity tracking. An on-robot SDK watchdog
also stops control if communication stalls; it is kicked only for valid,
unexpired commands. A transport/watchdog failure is latched until a new session.

The outer loop uses `control.dt`, rejects slow cycles via `max_cycle_time`,
and stops after `max_run_seconds` or the configured step count. Freshness checks
apply again just before a nonzero command, so a slow optimizer cannot send an
old image/state command. Completion, Ctrl+C, errors and timeouts close all
devices and stop owned arm control. Gripper cleanup disconnects without opening.

The [OnRobot RG2/RG6 register map](https://onrobot.com/sites/default/files/robot-comp-download/User_Manual_for_DOOSAN_v1.05_EN.pdf)
is used over Compute Box Ethernet (default port 502, configurable unit ID).
Force is converted to tenths of Newtons and width to tenths of millimeters.
One FC16 transaction writes force, width and control. Width feedback comes
from the gripper, not the requested width. Busy and safety/error flags reject
new commands explicitly. No gripper power cycle or auto-homing is performed.
`use_device_fingertip_offset=true` uses the offset-aware width/control registers;
calibrate the actual fingers in the device and set `max_width_m` accordingly.
RG2-FT/2FG7 and non-Compute-Box connections need a different custom adapter.

The current grasp-center transform is fixed and the gripper collision opening
is a symmetric translation approximation. RG2 linkage-dependent TCP depth
compensation is not modeled. Calibrate at the intended grasp opening or extend
the grasp/collision model for that motion before precision pickup.

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
No camera calibration, tag mounting, link radius, or simulated gripper dimension
is silently substituted for hardware calibration. Missing tag calibration
raises an explicit error if a detected tool tag is used.

The JSON loader builds these calculation objects without opening devices:

```python
from hardware_config import HardwareConfig
from controller_config import ControllerConfig

hardware = HardwareConfig.load("hardware_config.json")
hardware.validate_ready(ControllerConfig.load(), task="servo")
kinematics = hardware.build_kinematics()
processor = hardware.build_processor()
```

The JSON also supplies calibrated DH values, tool/cube marker IDs and sizes,
and every `cube_from_markers` transform. The template's six cube transforms
describe the nominal simulated 10 cm cube and must match the real labels.
PBVS targets the configured `reference_marker_id`, including after EKF filtering.
Tool and cube IDs must be distinct. Operating velocity/acceleration limits and
timing live in `control`; the remaining MPC settings live in `mpc`. Joint
position limits live in `robot`. Robot dynamics and torque calculations are
not part of this kinematic MPC. Simulation noise config is never injected
into real sensor samples.

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

Tests use offline fake devices and SDKs, not a connected UR3, gripper, or D435i.
Individual-arm calibration, physical collision envelopes, and camera-to-robot
calibration are still required. Finite-difference collision gradients also need
timing checks against the chosen control interval. Pickup retains the
existing latched-target sequence; this launcher does not implement moving-cube
pickup or the proposed continuous-tracking grasp behavior.
