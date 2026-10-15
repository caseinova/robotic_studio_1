# Cube approach controller

The cube carries six ArUco `DICT_4X4_50` markers: ID 0 on top, ID 2 on
+Y, ID 3 on +X, ID 4 on -Y, ID 5 on -X, and ID 6 on the bottom. The tool
keeps its single ID 1 marker. All markers have an 80 mm black square and
5 mm white margins. Bottom-face visibility requires lifting or turning the cube.

`TaggedCubeProcessor` in `camera_processing.py` converts each validated RGB-D
marker measurement through its known rigid mounting transform to the cube
center. It rejects faces inconsistent with the best measurement and combines
the remaining positions and rotations, weighting image area and reprojection
quality. One visible cube marker is sufficient. The camera overlay lists the
face IDs used; `CubeObservation.marker_observations` exposes their measurements.

The optional EKF receives one fused cube pose per exposure. Its measurement
covariance stays conservative because tags share camera calibration and depth
errors. Multiple visible faces can reduce random pose jitter without applying
the same camera timestamp repeatedly. `camera_from_marker` retains the canonical
top reference frame for PBVS, even when only a side marker is visible, so the
approach target does not jump between cube faces. For hardware, calibrate the
face mounting transforms in `TaggedCubeProcessor` to the physical labels.

Robot measurements and commands go through `RobotInterface` in `interface.py`.
The controller never reads live `RobotModel.data`. The responsibilities are:

| File | Responsibility |
| --- | --- |
| `interface.py` | Robot state, clock, RGBD acquisition, arm/gripper commands, resource cleanup |
| `robot_state.py` | Backend-independent `RobotState` and `RGBDFrame` data |
| `kinematics.py` | Shared calculation contract and MuJoCo prediction workspace |
| `hardware_kinematics.py` | UR3 FK, Jacobians, and geometric constraints without MuJoCo |
| `controller.py` | Camera processing, PBVS, and optional MPC |
| `simulate.py` | Physics stepping, experiment scene configuration, and viewer windows |

The UR3 backend and its calibration inputs are described in [HARDWARE.md](HARDWARE.md).
Both the approach and pickup controllers accept either calculation backend.

The controller reads one `RobotState` per update. The prediction workspace
is synchronized from measured arm positions and gripper width; it does not
copy the simulator's state. Finger positions are modeled symmetrically from
the measured width and remain fixed during arm prediction. Hardware can use
`UR3Kinematics` without MuJoCo; simulation uses `MujocoKinematics`.

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
The default is 0.18 m; the retained minimum is 0.16 m to provide space for the
gripper before pickup. Both controller modes use this target, including
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
minimum is 16 cm to provide gripper approach space. Use `--pick` for the
separate descent, grasp, and lift sequence after the approach.

`CameraProcessor` estimates marker pose from RGB and registered depth.
The tag plane fit uses a robust estimate of depth scatter to set its inlier
threshold (at least `depth_tolerance`, with adaptation capped at 10% of marker
size). This accepts moderate Gaussian depth noise without discarding a visible
tag solely because individual pixels differ by more than 3 mm. Insufficient
depth, excessive scatter, inconsistent metric tag size, and large reprojection
errors still reject the observation.
The plane normal is fitted using `1/Z = a*(X/Z) + b*(Y/Z) + c` on known pixel
rays. This accounts for depth noise moving points along their rays and avoids
the normal bias from treating Cartesian X/Y/Z scatter as isotropic. Both cube
and tool tags use this fit and metric residual checks.
If the primary contour detector misses the target, adaptive thresholding with
subpixel refinement retries detection. Both paths retain the same RGB-D pose
validation. Simulation disables offscreen multisampling to avoid depth-edge
artifacts in the distant fixed-camera view.
`PBVS` computes a tool-frame pose error, proportional tool twist, and
damped tool Jacobian inverse to obtain a joint-velocity reference in rad/s.
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
The scene cube remains fixed in approach-only runs. Pickup runs use a movable
version of the scene so the gripper can lift it through physical contact.

## Fixed-camera calibration scene

```powershell
py simulate.py --camera
py simulate.py --camera --camera-name external_camera
```

The external RGB-D camera is fixed at world `[-0.074, 1.05, 0.95]` meters, on
the workspace side (+Y), above and centered on the default cube's X coordinate.
This front view keeps the two open fingers to either side of the cube tag. It looks back
toward the gripper/cube workspace. The end-effector carries ArUco DICT_4X4_50
ID 1, with an 80 mm black square and a 5 mm white margin. Its raised horizontal
plate faces upward so turning the wrist does not turn the tag away from the
camera. The cube retains ID 0, and the camera view outlines
the end-effector marker separately. Both tags are detectable at 640x480 at home.

The board and bracket are visual-only, with no added mass or collision geometry.
`end_effector_marker_site` defines the marker face center and mounting axes.
The camera and marker are also included in the bare robot XML.

`RobotModel(camera_name="external_camera")` and
`Simulation(camera_name="external_camera")` select this sensor programmatically.
The wrist camera has been removed. Both CLI and programmatic defaults use
the external camera as the only RGB-D sensor. `--servo` runs eye-to-hand PBVS
in either controller mode; `--pick` adds the separate pickup sequence.
The end-effector marker is raised above the flange to keep the cube tag visible on
approach. Simulation task lights do not cast shadows onto the small tags.

The fixed camera measures the cube tag (ID 0) and tool tag (ID 1) from the same
interface RGB-D exposure. `CameraProcessor.process_marker()` estimates the
tool tag without assuming a cube behind it. The calibrated rigid mounting
transform `tool_from_end_effector_marker` comes from the model's marker site.
Define `camera_from_tool = camera_from_tool_tag @ inverse(tool_from_tool_tag)`
and `tool_from_cube_tag = inverse(camera_from_tool) @ camera_from_cube_tag`.
This relative measurement cancels camera-to-world extrinsics when both tags
are used. Encoders still provide the model Jacobian and collision prediction.

The desired `marker_from_tool` has rotation `Rz(tool_yaw_deg) @ diag(1, -1, -1)`
and translation `[0, 0, tool_clearance]`. PBVS drives the error
`tool_from_cube_tag @ desired_marker_from_tool` to identity, using its translation
and rotation logarithm to request tool-frame linear/angular velocity. The
damped inverse of the tool-frame flange Jacobian maps that twist to joint
velocity; MPC optionally tracks it with the existing collision constraints.
The fixed camera Jacobian is zero and is not used to control the robot.
The former turn-before-translation gate is removed. Translation and rotation
are commanded together from the measured pose error. `tool_yaw_deg` is a final
goal orientation in `controller_config.json`, not a motion phase; the default
90 degrees aligns the fingers with the cube's X faces. Use another yaw to
experiment, accounting for tag visibility and gripper fit around the cube.

`tool_pose_source` selects `tag_with_fk_fallback` (default), `tag_only`, or `fk`.
The default uses the tool tag when its RGB-D pose is valid and encoder FK if
the tool tag is lost/rejected. Strict `tag_only` stops with
`tool_lost_or_rejected`; `fk` selects the former encoder-only tool estimate.
`ControlResult.tool_observation` exposes the measured tool tag and
`tool_pose_source` reports `tag`, `fk_fallback`, or `fk` in the camera/console.
Cube-tag loss or stale/invalid state still stops motion in every mode.

When the optional EKF is enabled, independent trackers filter the moving cube
and the tool-tag-derived flange pose, using the same estimator noise settings.
Each estimates position, orientation, linear velocity, and angular velocity
of its own body. Rejected tool-tag estimates trigger the configured fallback
or stop; predicted-only tag estimates never command motion. With the EKF
disabled, both tag poses are measured directly each frame.
Two tags measure relative pose but do not make
an obscured cube tag observable. For hardware, calibrate intrinsics, the
tool-tag mounting, and robot kinematics. Camera-to-base extrinsics remain
necessary for FK fallback, world-frame target estimates, and pickup.

## Relative pose EKF

`estimator.py` provides `RelativePoseEKF`, an error-state extended Kalman
filter for a translating and rotating cube. The CLI loads
`estimator_config.json` (`--estimator-config path.json` selects another file).
`enabled` controls whether the approach controller uses the EKF; set it to
false to use the raw camera pose. Python callers can pass `EKFConfig` to
`Simulation` or `ApproachController` through `estimator_config`.

The filter's nominal state contains world-frame cube position, rotation,
linear velocity, and angular velocity. Prediction uses constant velocity:
`p_next = p + v*dt`, `R_next = Exp(omega*dt) @ R`. Acceleration process noise
allows these velocities to change. The 12-dimensional error state is
`[position error, rotation error, linear velocity error, angular velocity error]`.
Camera measurements correct position and orientation; velocities are inferred
over multiple frames. Rotation innovations use the SO(3) logarithm, avoiding
Euler-angle wraparound. Covariance uses a Joseph update and an orientation
error reset after injection. This follows the error-state formulation described
by [Solà](https://arxiv.org/abs/1711.02508).

Perception supplies `camera_from_cube`; the interface supplies the measured
`world_from_camera` for that exposure. Their product is a world-frame cube
measurement, so robot/camera movement is separated from cube movement. The
filter then converts its world estimate back to the current optical camera
frame. It consumes measurements only, with no access to MuJoCo object truth.
Camera extrinsic uncertainty is approximated in the configured pose measurement
noise; encoder biases and correlated errors are not separately estimated.

The returned `RelativePoseEstimate` includes:

- `cube_from_camera`: camera pose expressed in the cube frame, as requested.
- `camera_from_cube`: inverse transform, used for PBVS.
- `world_from_cube`, estimated world linear/angular velocity, and 12x12
  world error covariance.
- Measurement timestamp, acceptance flag, and status.

The approach controller exposes the estimate as `pose_estimate`, also carried
by successful `ControlResult` values. It converts the filtered cube pose back
to the marker pose before computing PBVS. `ControlResult.observation` retains
the raw perception output and segmentation; camera annotations still show
that raw measurement. Pickup latches the filtered target at handoff when the
EKF is enabled (the raw target otherwise);
tracking a moving cube during descent is not implemented by the picker.

The position/orientation measurement standard deviations are in meters/radians
and describe the full pose estimate, rather than individual depth pixels.
Acceleration settings are continuous noise densities (m/s^(3/2) and
rad/s^(3/2)); increasing them makes the filter respond faster to changes in
cube motion, at the cost of less smoothing. Initial velocity uncertainties
are in m/s and rad/s. `innovation_gate` limits the squared Mahalanobis
innovation distance. Tune these for the camera and expected object motion.

`predict()` supports short detection gaps up to `max_prediction_age` (default
0.3 s). Afterwards it returns `None`, and a subsequent measurement starts a
new track. The controller still stops on missing/stale/rejected measurements;
it does not command motion from prediction alone. Repeated exposure timestamps
are not fused twice. Call `reset()` after restarting an interface clock.

```python
from camera_processing import CameraProcessor
from estimator import EKFConfig, RelativePoseEKF

processor = CameraProcessor()
ekf = RelativePoseEKF(EKFConfig.load())
frame = interface.capture_rgbd()
observation = processor.process(frame)
estimate = ekf.step(frame, observation)
if estimate is not None:
    camera_pose_in_cube = estimate.cube_from_camera
```

The simulation cube remains stationary unless the experiment moves it. The
EKF's moving-target tests exercise simultaneous camera translation/rotation
and cube translation/rotation, noisy poses, dropouts, and pose outliers.
Headless approach runs still stop when the target is reached.

## Cube pickup

```powershell
venv/Scripts/python.exe simulate.py --pick
venv/Scripts/python.exe simulate.py --headless --pick --steps 15000
```

`--pick` enables the selected approach controller, then hands off its final
cube observation to `CubePickupController` in `controller.py`. `--servo`
continues to run only the approach.

The new class contains the complete pickup sequence: open, descend in world
Z while holding X/Y and orientation, close, wait for the measured finger
opening to settle on the cube, and lift along the same line to the entry
height. It then holds a fixed joint position with the gripper closed. The
grasp site's height is aligned with the detected cube center. Defaults are
30 mm/s vertical speed, 50 ms updates, and a 20-second timeout per phase;
constructor arguments can change them.

The sequence uses interface state and actuator commands, not simulator
measurements. It latches the detected target at handoff and does not require
the tag to remain visible during pickup. `start(observation)` begins it;
`update()` advances it without blocking and returns `ControlResult` with
`pickup_opening`, `pickup_descending`, `pickup_closing`, `pickup_lifting`,
`pickup_complete`, or `pickup_failed`. `finished` marks completion or failure.

Pickup uses Cartesian velocity feedback rather than the approach MPC. It
checks known collision geometry at the current and next predicted arm pose,
keeping 20 mm clearance except for intentional finger/cube contact. A blocked
path stops the sequence instead of planning a detour. Stale measurements,
joint-limit violations, timeouts, and empty/lost grasps also stop the arm.
Width settling is a grasp heuristic, not force or slip sensing.
During closing, the controller uses the median opening and a fitted width
trend over 0.5 seconds. A trend below 2 mm/s indicates settling; individual
noisy readings no longer reset a consecutive-sample timer. The window must
contain at least five distinct acquisitions. Empty grasps still stop the
sequence, and fingers that continue closing do not trigger the lift.

Only pickup scenes give the cube a free joint and use elliptic contacts with
increased tangential stiffness. The cube is never welded to the robot; the
tests check that friction/contact physically lifts it and keeps it raised.

## Measurement noise

The CLI loads `noise_config.json` independently of the working directory.
Its default profile enables mild measurement noise with seed 42:

```powershell
venv/Scripts/python.exe simulate.py --servo
venv/Scripts/python.exe simulate.py --pick
venv/Scripts/python.exe simulate.py --servo --no-noise
venv/Scripts/python.exe simulate.py --servo --noise-config my_noise.json
```

Set `enabled` to false to disable noise, or set individual standard deviations,
biases, and dropout probability to zero. Set `seed` to an integer for replay,
or null for a fresh random sequence. These values are example experiment
settings, not specifications for a particular real sensor.

| Setting | Units / meaning |
| --- | --- |
| `joint_position_std`, `joint_position_bias` | Radians; scalar or six values in arm joint order |
| `joint_velocity_std`, `joint_velocity_bias` | Radians/second; scalar or six values |
| `gripper_width_std`, `gripper_width_bias` | Meters of inner opening; readings clipped to the gripper range |
| `rgb_pixel_std` | Intensity levels on the 0–255 scale; RGB remains uint8 |
| `depth_std` | Base depth standard deviation, meters |
| `depth_std_per_meter_squared` | Added standard deviation per squared meter of depth; coefficient units 1/m |
| `depth_bias` | Fixed depth offset, meters |
| `depth_dropout_probability` | Probability 0–1 that a valid depth pixel becomes a hole |

`noise.py` implements `MeasurementNoise`, with independent Gaussian errors
and fixed configurable biases. The depth model is
`sigma(z) = depth_std + depth_std_per_meter_squared * z**2`. Invalid/background
depth stays invalid; dropouts and readings outside the valid range become zero.
RGB values are rounded and clipped, and all returned arrays are owned copies.

`MujocoInterface` applies this model only to outgoing measurements. Physics,
actuator commands, and timestamps stay clean. Camera images are rendered from
true physics, while `world_from_camera` remains the calibrated fixed mount
pose. Encoder noise affects the tool FK used by PBVS, and RGB/depth noise
affects cube perception. No independent extrinsic pose noise is added.
An estimated cube bottom can fluctuate slightly below the floor under noise;
this is measurement uncertainty, not a change to the simulated cube position.
No ground-truth floor correction is applied to perception. A controller status
of `target_lost_or_stale` with `error inf` means there is no usable fresh target
measurement, rather than an infinite or underground physical pose. The appended
diagnostic identifies the rejection reason, such as `tag_depth_not_planar`.

Repeated reads of an unchanged state return one cached noisy sample with
independent arrays. Camera noise is drawn on each capture; its random streams
do not alter encoder samples. `Simulation.reset()` restarts seeded noise and
restores nominal home commands. Replay requires the same measurement/capture
sequence and configuration.

Python callers use noise-free defaults unless they supply settings explicitly:

```python
from noise_config import NoiseConfig
from simulate import Simulation

simulation = Simulation(noise_config=NoiseConfig.load())
state = simulation.interface.read_state()  # Noisy feedback, clean physics.
```

Only the MuJoCo adapter injects synthetic noise; hardware adapters return their
real measurements. The relative pose EKF filters the camera target downstream
of this interface. The noise module does not add actuator disturbances,
communication delays/dropouts, or calibration perturbations.

## Simulation plots

`plots.py` provides `SimulationPlots`. Simulation records every control update
in `simulation.plots`, including approach, target loss, and pickup. Samples use
interface joint position/velocity measurements and a separate FK/Jacobian
workspace. Recording does not read simulator ground truth, send commands, or
alter the controller's prediction workspace.

To show live plots alongside the model and camera:

```powershell
py simulate.py --servo --plots
py simulate.py --pick --plots
```

To export without opening plot windows:

```powershell
py simulate.py --headless --servo --plots --steps 6000
py simulate.py --servo --plot-dir results/test_01
```

`--plots` saves to `simulation_plots` after the run ends; `--plot-dir` selects
another output directory and enables export. Headless mode exports without a
GUI backend. The generated files are:

- `trajectory.png`: measured camera and tool flange paths in world coordinates,
  plus the estimated cube path when an observation is available.
- `linear_velocity.png`: tool flange X/Y/Z linear velocity, in m/s.
- `angular_velocity.png`: tool flange X/Y/Z angular velocity, in rad/s.
- `trajectory.csv`: all plotted values, controller status, elapsed simulation
  time, and absolute interface timestamps. Positions use meters; velocity
  columns use m/s and rad/s as above.

The velocity figures compare measured motion, the PBVS joint reference, and
the applied joint command after mapping all three through the tool Jacobian
into world axes. Measured velocity comes from `J(q) @ measured_joint_velocity`,
not numerical differentiation of noisy positions. With MPC enabled, its applied
command may differ from the PBVS reference. Pickup samples retain measured and
commanded motion but leave the PBVS reference blank because pickup uses a
separate controller.

Matplotlib is included in `requirements.txt`. Python callers can export after
`run_controlled()` with `simulation.plots.save("results/test_01")`, inspect the
owned arrays in `simulation.plots.data`, and release figure resources with
`simulation.plots.close()`. Simulation reset clears the recorded history.
