# Robotic Studio

Robotic Studio is an experimental UR3e and OnRobot RG2 workspace for visual
servoing, cube pickup, and marker-guided placement. It provides a MuJoCo
simulation and a real-hardware backend that use shared perception, control,
and pick-and-place actions. A desktop launcher can edit configurations and run
experiments without changing the command-line entry points.

The project is research software, not a safety-rated robot controller.
Simulation results, collision geometry, calibration checks, and offline
validation do not certify a physical installation. **Keep the physical emergency
stop accessible and verify the robot has stopped on its pendant.** The GUI Stop
button is a cooperative software stop, not an emergency stop.

## What it does

- Detects a tagged cube from RGB-D images and estimates its pose. The cube uses
  ArUco `DICT_4X4_50` IDs 0 and 2-6; the tool tag uses ID 1. Marker placement
  and transforms must match the physical installation.
- Approaches a cube with position-based visual servoing (PBVS), optionally
  using joint-space MPC, then opens, descends, grips, and lifts.
- Runs a pick-and-place sequence toward a configured destination marker, with
  placement phases, timeouts, and geometric collision checks.
- Offers independent cube and tool EKF settings. The optional tool FK-increment
  prediction uses joint feedback for motion prediction and accepted tag poses
  for correction; it is not an online camera-calibration solution.
- Simulates RGB-D, joint, and gripper measurement noise and exports estimated
  poses, uncertainty, and MuJoCo truth for analysis.

See [CONTROLLER.md](CONTROLLER.md) for controller, estimation, and perception
details, and [SEQUENCES.md](SEQUENCES.md) for action flow and limitations.

## Fresh-Environment Setup

The supplied API guide uses Ubuntu and Python 3.12. The desktop launcher has
been tested on Linux; availability of MuJoCo, RealSense, RTDE, and other device
dependencies varies by operating system and Python version. Clone the repository,
open a terminal in the cloned `robotic_studio_1` directory containing this
README, and use a virtual environment. The `UR3_RG2_rosbridge/` directory is
included in the repository; keep it alongside `common/` and `hardware/`.

```bash
python3 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip

# Simulation, plotting, and command-line runs
python -m pip install -r requirements.txt

# Optional desktop launcher (also installs simulation dependencies)
python -m pip install -r requirements-gui.txt

# Also install these if using real hardware
python -m pip install -r requirements-hardware.txt
```

On Windows, activate with `venv\Scripts\activate` and use the appropriate
Python launcher. For a simulation-only command-line setup, the GUI and hardware
requirements are optional. For the desktop launcher with hardware, install both
`requirements-gui.txt` and `requirements-hardware.txt`. Real hardware additionally
needs compatible device SDKs, network/USB access, and robot-side setup. See [HARDWARE.md](HARDWARE.md) and
[the supplied UR3/RG2 API guide](UR3_RG2_rosbridge/README.md).

## Desktop launcher

**Before running `run_experiments.py`, complete the [fresh-environment setup](#fresh-environment-setup)**:
create and activate the virtual environment, then install `requirements-gui.txt`.
To use the Hardware tab, also install `requirements-hardware.txt` and complete
the [hardware setup and calibration](#hardware-from-a-fresh-installation).

```bash
python run_experiments.py
```

The **Simulation** and **Hardware** tabs have separate editable drafts.
Settings are organized into collapsible categories; arrays and matrices can be
edited as JSON. Choose Observe, Servo, Pick, or Pick and place, and select a
destination for the latter. Validate performs a configuration preflight;
Start validates and then runs the same snapshot. Hardware runs require an
additional confirmation.

Draft edits do not automatically overwrite `config/`. Use **Save profile** to
persist them. Each launched experiment records a configuration snapshot and
console log under `experiment_runs/`. See [DESKTOP.md](DESKTOP.md) for profile,
process, and stop behavior.

## Simulation

Use an explicit backend so a run cannot accidentally select hardware from a
saved runtime profile:

```bash
# Read-only configuration preflight
python run.py --backend simulation --task pick --validate

# Approach a cube, then optionally pick it up
python run.py --backend simulation --task servo
python run.py --backend simulation --task pick

# Run without a viewer
python run.py --backend simulation --task pick --headless --steps 10000

# Pick up and place at a named destination
python run_sequence.py drop_zone_b --backend simulation --validate
python run_sequence.py drop_zone_b --backend simulation
python run_sequence.py drop_zone_b --backend simulation --headless
```

`run.py --task pick` is pickup only. `run_sequence.py` performs pickup and
placement; always specify `--backend simulation` when simulating because the
sequence CLI otherwise defaults to hardware. Destination marker settings are
in `config/sequence_config.json`, and their simulated positions are in
`config/simulation_config.json`.

Headless servo/pick still renders RGB-D frames for perception and therefore
needs a working MuJoCo offscreen OpenGL backend. On a Linux setup with EGL
available, for example, prefix a headless command with `MUJOCO_GL=egl`.
The correct backend and system graphics packages depend on the machine; see
[the simulator/API setup guide](UR3_RG2_rosbridge/README.md) if rendering fails.

The simulation runners save an estimation graph, orientation-error graph, and
CSV in a unique folder under `simulation_plots/`; the console prints its path.
The plots compare simulated truth with available estimates or unfiltered poses.
Uncertainty bands represent filter covariance, not guaranteed physical error.
Set `viewer.show_estimates` in `simulation_config.json` to show available cube
and tool estimates in the MuJoCo viewer. Set `noise_config.json` to configure
measurement noise; it does not directly add actuator disturbances.

For additional simulator options and optional live control plots:

```bash
python -m simulation.simulate --help
python -m simulation.simulate --servo --plots
```

## Hardware: From a Fresh Installation

The hardware path uses the bundled `UR3_RG2_rosbridge` arm API, a RealSense
camera adapter, and a separately configured gripper adapter. Complete the
steps below for the **actual robot, camera, and workspace**. Existing addresses,
calibrations, limits, marker transforms, and collision geometry in the repository
may describe a different installation: populated values are not proof that they
are correct or safe for yours.

### 1. Prepare and Connect the Devices

1. Install `requirements-hardware.txt` in the environment above. For the GUI,
   install `requirements-gui.txt` too. Ensure the system can access the
   RealSense camera over USB and reach the robot controller over the network.
   Use your lab's device/SDK setup and USB-permission instructions.
2. Prepare the robot using its pendant and your lab's procedure. Check the
   operating mode, payload, emergency stop, workspace, and any required robot
   program or URCap. Connect the gripper as required by your installation.
3. Open `config/hardware_config.json`. Confirm `robot.driver`, `robot.api_path`,
   and `robot.host`; supply the reachable robot controller address. Confirm the
   camera factory and stream resolution, and select `camera.options.serial`
   when appropriate. Confirm the gripper driver, factory, host, port, force,
   and physical maximum opening against your device.

For the wrist-connected XML-RPC adapter, the documented factory is
`hardware.hardware_onrobot_xmlrpc:create_gripper`; the server details still
need to match the robot installation. For arm-only observation or servo, a
gripper can be disabled, subject to the static-tool-geometry requirements in
[HARDWARE.md](HARDWARE.md). Pickup requires a working gripper adapter and
feedback. Camera, arm, and gripper are separate interfaces; do not assume a
gripper host without checking the actual connection.

### 2. Print and Mount the Markers

The configured cube and tool markers use `DICT_4X4_50`. The default configured
**black marker square**, including its black border, is 40 mm by 40 mm; the
white margin is additional. The cube defaults to 64 mm per side. Check the
actual dimensions and marker placement against `perception.cube_size`,
`perception.marker_size`, `perception.tool_marker_id`,
`perception.tool_marker_size`, and `perception.cube_from_markers` in
`config/hardware_config.json`. ID 1 is the tool tag; IDs 0, 2, 3, 4, 5, and 6
are the configured cube faces. A placement destination is configured separately
in `config/sequence_config.json` with its own ID and size.

Mount tags flat and rigidly, with a clear white border. Their rotation on each
cube face matters: each `cube_from_markers` transform assumes a particular
position and orientation. See [CONTROLLER.md](CONTROLLER.md) and run
`python tools/visualize_cube_tags.py --side y` or `--side x` to inspect the
configured top/side relationship. Do not assume that tags copied from another
cube match your mounting transforms.

The printable A4 sheets are included in the repository:

- [Cube markers](printables/cube_aruco_40mm_A4.pdf): IDs 0 and 2-6, with
  40 mm black squares and 5 mm white margins.
- [Tool marker](printables/tool_aruco.pdf): ID 1, with a 40 mm black square
  and 5 mm white margin.
- [Destination marker](printables/destination_aruco.pdf): ID 7, with an 80 mm
  black square and 5 mm white margin. This matches the supplied `drop_zone_b`
  marker ID and size; verify your active sequence configuration before use.

Print on A4 at **100% / actual size**, with fit-to-page disabled. Check the
100 mm scale line and measure each black marker square with a ruler; the white
margin and cutting outline are not part of the configured marker size. Mount
markers flat and rigidly, preserving the white margin.

**Destination sheet caveat:** The PDF calls ID 7 a table reference and mentions
base-reference calibration. That older calibration mode is not supported; the
current sequence can use ID 7 as a destination, with its pose and orientation
specified in `config/sequence_config.json`. Its printed cutting outline can
also be mistaken for the marker's outer boundary: in an uncut rasterized test,
OpenCV detected the 90 mm outline instead of the 80 mm black square. Trim off
the cutting line while retaining the white margin, then check the detected
corners on the mounted tag before using its pose. Do not change the configured
80 mm marker size to compensate for a misdetected outline.

### 3. Import and Check Camera Intrinsics

Camera intrinsics describe the camera's optical projection and distortion;
they are **not** the camera's position relative to the robot. For a connected
RealSense, close RealSense Viewer and other applications using the device, then:

```bash
# Inspect the connected camera and proposed factory calibration only
python update_camera_intrinsics.py --dry-run

# Save intrinsics, distortion, depth scale, and serial to hardware_config.json
python update_camera_intrinsics.py

# With multiple cameras, explicitly select the intended serial
python update_camera_intrinsics.py --serial YOUR_CAMERA_SERIAL
```

The script uses the configured stream profile, does not connect to the robot,
and creates a configuration backup under `old_version/` when saving. It refuses
to write if the SDK's distortion model cannot be represented by the expected
OpenCV model; in that case provide a suitable measured OpenCV calibration. If
the camera, stream profile, mounting, robot, or tool-tag mounting has changed,
collect a new extrinsic calibration as appropriate; do not reuse or append an
incompatible earlier camera's samples.

Check the camera feed and tag IDs without connecting to the robot:

```bash
python camera_tag_test.py
# No window: collect a bounded number of frames
python camera_tag_test.py --headless --frames 60
```

Press Q or Escape in the camera window to exit. This is a **detection test**;
it does not establish the tag's world pose or validate camera-to-base
calibration. If the camera reports no device, check its serial, USB permissions,
connection, and whether another process owns the camera. RGB exposure, gain,
and brightness are optional `camera.options` settings; see [HARDWARE.md](HARDWARE.md)
for the auto/manual interaction and device-specific ranges.

### 4. Calibrate Camera to Robot Base and Tool Tag

This project uses a tool-mounted tag and several manually selected robot poses
to estimate `world_from_camera` and `tool_from_marker` jointly. The older
single fixed table/base-tag calibration mode is not supported.

Before collection:

- Set `calibration.world_from_base` to match your chosen world and controller
  base frames. If they are genuinely identical, this is the identity transform;
  verify frame conventions before applying it. Check the robot's configured DH
  model and the camera intrinsics, distortion, profile, depth scale, marker ID,
  and black-square size.
- Keep the camera fixed and mount the tool marker flat and rigidly. Use the
  correct `flange_from_tool` frame definition if one is already configured. If
  it is unset, the collector defines tool as the nominal DH flange and the solver
  records identity for that frame definition. This does **not** measure the TCP
  or `tool_from_grasp`.
- Confirm `robot.host` and the `ur3_rg2_rosbridge` driver. Collection connects
  to the camera and reads robot joint state; it does not command robot or
  gripper motion. Move the arm only through its normal approved controls.

From the application directory:

```bash
# Start a NEW sample file; replace the name for each new installation/session
python calibrate_camera.py collect --samples my_camera_samples.json

# Alternatively, specify the robot host just for collection:
python calibrate_camera.py collect --samples my_camera_samples.json --robot-ip ROBOT_IP

# Resume only a compatible, unchanged camera/tag/model installation
python calibrate_camera.py collect --samples my_camera_samples.json --append

# Solve offline and inspect the result; does not update hardware_config.json
python calibrate_camera.py solve --samples my_camera_samples.json --output my_camera_result.json

# Only after reviewing a passing result: solve and apply to the config
python calibrate_camera.py solve --samples my_camera_samples.json --output my_camera_result.json --write-config
```

The first two collection commands are alternatives, not commands to run one
after the other. In the calibration camera window, move the arm manually to
a safe pose, **stop it**, then press Space in that window. Accepted captures are
saved immediately; Q/Escape finishes collection. Collect at least 12 distinct
poses, preferably 15-25 or more well-distributed positions with tilt around at
least two axes. Keep the tag clearly visible and vary position and orientation.
The collector rejects stale, moving, unstable, and near-duplicate samples.
An existing sample file is not overwritten; `--append` is only for a compatible
continuation, not a replacement or moved camera.

Inspect `pose_count`, `validation_errors`, `fit_errors`, and `passed` in the
result JSON. The solver tests held-out poses and fits all poses; `--write-config`
only applies a passing result when the recorded camera/model/tag metadata still
matches, and saves a backup. A passing numerical fit is **not** proof of physical
accuracy or safety. Independently check known positions and orientations before
using the result for motion. For thresholds, options, and troubleshooting, see
[the calibration section of HARDWARE.md](HARDWARE.md#tool-marker-calibration).

### 5. Finish Installation Geometry and Validate

The camera calibration does not fill in every hardware measurement. Before
motion, measure and verify `tool_from_grasp`, any nonidentity
`flange_from_tool`, the physical gripper's configuration/opening, actual cube
and marker mounting, destination marker transforms, and robot joint limits.
Populate and verify robot/tool collision geometry and workspace obstacles in
`collision`, including the table/support plane. Check `control` velocity,
acceleration, freshness and timeout settings, and `mpc.clearance` against the
real installation and lab rules. Never substitute identity transforms or guess
clearance simply to pass validation. `tool_clearance` is the approach standoff,
not the collision margin.

Start with preflight and observation:

```bash
# Offline: does not open devices
python run.py --backend hardware --task observe --validate
python run.py --backend hardware --task servo --validate

# Reads the camera and robot without commanding arm movement
python run.py --backend hardware --task observe

# Offline destination and sequence preflight
python run_sequence.py drop_zone_b --backend hardware --validate
```

For observation, configure any enabled camera/gripper adapters and their required
connection settings. If the camera is in use by the tag test, close that test
first. Inspect detected markers and compare computed poses against independent
measurements. `--validate` checks configuration, not connectivity or physical
safety; `observe` does not prove that later motion is safe. For a live camera
window, set `camera: true` and `headless: false` in `config/runtime_config.json`,
or use the desktop launcher's corresponding Run settings.

### 6. Run Hardware Motion Only After Verification

The following commands can **move the physical robot**. Run them only after
calibration and physical safety checks, with the workspace supervised and the
physical emergency stop available:

```bash
# Arm approaches the observed cube; this commands robot motion
python run.py --backend hardware --task servo

# Approach, descend, grip, and lift; requires the gripper
python run.py --backend hardware --task pick

# Pick, transfer, descend, release, and retreat at a configured destination
python run_sequence.py drop_zone_b --backend hardware
```

Begin with the lab's approved conservative limits and clearances, watch the
robot and logs, and stop if tracking, direction, contact, or stopping behavior
is unexpected. Failure or exit does not guarantee that the robot stopped or
that the load was released. Verify the pendant and follow the lab procedure.
See [HARDWARE.md](HARDWARE.md) and [SEQUENCES.md](SEQUENCES.md) for command
watchdogs, feedback, timing, collision assumptions, and action failure behavior.

### Troubleshooting

| Symptom | First checks |
| --- | --- |
| Camera reports no device | Close other camera applications; check USB access, connected serial and the configured RealSense stream profile. Run `update_camera_intrinsics.py --dry-run`. |
| Tag missing or `tag_metric_size_inconsistent` | Check dictionary/ID, print scale, measured black-square size, flat mounting, lighting, depth quality, and the correct camera intrinsics. A detection test alone does not validate pose. |
| Calibration returns `passed: false` | Review rejection messages and held-out/fit errors; check mounting, robot model and intrinsics, then collect varied, stationary poses. Do not treat failed validation as a safe calibration. |
| `stale_rgbd_frame` or stale robot state | Stop motion and inspect stream delivery, camera processing time, timestamp synchronization, and control-cycle timing. Age and watchdog settings are safety checks, not performance tuning substitutes. |
| `pickup_path_blocked` or placement blocked | Inspect the named collision pair, modeled geometry, current and predicted clearance, and the actual workspace. Do not disable checks or reduce margins without verifying the geometry and physical risk. |
| Gripper does not move or times out | Check the correct adapter, host/port, device state and physical opening, then follow the gripper/lab procedure. An RPC acknowledgment is not a confirmed grasp. |
| Headless simulation cannot render | Confirm MuJoCo and a supported offscreen graphics backend; for EGL-capable Linux, try `MUJOCO_GL=egl`. No physical device is needed for simulation. |

## Configuration

Active settings are in [`config/`](config/README.md):

| File | Purpose |
| --- | --- |
| `runtime_config.json` | Backend, task, headless/camera display, and config paths |
| `controller_config.json` | PBVS/MPC mode, approach and pickup settings, tool-pose source |
| `sequence_config.json` | Destinations, marker transforms, placement speeds and timeouts |
| `estimator_config.json` | Cube/tool EKF switches and tuning |
| `simulation_config.json` | Scene, viewer, destination positions, simulation collision margins |
| `noise_config.json` | Simulation measurement-noise model and seed |
| `hardware_config.json` | Device connections, calibration, limits, and collision geometry |

Transforms named `A_from_B` map coordinates from frame B into frame A. Units
are specified by configuration keys and the detailed guides. Configuration
and calibration values are installation-specific; inspect them before use.

## Tests

Install pytest in the project environment and run the test suite from this
directory:

```bash
python -m pip install pytest
python -m pytest tests
```

Rendering, desktop, and hardware-related tests have additional environment and
dependency requirements. Hardware operation must be validated separately;
passing software tests does not establish physical safety. See
[STRUCTURE.md](STRUCTURE.md) for package boundaries and entry points.

## Further documentation

- [Controller, perception, EKFs, pickup, and noise](CONTROLLER.md)
- [Simulation and hardware pick-and-place sessions](SEQUENCES.md)
- [Hardware setup, calibration, and safety checks](HARDWARE.md)
- [Desktop launcher and profiles](DESKTOP.md)
- [Configuration reference](config/README.md)
- [Source layout](STRUCTURE.md)
- [Simplified physical gripper collision model](SIMPLIFIED_GRIPPER.md)
- [UR3e simulation model](models/ur3e/README.md) and [RG2 model](models/rg2/README.md)
- [Bundled UR3/RG2 API and setup](UR3_RG2_rosbridge/README.md)
