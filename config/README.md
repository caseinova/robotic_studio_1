# Configuration

Simulation commands save `estimation.png`, `orientation_error.png`, and
`estimation.csv` to a unique directory under `robotic_studio_1/simulation_plots`
for each run, including headless and interrupted runs. The console prints the
directory. Position plots compare simulated truth and EKF estimates with ±2σ
filter covariance; orientation plots show world-axis rotation-vector error
against the true pose and local-linear ±2σ. The CSV records status and errors;
raw tag/FK poses used without an EKF are labeled by source and plotted without
an uncertainty band. During pickup, `latched_pick_target` is the fixed handoff
target, not a fresh measurement or an estimate of the carried cube's state.
Missing/stale poses and unknown uncertainty are NaN, not ground truth or zero.
The tracker is not run purely for visualization; observe-only runs or disabled
EKFs may have no estimates. Truth is read only by simulation telemetry.

Set `viewer.show_estimates: true` in this file to draw EKF cube (green) and
tool (orange) positions, RGB orientation axes, and ±2σ position ellipsoids in MuJoCo. This option is
also exposed in the desktop Simulation scene settings. Overlays use the MuJoCo
user scene, not collision geometry, and are absent when an estimate is stale.
Filter covariance and ellipsoids are not physical error guarantees or safety
limits. Orientation uncertainty is plotted, not visualized as an ellipsoid.

The optional tool `prediction_model` in `estimator_config.json` can be
`constant_velocity` or `fk_increment`. The latter is active only when the tool
EKF is enabled and `tool_pose_source` is `tag_with_fk_fallback`. It maintains
one six-dimensional pose-error covariance: FK increments predict, accepted tag
poses correct. It initializes from FK when a tag is unavailable. `fk_unverified`
means no visual correction has yet been accepted; `tag_fused` means the current
tag was accepted; `fk_predicted` means prediction without a new accepted tag.

`fk_position_noise` (m/sqrt(s)) and `fk_orientation_noise` (rad/sqrt(s)) set
process noise. `max_position_std` (m), `max_orientation_std` (rad), and
`max_prediction_age` (seconds since initialization or last accepted tag) are
hold limits, not collision margins or guarantees of physical accuracy.
`max_position_correction` (m) and `max_rotation_correction` (rad) limit a
single accepted tag correction. `max_measurement_skew` (seconds) rejects tag
updates when camera and robot-state timestamps are too far apart. This is a
bounded-skew approximation, not motion-compensated timestamp interpolation.
Expiry or excess uncertainty latches a hold; restart the session to reset it.
Configured measurement noise and bounds require experimental validation.

Approach and pickup share this state with placement in a sequence; standalone
pick also uses it. Cube estimation and encoder-based collision checks are
unchanged. `fk` and `tag_only`, disabled tool EKF, and `constant_velocity`
retain their existing behavior. A tool EKF is not a safety-rated function and
does not estimate camera-to-base calibration bias.

Simulation collision margins in `simulation_config.json` (all meters):
- `approach_collision_clearance_m`: approach MPC minimum separation (default 0.015).
- `pickup_collision_clearance_m`: pickup current/next-pose checks (default 0.015).
- `placement_collision_clearance_m`: placement guard and transfer MPC minimum separation (default 0.015).
- `mpc_clearance_buffer_m`: extra linearized MPC prediction margin for approach and transfer (default 0.003; zero allowed).

These apply to standalone and sequence simulation and appear in the desktop
Simulation scene editor. Pure PBVS approach does not perform MPC collision
checks; configuring a margin does not enable MPC. Pickup and placement retain
their explicit collision guards. Intentional contact exemptions are unchanged.
`controller_config.tool_clearance` is the separate approach standoff, not a
collision margin. Hardware configuration is unchanged.

Simulation pickup collision clearance is `pickup_collision_clearance_m` in
`simulation_config.json` (meters; default `0.015`). It applies to standalone
pickup and the pickup portion of a sequence, in GUI and headless runs.
It does not change approach/placement margins, arrival tolerance, or the
allowed finger/cube contact pairs. Hardware retains `hardware_config.json`
`mpc.clearance`. The desktop editor exposes this under Simulation scene.

Active JSON configuration files live here. Default loaders resolve this directory
relative to the application, independently of the current working directory.
Explicit command-line config paths still work.

`runtime_config.json` controls camera overlays with `annotations` (default
`true`). Set it to `false` for a clean camera feed, or use `--no-annotations`
with `run.py` or `run_sequence.py`; `--annotations` turns overlays back on.
The desktop launcher exposes the checkbox under Run settings. This applies to
simulation and hardware, including pick-and-place. Detection, estimation and
control remain active; only camera annotations and phase labels are hidden.

`simulation_config.json` includes a `viewer` section. Both MuJoCo side panels
start hidden. With `resize_window: true`, startup requests a normal, centered
window covering `screen_fraction` (default `0.9`) of the desktop work area.
Optional `width`, `height`, `x`, and `y` override size/position in desktop pixels;
null values use fractional sizing and centering. These settings affect both
simulation entry points, not the perception camera or its separate RGB window.
They also appear under Simulation scene in the desktop launcher.

Automatic geometry uses native Windows or Linux X11/XWayland window-manager
requests and may be overridden by desktop policies. Native Wayland and macOS
fall back to manual resizing with a warning; panel visibility still applies.
Simulation prefers the GLFW X11 backend when `DISPLAY` is available, including
XWayland sessions. An explicit `PYGLFW_LIBRARY_VARIANT` overrides that preference.
Set `resize_window: false` to retain MuJoCo's geometry without a warning.
On multi-monitor desktops, explicit dimensions/coordinates can select the desired
area. Geometry requests are restricted to MuJoCo windows in the current process.

`python run_experiments.py` opens the optional desktop configuration editor and
experiment launcher. See [desktop setup and profiles](../DESKTOP.md). GUI runs
use separate snapshots; editing a draft does not overwrite these files.

To refresh RealSense factory RGB intrinsics at the configured resolution/FPS,
close other camera programs and run `python update_camera_intrinsics.py` from
the application root. It updates camera serial, intrinsics, distortion and depth
scale, preserving exposure settings and all robot transforms. Use `--dry-run`
to inspect only, or `--serial SERIAL` to select a camera. A sole connected camera
is selected even if the config still references a previous device; with multiple
devices the saved serial must match or an explicit selection is required.
The script briefly starts RGB/depth streams but never connects to the robot.
Backups go to `old_version/`. Replacement/moved cameras still require fresh
camera-to-base calibration; old-camera samples must not be appended.

Paths inside `runtime_config.json` are relative to this directory.
`hardware_config.json` resolves `robot.api_path` relative to this directory too.
Python configuration/schema modules live in `common/`, `hardware/` and
`simulation/`, according to their ownership.

`estimator_config.json` has independent `cube_enabled` and `tool_enabled`
booleans. For cube-only filtering, set `cube_enabled: true` and
`tool_enabled: false`. A disabled tool EKF leaves raw tool-tag feedback available;
it does not force FK. Separate `cube` and `tool` objects contain each tracker's
measurement noise, acceleration noise, initial velocity uncertainty, innovation
gate and prediction-age limit. The desktop Pose estimation category exposes both
as collapsible groups. Both simulation and hardware use these settings.
Existing tuning values were copied into both groups without retuning either filter.
Position standard deviation is in meters; orientation standard deviation is in
radians; prediction age is in seconds. These are EKF settings, not artificial
sensor noise from `noise_config.json`.

Older flat tuning fields remain supported as defaults; a field inside `cube` or
`tool` overrides its flat default for that tracker only. Missing fields retain
their legacy defaults. Older configs using `enabled` still control both trackers;
an explicit per-target boolean overrides that fallback. Enable switches remain
top-level, outside the tuning objects. Restart an active run after edits.

`controller_config.json` supports `grasp_height_offset_m` (default `0.0`).
This shifts the pickup grasp-center target along world Z relative to the cube
center: `0.005` grips 5 mm higher, `-0.005` grips 5 mm lower. It does not change
the approach target, camera calibration, or measured tool-to-grasp transform.
Simulation and hardware share this setting. Collision, freshness and timeout
checks remain active; an offset is not a substitute for table/gripper geometry.

`pickup_dwell_seconds` and `pickup_dwell_radius_m` add a wait after confirmed
gripper opening and before descent. The measured grasp center must stay inside
the XY radius around the latched cube center for the full duration; motion within
the radius is allowed. Leaving the radius or losing the required opening resets
the timer. The arm receives zero velocity during the wait; it does not chase the
cube. Pickup does not re-detect cube motion after handoff. The grasp must remain
above the target height. Zero seconds disables the dwell (legacy behavior).
The dwell phase has a timeout equal to the ordinary phase timeout plus the dwell
duration; hardware's overall run-time limit still applies.

Current calibration results and samples remain in the application root.
Historical calibration copies and configuration backups live in `../old_version/`.
New calibration and capsule-generation backups are also written there.
Archived snapshots retain their original contents, including historical paths;
review relative paths when restoring one into this directory.
