# Configuration

Active JSON configuration files live here. Default loaders resolve this directory
relative to the application, independently of the current working directory.
Explicit command-line config paths still work.

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
