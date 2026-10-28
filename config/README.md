# Configuration

Active JSON configuration files live here. Default loaders resolve this directory
relative to the application, independently of the current working directory.
Explicit command-line config paths still work.

Paths inside `runtime_config.json` are relative to this directory.
`hardware_config.json` resolves `robot.api_path` relative to this directory too.
Python configuration/schema modules remain in the application root.

`estimator_config.json` has independent `cube_enabled` and `tool_enabled`
booleans. For cube-only filtering, set `cube_enabled: true` and
`tool_enabled: false`. A disabled tool EKF leaves raw tool-tag feedback available;
it does not force FK. Noise/tuning settings remain shared, but tracker states
are separate. Older configs using `enabled` still control both trackers; an
explicit per-target boolean overrides that legacy fallback. Restart after edits.

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
