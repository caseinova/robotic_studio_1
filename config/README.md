# Configuration

Active JSON configuration files live here. Default loaders resolve this directory
relative to the application, independently of the current working directory.
Explicit command-line config paths still work.

Paths inside `runtime_config.json` are relative to this directory.
`hardware_config.json` resolves `robot.api_path` relative to this directory too.
Python configuration/schema modules remain in the application root.

`controller_config.json` supports `grasp_height_offset_m` (default `0.0`).
This shifts the pickup grasp-center target along world Z relative to the cube
center: `0.005` grips 5 mm higher, `-0.005` grips 5 mm lower. It does not change
the approach target, camera calibration, or measured tool-to-grasp transform.
Simulation and hardware share this setting. Collision, freshness and timeout
checks remain active; an offset is not a substitute for table/gripper geometry.

Current calibration results and samples remain in the application root.
Historical calibration copies and configuration backups live in `../old_version/`.
New calibration and capsule-generation backups are also written there.
Archived snapshots retain their original contents, including historical paths;
review relative paths when restoring one into this directory.
