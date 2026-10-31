# Desktop experiments

Install into the same Python environment used for the robot/simulator:

```bash
python -m pip install -r requirements-gui.txt
python run_experiments.py
```

Hardware also needs the existing dependencies in `requirements-hardware.txt`
and the appropriate RealSense/robot SDK installation. PySide6 supports Linux,
Windows and macOS; this launcher has been tested on Linux only. SDK support,
USB access and robot network setup must be verified on each platform.

## Configuration

Simulation and Hardware tabs have independent drafts initialized from `config/`.
Choose Observe, Servo, Pick, or Pick and place. Pick and place also takes a named
destination from the sequence configuration.

Settings are grouped into collapsible categories for runtime, approach/pickup,
placement/destinations, estimation, scene, noise and hardware installation.
Numeric values retain the units used by the JSON keys: meters, seconds, radians
or degrees where explicitly named. Existing values are preserved until edited.
Matrices, arrays, nullable values and collision shapes use JSON dialogs; each
category also has a JSON editor for adding/removing fields or destinations.
Application schema validation remains authoritative; no limits are relaxed.

Use a category's folder button to load a replacement configuration file.
Load profile selects a directory containing `runtime_config.json`, its referenced
configuration files, and `sequence_config.json`. Save profile writes the current
tab to a directory, asking before overwriting existing configuration files.
Settings stay in memory until saved; the default `config/` files are not
automatically overwritten. Unsaved drafts are discarded when the window closes.

Backend/task come from the tab and experiment selector. Configuration-source
references come from the profile/file loaders, rather than editable runtime
path fields. Saved bundles normalize those references to sibling JSON files.
Hardware API paths resolve against the loaded hardware config and are stored
as absolute paths in profiles; adjust them when moving a profile to another PC.
If editing `robot.api_path` directly, supply an absolute path.

## Execution and stopping

Validate runs the existing selected task's offline preflight in a child process.
Start first validates, then runs that same snapshot. A failed preflight does not
launch an experiment. Hardware runs require a separate confirmation after
validation, defaulting to No. Preflight is not physical safety certification.

Every run has a timestamped directory under `experiment_runs/`, with the complete
configuration snapshot, `request.json` and `console.log`. This generated folder
is ignored by Git. The live log is bounded to 5,000 lines; the file retains all
output. Edits and additional runs are locked while a child process is active.
Simulator/camera windows are separate windows owned by the child process.

Stop writes a cooperative request monitored by the Python worker. It interrupts
the entry point so its normal cleanup can stop commands and close devices.
This works without POSIX-only process signals, including on Windows, but native
blocking calls can delay Python cleanup. After ten seconds the GUI reports
shutdown as unconfirmed; it does not forcibly kill the hardware worker or claim
the robot is stopped. Closing during a run offers to request shutdown and leaves
the window open until the child exits.

**GUI Stop is not an emergency stop. Verify the physical robot on the pendant,
and keep the physical emergency stop accessible.** An exited process alone does
not certify that transport cleanup or physical stopping succeeded; inspect logs.

The desktop code is in `desktop/`; controllers and backend APIs are unchanged.
`run.py` and `run_sequence.py` remain available without PySide6 installed.
