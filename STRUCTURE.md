# Source layout

Run commands from this directory as before:

```bash
python run.py --backend simulation --task pick
python run_sequence.py drop_zone_b --backend simulation
python run_sequence.py drop_zone_b --backend hardware --validate
```

Root Python files are command-line entry points: `run.py`, `run_sequence.py`,
`simulate.py`, `calibrate_camera.py`, `camera_tag_test.py`, and
`update_camera_intrinsics.py`. Calibration and camera commands operate on
hardware; simulation commands do not connect to physical devices.

- `common/`: perception, estimation, controllers, pickup/placement actions,
  interfaces, collision geometry, and shared configuration loaders.
- `simulation/`: MuJoCo model, I/O, prediction workspace, physics runner,
  synchronous sequence session, simulated noise, and plots.
- `hardware/`: device adapters, calibrated UR kinematics, measured I/O,
  configuration loader, runtime, and threaded robot session.
- `config/`: JSON settings; filenames and numerical settings are unchanged.
- `models/`, `printables/`, `simulation_plots/`: assets and generated outputs.
- `tests/`, `tools/`: tests and standalone maintenance/visualization utilities.
- `UR3_RG2_rosbridge/`: vendored robot API; its internal layout is unchanged.
- `old_version/`: archived files, not active source or configuration.

Both backends depend on `common`; `common` imports neither backend. Hardware
code does not import simulation code. Shared actions operate on the session's
interface and prediction workspace. The simulation placement guard is injected
into the same placement action used for hardware.

Python callers now use qualified imports, for example:

```python
from common.controller_config import ControllerConfig
from common.sequence_config import SequenceConfig
from simulation.simulate import Simulation
from simulation.simulation_session import SimulationSession
from hardware.robot_session import RobotSession
```

Default configuration, model, and vendored API paths resolve from source
locations, independent of the current working directory. Paths inside runtime
JSON remain relative to that JSON file. Explicit relative command-line paths
remain relative to the working directory.

Custom device factories use package names, for example
`hardware.hardware_realsense:create_camera` and
`hardware.hardware_onrobot_xmlrpc:create_gripper`. Update these strings in any
external copies of hardware configuration. The bundled active config is updated;
historical backups are intentionally unchanged.

Use `python -m hardware.hardware_realsense` for the adapter's diagnostic CLI,
not direct execution of a file inside a package. Existing root entry-point
commands are unchanged, but old flat-module Python imports are not retained.
