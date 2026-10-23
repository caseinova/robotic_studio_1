"""Adapt the supplied RealUR3e API to the measured-state command worker."""

import hashlib
import importlib
import importlib.util
from pathlib import Path
import sys

from hardware_devices import URRTDE


DEFAULT_API_PATH = Path(__file__).resolve().parents[1] / "UR3_RG2_rosbridge"


def load_backend(path):
    """Load the sibling package without modifying sys.path or importing a generic robot package."""
    package = Path(path).resolve() / "robot"
    if not (package / "real_ur_backend.py").is_file():
        raise ValueError(f"UR3_RG2_rosbridge backend not found in {package}")
    name = "_ur3_api_" + hashlib.sha256(str(package).encode()).hexdigest()[:12]
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, package / "__init__.py",
                                                     submodule_search_locations=[str(package)])
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        try:
            spec.loader.exec_module(module)
        except Exception:
            sys.modules.pop(name, None)
            raise
    return importlib.import_module(name + ".real_ur_backend").RealUR3e


class RosbridgeArm(URRTDE):
    """Keep vision-rate commands separate from the API's paced speedJ calls.

    The read-only instance never uploads a control script. A second instance
    is constructed and exclusively owned by the command worker on first use.
    The API's RTDE handles supply packet timestamps, safety state, and watchdog
    operations missing from its public contract; measurements and motion use
    get_joint_positions/get_joint_velocities/set_joint_velocity.
    """

    def _connect_receive(self):
        self._backend_type = load_backend(self.settings.get("api_path", DEFAULT_API_PATH))
        self._reader = self._new_backend(enable_control=False)
        self._receive = self._reader.rtde_r

    def _new_backend(self, *, enable_control):
        return self._backend_type(
            self.settings["host"], control_frequency=self.settings["frequency_hz"],
            enable_control=enable_control, max_joint_speed=self.control["velocity_limit"],
            acceleration=self.control["acceleration_limit"],
            stop_acceleration=self.control["acceleration_limit"])

    def _joint_values(self):
        return self._reader.get_joint_positions(), self._reader.get_joint_velocities()

    def _connect_command(self):
        return self._new_backend(enable_control=True)

    def _watchdog_transport(self, sender):
        return sender.rtde_c

    def _send_velocity(self, sender, velocity, period):
        sender.set_joint_velocity(velocity)
        return True  # The supplied API raises on failure and returns None on success.

    def _close_command(self, sender):
        sender.close()

    def _close_receive(self):
        self._reader.close()
