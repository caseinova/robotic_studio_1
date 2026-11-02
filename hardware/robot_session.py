"""One hardware owner thread, with composable pick and place actions.

Only this thread touches devices. Blocking convenience calls wait on Futures;
nonblocking callers can poll the same Futures. An idle session sends zero
velocity, checks feedback, and expires after a bounded interval.
"""

from concurrent.futures import Future
from queue import Queue, Empty
import threading
import time

import numpy as np

from common.controller import ApproachController, CubePickupController, start_pickup_or_reapproach
from hardware.hardware_runtime import open_hardware
from common.sequence_config import SequenceConfig


from common.pickup import ActionResult, PickAction

class RobotSession:
    def __init__(self, hardware, controller_config, estimator_config, sequence_config=None,
                 *, interface_factory=open_hardware, clock=time.monotonic, camera_view=False,
                 annotations=True):
        hardware.validate_ready(controller_config, task="pick")
        self.hardware, self.controller_config = hardware, controller_config
        self.estimator_config = estimator_config
        self.config = sequence_config or SequenceConfig.load()
        self.kinematics = hardware.build_kinematics()
        self.settings = hardware.build_mpc_config()
        self.cube_size = hardware.perception["cube_size"]
        self.tool_from_cube = None
        self.status, self.error = "closed", None
        self.diagnostic = ""
        self.tracking_report = ""
        self.interface = None
        self.camera_view = camera_view
        self.annotations = annotations
        self._preview_lock = threading.Lock()
        self._preview = None
        self._factory, self._clock = interface_factory, clock
        self._requests = Queue()
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._pending = None
        self._thread = None
        self._ready = Future()

    def __enter__(self):
        if self._thread is not None:
            raise RuntimeError("sessions cannot be reopened")
        self._thread = threading.Thread(target=self._run, name="robot-session", daemon=True)
        self._thread.start()
        try:
            self._ready.result(timeout=30.)
        except BaseException:
            self.close()
            raise
        return self

    def __exit__(self, exc_type, exc, tb):
        try:
            self.close()
        except Exception as cleanup:
            if exc is not None and exc is self.error:
                return False  # The action already propagated this same session failure.
            if exc is None:
                raise
            raise RuntimeError(f"{exc}; additionally: {cleanup}") from exc

    def publish_preview(self, frame):
        """Publish only the latest frame; no rendering or unbounded frame queue."""
        if self.camera_view:
            with self._preview_lock:
                self._preview = frame

    def latest_preview(self):
        with self._preview_lock:
            return self._preview

    def validate_destination(self, name):
        """Offline preflight; call before pickup when planning a full sequence."""
        return self.config.validate_destination(name, self.hardware, self.kinematics)

    def _submit(self, kind, destination=None):
        with self._lock:
            if self._stop.is_set() or self._thread is None or not self._thread.is_alive():
                raise RuntimeError(f"session is not running: {self.error or self.status}")
            if self._pending is not None and not self._pending.done():
                raise RuntimeError("an action is already active")
            future = Future()
            future.set_running_or_notify_cancel()
            self._pending = future
            self._requests.put((kind, destination, future))
            return future

    def start_servo_to_pick_up(self):
        return self._submit("pick")

    def start_move_to_destination_and_drop(self, name):
        self.validate_destination(name)
        return self._submit("place", name)

    def servo_to_pick_up(self):
        return self.start_servo_to_pick_up().result()

    def move_to_destination_and_drop(self, name):
        return self.start_move_to_destination_and_drop(name).result()

    def close(self):
        """Cancel motion and close devices; never automatically release a load."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5.)
            if self._thread.is_alive():
                raise RuntimeError("session did not stop; verify robot stopped on pendant")
        if self.error is not None:
            raise RuntimeError(f"robot session failed: {self.error}") from self.error

    def _hold(self):
        self.interface.require_camera_for_motion = False
        state = self.interface.read_state()
        self.interface.set_joint_velocities(np.zeros(6))
        if self.tool_from_cube is not None and not self.cube_size * .5 <= state.gripper_width <= self.cube_size * 1.15:
            raise RuntimeError("grasp_lost while holding")

    def _run(self):
        active = future = None
        kind = None
        try:
            with self._factory(self.hardware, clock=self._clock) as interface:
                self.interface = interface
                interface.prepare_motion()
                self.status = "idle"
                self._ready.set_result(None)
                idle_since = self._clock()
                while not self._stop.is_set():
                    cycle = self._clock()
                    previous_preview = self.latest_preview()
                    if active is None:
                        try:
                            kind, destination, future = self._requests.get_nowait()
                        except Empty:
                            if cycle - idle_since > self.config.idle_timeout_s:
                                raise TimeoutError("session idle_timeout_s exceeded")
                            self._hold()
                        else:
                            if kind == "pick":
                                if self.tool_from_cube is not None:
                                    raise RuntimeError("already holding a cube")
                                active = PickAction(self)
                            else:
                                if self.tool_from_cube is None:
                                    raise RuntimeError("placement requires a successful pickup in this session")
                                from common.placement import PlaceAction
                                active = PlaceAction(self, self.validate_destination(destination))
                            action_started = cycle
                    completed = False
                    if active is not None:
                        if cycle - action_started > min(self.config.action_timeout_s,
                                                       self.hardware.control["max_run_seconds"]):
                            raise TimeoutError("action time limit exceeded")
                        completed = active.update()
                    if self.camera_view and self.latest_preview() is previous_preview:
                        # Latched pickup/placement phases need no vision for motion,
                        # but the operator still needs fresh display images.
                        self.publish_preview(interface.capture_rgbd())
                    if self._clock() - cycle > self.hardware.control["max_cycle_time"]:
                        raise TimeoutError("session control cycle exceeded max_cycle_time")
                    if completed:
                        future.set_result(ActionResult(kind, "complete", self.tool_from_cube is not None))
                        active = future = None
                        self.status = "holding" if self.tool_from_cube is not None else "idle"
                        idle_since = self._clock()
                    self._stop.wait(max(0., self.settings.dt - (self._clock() - cycle)))
                interface.set_joint_velocities(np.zeros(6))
        except BaseException as error:
            self.error = error
            self.diagnostic = f"during {self.status}: {error}"
            if not self._ready.done():
                self._ready.set_exception(error)
        finally:
            self._stop.set()
            self.status = "failed" if self.error else "closed"
            with self._lock:
                if self._pending is not None and not self._pending.done():
                    self._pending.set_exception(self.error or RuntimeError("session cancelled"))
