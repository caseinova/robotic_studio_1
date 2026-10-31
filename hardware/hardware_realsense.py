"""D435i RGB-D adapter. The SDK is imported only when opening the camera.

Depth is aligned to RGB using librealsense; HardwareInterface handles the
configured OpenCV rectification and metric depth scaling. The IMU is unused.
https://github.com/realsenseai/librealsense/blob/master/wrappers/python/examples/align-depth2color.py
"""

import argparse
import copy
import json
from pathlib import Path
import time

import numpy as np

from hardware.hardware_devices import CameraSample


def validate_settings(config):
    options = config.get("options", {})
    allowed = {"serial", "fps", "timeout_ms", "warmup_frames", "max_rgb_depth_skew_ms", "timestamp_margin_s",
               "rgb_auto_exposure", "rgb_exposure", "rgb_gain", "rgb_brightness"}
    if not isinstance(options, dict) or set(options) - allowed:
        raise ValueError("unknown D435i camera options")
    auto = options.get("rgb_auto_exposure")
    if auto is not None and type(auto) is not bool:
        raise ValueError("camera.options.rgb_auto_exposure must be boolean or null")
    for name in ("rgb_exposure", "rgb_gain", "rgb_brightness"):
        value = options.get(name)
        if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float))
                                  or not np.isfinite(value)):
            raise ValueError(f"camera.options.{name} must be finite numeric or null")
    if auto is True and any(options.get(name) is not None for name in ("rgb_exposure", "rgb_gain")):
        raise ValueError("manual rgb_exposure/rgb_gain requires rgb_auto_exposure=false or null")
    for name, minimum in (("fps", 1), ("timeout_ms", 1), ("warmup_frames", 0)):
        if type(options.get(name)) is not int or options[name] < minimum:
            raise ValueError(f"camera.options.{name} must be an integer >= {minimum}")
    serial = options.get("serial")
    if serial is not None and (not isinstance(serial, str) or not serial.strip()):
        raise ValueError("camera.options.serial must be a nonempty string or null")
    for name in ("max_rgb_depth_skew_ms", "timestamp_margin_s"):
        value = options.get(name)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value) or value < 0:
            raise ValueError(f"camera.options.{name} must be finite and non-negative")
    for name in ("width", "height"):
        if type(config.get(name)) is not int or config[name] <= 0:
            raise ValueError(f"camera.{name} must be a positive integer")


class RepeatedFrameError(RuntimeError):
    """A frameset reused an already delivered color or depth frame."""


class RealSenseD435i:
    def __init__(self, config, clock=time.monotonic, *, sdk=None, wall_clock=time.time):
        validate_settings(config)
        if sdk is None:
            try:
                import pyrealsense2 as sdk
            except ImportError as error:
                raise RuntimeError("D435i requires pyrealsense2; install requirements-hardware.txt") from error
        self.sdk, self.config, self.clock, self.wall_clock = sdk, config, clock, wall_clock
        self.options = config["options"]
        self._wall_offset = clock() - wall_clock()
        self._last_numbers = None
        self._closed, self._started, self._start_attempted = False, False, False
        self.pipeline = sdk.pipeline()
        try:
            stream_config = sdk.config()
            serial = self.options.get("serial")
            if serial is not None:
                stream_config.enable_device(serial)
            elif len(sdk.context().query_devices()) > 1:
                raise RuntimeError("multiple RealSense devices found; set camera.options.serial")
            for stream, pixel_format in ((sdk.stream.color, sdk.format.rgb8), (sdk.stream.depth, sdk.format.z16)):
                stream_config.enable_stream(stream, config["width"], config["height"], pixel_format, self.options["fps"])
            device = stream_config.resolve(sdk.pipeline_wrapper(self.pipeline)).get_device()
            self.device_name = device.get_info(sdk.camera_info.name)
            self.serial = device.get_info(sdk.camera_info.serial_number)
            if "D435I" not in self.device_name.upper():
                raise RuntimeError(f"expected a D435i, found {self.device_name}")
            for sensor in device.query_sensors():
                if sensor.supports(sdk.option.global_time_enabled):
                    sensor.set_option(sdk.option.global_time_enabled, 1)
                if sensor.supports(sdk.option.frames_queue_size):
                    sensor.set_option(sdk.option.frames_queue_size, 1)
            self._start_attempted = True
            profile = self.pipeline.start(stream_config)
            self._started = True
            self.depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
            if not np.isfinite(self.depth_scale) or self.depth_scale <= 0:
                raise RuntimeError("D435i reported an invalid depth scale")
            if (config.get("depth_scale") is not None
                    and not np.isclose(config["depth_scale"], self.depth_scale, rtol=1e-5, atol=1e-9)):
                raise ValueError(f"camera.depth_scale must match D435i depth units: {self.depth_scale}")
            self.intrinsics = profile.get_stream(sdk.stream.color).as_video_stream_profile().get_intrinsics()
            if (self.intrinsics.width, self.intrinsics.height) != (config["width"], config["height"]):
                raise RuntimeError("D435i returned a different color resolution than configured")
            self.align = sdk.align(sdk.stream.color)
            self._configure_rgb(profile.get_device())
            for _ in range(self.options["warmup_frames"]):
                self.pipeline.wait_for_frames(self.options["timeout_ms"])
        except BaseException:
            try:
                self.close()
            except Exception:
                pass  # Preserve the initialization failure after attempting cleanup.
            raise

    def _configure_rgb(self, device):
        values = {name: self.options.get("rgb_" + name)
                  for name in ("exposure", "gain", "brightness")}
        auto = self.options.get("rgb_auto_exposure")
        if values["exposure"] is not None or values["gain"] is not None:
            auto = False
        requested = [("enable_auto_exposure", auto), *values.items()]
        requested = [(name, float(value)) for name, value in requested if value is not None]
        if not requested:
            return
        sensor = device.first_color_sensor()
        checked = []
        for name, value in requested:
            option = getattr(self.sdk.option, name)
            if not sensor.supports(option) or sensor.is_option_read_only(option):
                raise ValueError(f"D435i RGB option {name} is unsupported or read-only")
            limits = sensor.get_option_range(option)
            if (not limits.min <= value <= limits.max or
                    (limits.step > 0 and not np.isclose((value - limits.min) / limits.step,
                     round((value - limits.min) / limits.step), rtol=0, atol=1e-5))):
                raise ValueError(f"D435i RGB {name}={value:g} must be in "
                                 f"[{limits.min:g}, {limits.max:g}] with step {limits.step:g} (SDK units)")
            checked.append((option, value))
        # Disable auto exposure before manual values; never change the depth sensor.
        for option, value in checked:
            sensor.set_option(option, value)

    def capture(self):
        if self._closed:
            raise RuntimeError("D435i camera is closed")
        frames = self.pipeline.wait_for_frames(self.options["timeout_ms"])
        aligned = self.align.process(frames)
        color, depth = aligned.get_color_frame(), aligned.get_depth_frame()
        if not color or not depth:
            raise RuntimeError("D435i did not provide a complete aligned RGB-D frame")
        # Hardware/system-domain timestamps are not silently re-stamped as now.
        if any(frame.get_frame_timestamp_domain() != self.sdk.timestamp_domain.global_time for frame in (color, depth)):
            raise RuntimeError("D435i global acquisition timestamps are not ready; check firmware/driver or increase warmup_frames")
        times = [frame.get_timestamp() / 1000 for frame in (color, depth)]
        if not np.all(np.isfinite(times)):
            raise RuntimeError("D435i timestamps are invalid")
        if abs(times[0] - times[1]) * 1000 > self.options["max_rgb_depth_skew_ms"]:
            raise RuntimeError("D435i color/depth acquisition skew exceeds configured limit")
        numbers = (color.get_frame_number(), depth.get_frame_number())
        if self._last_numbers is not None:
            if any(a < b for a, b in zip(numbers, self._last_numbers)):
                raise RuntimeError("D435i restarted an RGB/depth frame stream; restart collection")
            if any(a == b for a, b in zip(numbers, self._last_numbers)):
                raise RepeatedFrameError("D435i repeated an RGB/depth frame stream")
        if abs((self.clock() - self.wall_clock()) - self._wall_offset) > .02:
            raise RuntimeError("host wall clock changed; restart the D435i session to remap acquisition time")
        timestamp = min(times) + self._wall_offset - self.options["timestamp_margin_s"]
        if timestamp > self.clock():
            raise RuntimeError("D435i global timestamp is in the future; clock synchronization has not converged")
        self._last_numbers = numbers
        return CameraSample(timestamp, np.asanyarray(color.get_data()).copy(),
                            np.asanyarray(depth.get_data()).copy())

    def factory_calibration(self):
        """Report device intrinsics; never invent camera-to-robot extrinsics."""
        intr = self.intrinsics
        model = str(intr.model)
        coefficients = list(intr.coeffs)
        # OpenCV coefficients can be copied directly for Brown-Conrady or
        # zero-distortion profiles. Other models need explicit calibration.
        supported = intr.model in (self.sdk.distortion.none, self.sdk.distortion.brown_conrady)
        distortion = coefficients if supported or not np.any(coefficients) else None
        return {"device": {"name": self.device_name, "serial": self.serial,
                           "sdk_distortion_model": model, "sdk_distortion_coefficients": coefficients},
                "camera": {"width": intr.width, "height": intr.height,
                           "intrinsics": [[intr.fx, 0, intr.ppx], [0, intr.fy, intr.ppy], [0, 0, 1]],
                           "distortion": distortion, "depth_scale": self.depth_scale}}

    def close(self):
        if not self._closed:
            self._closed = True
            if self._start_attempted:
                try:
                    self.pipeline.stop()
                except RuntimeError:
                    if self._started:
                        raise


def create_camera(config, clock):
    return RealSenseD435i(config, clock)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Print D435i factory calibration without connecting to the robot")
    parser.add_argument("--config", type=Path, default=Path(__file__).resolve().parents[1] / "config" / "hardware_config.json")
    args = parser.parse_args(argv)
    camera = None
    try:
        with args.config.open(encoding="utf-8") as source:
            settings = copy.deepcopy(json.load(source)["camera"])
        settings["depth_scale"] = None  # Inspection reports actual scale rather than requiring it in advance.
        camera = RealSenseD435i(settings)
        print(json.dumps(camera.factory_calibration(), indent=2))
    except (OSError, ValueError, RuntimeError, KeyError) as error:
        parser.exit(2, f"Error: {error}\n")
    finally:
        if camera is not None:
            camera.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
