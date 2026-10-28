"""Fixed-camera / robot-base calibration with a rigid tool-mounted ArUco tag.

collect: manually position the arm, stop, then press Space to save a pose.
solve: fit camera/tool-marker transforms offline; --write-config applies them.
This script never sends robot or gripper commands. Keep the camera and tag
mount fixed throughout collection. Lengths are meters and angles radians.
"""

import argparse
from contextlib import ExitStack
from datetime import datetime, timezone
import json
from pathlib import Path
import tempfile
import time

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from camera_processing import CameraProcessor
from hardware_config import HardwareConfig, DEFAULT_HARDWARE_CONFIG_PATH
from hardware_kinematics import UR3DH, UR3Kinematics, _pose
from hardware_realsense import RealSenseD435i, RepeatedFrameError
from hardware_rosbridge import RosbridgeArm
from robot_state import RGBDFrame


DEFAULT_SAMPLES = Path(__file__).with_name("camera_calibration_samples.json")
DEFAULT_RESULT = Path(__file__).with_name("camera_calibration_result.json")
MIN_POSES = 12


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, encoding="utf-8", delete=False) as output:
            temporary = Path(output.name)
            json.dump(value, output, indent=2, allow_nan=False)
            output.write("\n")
        temporary.replace(path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def metadata(config):
    world = config.calibration.get("world_from_base")
    if world is None:
        raise ValueError("set calibration.world_from_base before collecting")
    tool = config.calibration.get("flange_from_tool")
    return {
        "world_from_base": _pose(world, "world_from_base").tolist(),
        "flange_from_tool": _pose(np.eye(4) if tool is None else tool, "flange_from_tool").tolist(),
        "tool_was_unset": tool is None,
        "dh": {key: value.tolist() for key, value in vars(UR3DH(**config.robot["dh"])).items()},
        "camera": {key: config.camera[key] for key in
                   ("width", "height", "intrinsics", "distortion", "depth_scale")},
        "serial": config.camera["options"].get("serial"),
        "marker_id": config.perception["tool_marker_id"],
        "marker_size": config.perception["tool_marker_size"],
    }


def workspace(meta):
    # Bounds are unused for FK-only calibration; these are not operating limits.
    return UR3Kinematics(joint_limits=([-1e6] * 6, [1e6] * 6),
        world_from_base=meta["world_from_base"], flange_from_tool=meta["flange_from_tool"],
        tool_from_grasp=np.eye(4), dh=UR3DH(**meta["dh"]))


def pose_distance(a, b):
    return (float(np.linalg.norm(a[:3, 3] - b[:3, 3])),
            float(Rotation.from_matrix(a[:3, :3].T @ b[:3, :3]).magnitude()))


def mean_pose(poses):
    result = np.eye(4)
    result[:3, :3] = Rotation.from_matrix(np.array(poses)[:, :3, :3]).mean().as_matrix()
    result[:3, 3] = np.mean(np.array(poses)[:, :3, 3], axis=0)
    return result


def rectifier(settings):
    k = np.asarray(settings["intrinsics"], dtype=float)
    maps = cv2.initUndistortRectifyMap(k, np.asarray(settings["distortion"], dtype=float),
        None, k, (settings["width"], settings["height"]), cv2.CV_32FC1)

    def convert(sample):
        rgb = cv2.remap(sample.rgb, *maps, cv2.INTER_LINEAR)
        depth = cv2.remap(sample.depth.astype(np.float32) * settings["depth_scale"],
                          *maps, cv2.INTER_NEAREST)
        # Identity here makes camera coordinates the temporary observation frame;
        # only camera_from_marker is consumed, never an assumed world pose.
        return RGBDFrame(rgb, depth, k.copy(), np.eye(4), sample.timestamp, tuple(settings["depth_range"]))
    return convert


def capture_fresh(camera):
    # Calibration may wait for new frames; live control retains fail-fast behavior.
    for attempt in range(5):
        try:
            return camera.capture()
        except RepeatedFrameError:
            if attempt == 4:
                raise RuntimeError("D435i repeated frames on 5 consecutive attempts; "
                                   "check the camera stream/USB connection and restart collection")


def capture_pose(arm, camera, convert, processor, kinematics, samples, *, clock=time.monotonic,
                 check_tag_stability=True):
    joint_samples, poses, times = [], [], []
    for _ in range(10):
        before = arm.read_joint_state()
        frame = convert(capture_fresh(camera))
        after = arm.read_joint_state()
        now = clock()
        ages = {"robot_before": now - before.timestamp, "camera": now - frame.timestamp,
                "robot_after": now - after.timestamp}
        if any(not 0 <= age <= .15 for age in ages.values()):
            details = ", ".join(f"{name}={age * 1000:.1f} ms" for name, age in ages.items())
            raise ValueError("stale measurements; retry after streams stabilize "
                             f"(ages: {details}; allowed 0..150 ms)")
        if min(abs(frame.timestamp - before.timestamp), abs(frame.timestamp - after.timestamp)) > .075:
            raise ValueError("robot/camera timestamp mismatch")
        if max(np.max(np.abs(before.velocities)), np.max(np.abs(after.velocities))) > .01:
            raise ValueError("arm is moving; stop and wait before pressing Space")
        observation = processor.process_marker(frame)
        if observation is None:
            raise ValueError("tool tag rejected: " + processor.last_status)
        joint_samples.extend((before.positions, after.positions))
        poses.append(observation.camera_from_marker)
        times.append([before.timestamp, frame.timestamp, after.timestamp])
    if np.max(np.ptp(joint_samples, axis=0)) > .002:
        raise ValueError("joint position changed during capture")
    tag = mean_pose(poses)
    deviations = [pose_distance(tag, p) for p in poses]
    if check_tag_stability and any(distance > .005 or angle > np.deg2rad(3) for distance, angle in deviations):
        raise ValueError("tool-tag pose is unstable; improve visibility or move camera closer")
    q = np.mean(joint_samples, axis=0)
    tool = kinematics.forward_kinematics(q)
    for saved in samples:
        distance, angle = pose_distance(tool, kinematics.forward_kinematics(saved["joint_positions"]))
        if distance < .015 and angle < np.deg2rad(8):
            raise ValueError("pose too similar to an existing capture; vary position and orientation")
    return {"joint_positions": q.tolist(), "camera_from_marker": tag.tolist(),
            "timestamps": times, "frames_averaged": len(poses),
            "tag_stability_checked": check_tag_stability,
            "tag_position_max_deviation_mm": max(d[0] for d in deviations) * 1000,
            "tag_rotation_max_deviation_deg": float(np.rad2deg(max(d[1] for d in deviations)))}


def collection_data(path, meta, host, append=False):
    if not path.exists():
        if append:
            raise ValueError(f"cannot append: {path} does not exist")
        return {"version": 1, "metadata": meta, "samples": []}
    if not append:
        raise ValueError(f"{path} already exists; use --append or choose a new --samples path")
    with path.open(encoding="utf-8") as source:
        data = json.load(source)
    if not isinstance(data, dict) or data.get("version") != 1 or not isinstance(data.get("samples"), list):
        raise ValueError("cannot append: invalid version-1 sample dataset")
    if data.get("metadata") != meta:
        raise ValueError("cannot append: camera/model/tag settings changed since collection")
    if data.get("robot_host") not in (None, host):
        raise ValueError("cannot append: robot host differs from original collection")
    model = workspace(meta)
    for sample in data["samples"]:
        if not isinstance(sample, dict):
            raise ValueError("cannot append: invalid sample")
        model.forward_kinematics(sample["joint_positions"])
        _pose(sample["camera_from_marker"], "camera_from_marker")
    return data


def collect(args):
    config = HardwareConfig.load(args.config)
    if config.robot["driver"] != "ur3_rg2_rosbridge":
        raise ValueError("collection requires robot.driver=ur3_rg2_rosbridge")
    settings = dict(config.robot)
    settings["host"] = args.robot_ip or settings.get("host")
    if not settings["host"]:
        raise ValueError("set robot.host or pass --robot-ip")
    meta = metadata(config)
    data = collection_data(args.samples, meta, settings["host"], args.append)
    kinematics = workspace(meta)
    processor = CameraProcessor(marker_id=meta["marker_id"], marker_size=meta["marker_size"],
                                cube_size=max(.1, meta["marker_size"]))
    convert = rectifier(config.camera)
    if args.append:
        print(f"Resuming {len(data['samples'])} saved poses. Camera and tag mounting must be unchanged.")
    print(f"Tool tag ID {meta['marker_id']}, black-square size {meta['marker_size'] * 1000:g} mm")
    if meta["tool_was_unset"]:
        print("Calibration tool frame is the nominal DH flange (flange_from_tool = identity).")
    print("Move the arm manually, stop, then press Space. Q/Esc exits. Aim for 15-25 varied poses.")
    print("Tilt around at least two axes; vary position and distance. Keep the camera and tag mount fixed.")
    if args.skip_tag_stability_check:
        print("WARNING: tool-tag pose stability check disabled; noisy captures may reduce calibration accuracy. "
              "Arm-motion, freshness, per-frame pose, and final fit checks remain enabled.")
    with ExitStack() as stack:
        camera = RealSenseD435i(config.camera)
        stack.callback(camera.close)
        if data.get("device_serial") not in (None, camera.serial):
            raise ValueError("cannot append: connected camera serial differs from original collection")
        data["device_serial"] = camera.serial
        data["robot_host"] = settings["host"]
        arm = RosbridgeArm(settings, config.control)
        stack.callback(arm.close)
        name = "Camera-to-base calibration"
        cv2.namedWindow(name, cv2.WINDOW_AUTOSIZE)
        stack.callback(cv2.destroyAllWindows)
        atomic_json(args.samples, data)
        while True:
            frame = convert(capture_fresh(camera))
            observation = processor.process_marker(frame)
            view = cv2.cvtColor(frame.rgb, cv2.COLOR_RGB2BGR)
            if observation is not None:
                cv2.aruco.drawDetectedMarkers(view, [observation.corners.reshape(1, 4, 2).astype(np.float32)],
                                             np.array([[meta["marker_id"]]], dtype=np.int32))
            status = f"Poses: {len(data['samples'])} | " + ("tag valid" if observation else processor.last_status)
            cv2.putText(view, status, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 0, 0), 3)
            cv2.putText(view, status, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 1)
            cv2.imshow(name, view)
            key = cv2.waitKey(1) & 255
            if key in (27, ord("q"), ord("Q")) or cv2.getWindowProperty(name, cv2.WND_PROP_VISIBLE) < 1:
                break
            if key == 32:
                try:
                    sample = capture_pose(arm, camera, convert, processor, kinematics, data["samples"],
                                          check_tag_stability=not args.skip_tag_stability_check)
                except ValueError as error:
                    print(f"Not saved: {error}", flush=True)
                    continue
                data["samples"].append(sample)
                atomic_json(args.samples, data)
                print(f"Saved pose {len(data['samples'])} to {args.samples}", flush=True)
    print(f"Collected {len(data['samples'])} poses. Solve requires at least {MIN_POSES}.")


def check_diversity(poses):
    rotations = np.array([Rotation.from_matrix(poses[0][:3, :3].T @ p[:3, :3]).as_rotvec() for p in poses])
    singular = np.linalg.svd(rotations / np.sqrt(len(poses)), compute_uv=False)
    if singular[1] < np.deg2rad(8):
        raise ValueError("insufficient orientation diversity: tilt around at least two different axes")
    if np.linalg.norm(np.ptp(np.array(poses)[:, :3, 3], axis=0)) < .05:
        raise ValueError("insufficient position spread: collect poses across more of the workspace")


def fit(a, b):
    # A X = Z B, with A=world_from_tool, B=camera_from_marker,
    # X=tool_from_marker, Z=world_from_camera. Relative motions eliminate Z:
    # log(R_Aj.T R_Ai) = R_X log(R_Bj.T R_Bi).
    va, vb = [], []
    for i in range(len(a)):
        for j in range(i):
            ra = Rotation.from_matrix(a[j][:3, :3].T @ a[i][:3, :3]).as_rotvec()
            rb = Rotation.from_matrix(b[j][:3, :3].T @ b[i][:3, :3]).as_rotvec()
            # Avoid the sign ambiguity of rotation logarithms near 180 degrees.
            if max(np.linalg.norm(ra), np.linalg.norm(rb)) < np.deg2rad(170):
                va.append(ra)
                vb.append(rb)
    if len(va) < 3 or np.linalg.svd(vb, compute_uv=False)[1] < .1:
        raise ValueError("insufficient independent relative rotations")
    rx = Rotation.align_vectors(va, vb)[0].as_matrix()
    rz = Rotation.from_matrix(np.array([
        ai[:3, :3] @ rx @ bi[:3, :3].T for ai, bi in zip(a, b)
    ])).mean().as_matrix()
    lhs = np.vstack([np.hstack((ai[:3, :3], -np.eye(3))) for ai in a])
    rhs = np.concatenate([rz @ bi[:3, 3] - ai[:3, 3] for ai, bi in zip(a, b)])
    translation, _, rank, _ = np.linalg.lstsq(lhs, rhs, rcond=None)
    if rank < 6:
        raise ValueError("translation solve is underconstrained; vary arm orientations")
    x, z = np.eye(4), np.eye(4)
    x[:3, :3], x[:3, 3] = rx, translation[:3]
    z[:3, :3], z[:3, 3] = rz, translation[3:]
    return _pose(x, "tool_from_marker"), _pose(z, "world_from_camera")


def errors(a, b, x, z):
    residuals = np.array([pose_distance(ai @ x, z @ bi) for ai, bi in zip(a, b)])
    return {"position_rms_mm": float(np.sqrt(np.mean(residuals[:, 0] ** 2)) * 1000),
            "position_max_mm": float(residuals[:, 0].max() * 1000),
            "rotation_rms_deg": float(np.rad2deg(np.sqrt(np.mean(residuals[:, 1] ** 2)))),
            "rotation_max_deg": float(np.rad2deg(residuals[:, 1].max()))}


ACCEPTANCE_LIMITS = {
    "validation": {"position_rms_mm": 8, "position_max_mm": 11,
                   "rotation_rms_deg": 5, "rotation_max_deg": 8},
    "fit": {"position_rms_mm": 8, "position_max_mm": 11,
            "rotation_rms_deg": 5, "rotation_max_deg": 10},
}


def acceptable(metrics, stage):
    return all(0 <= metrics[name] <= limit for name, limit in ACCEPTANCE_LIMITS[stage].items())


def solve(data):
    if data.get("version") != 1 or len(data.get("samples", [])) < MIN_POSES:
        raise ValueError(f"need a version-1 dataset with at least {MIN_POSES} poses")
    kinematics = workspace(data["metadata"])
    a = [kinematics.forward_kinematics(s["joint_positions"]) for s in data["samples"]]
    b = [_pose(s["camera_from_marker"], "camera_from_marker") for s in data["samples"]]
    training = [i for i in range(len(a)) if i % 4 != 0]
    validation = [i for i in range(len(a)) if i % 4 == 0]
    check_diversity([a[i] for i in training])
    x, z = fit([a[i] for i in training], [b[i] for i in training])
    held_out = errors([a[i] for i in validation], [b[i] for i in validation], x, z)
    x, z = fit(a, b)
    all_poses = errors(a, b, x, z)
    return {"version": 1, "metadata": data["metadata"], "pose_count": len(a),
            "device_serial": data.get("device_serial"), "robot_host": data.get("robot_host"),
            "validation_indices": validation, "validation_errors": held_out,
            "fit_errors": all_poses,
            "acceptance_limits": {stage: dict(limits) for stage, limits in ACCEPTANCE_LIMITS.items()},
            "passed": acceptable(held_out, "validation") and acceptable(all_poses, "fit"),
            "calibration": {"world_from_camera": z.tolist(), "tool_from_marker": x.tolist(),
                            "flange_from_tool": data["metadata"]["flange_from_tool"]}}


def apply_result(config_path, result):
    if not result["passed"]:
        raise ValueError("calibration failed residual checks; configuration was not changed")
    current = HardwareConfig.load(config_path)
    if metadata(current) != result["metadata"]:
        raise ValueError("camera/model/tag settings changed since collection; configuration was not changed")
    original = Path(config_path).read_text(encoding="utf-8")
    values = json.loads(original)
    values["calibration"].update(result["calibration"])
    config_path = Path(config_path)
    archive = (config_path.parent.parent if config_path.parent.name == "config"
               else config_path.parent) / "old_version"
    archive.mkdir(parents=True, exist_ok=True)
    backup = archive / (config_path.name + ".before-calibration-" + str(time.time_ns()) + ".bak")
    with backup.open("x", encoding="utf-8") as output:
        output.write(original)
    atomic_json(config_path, values)
    return backup


def load_reference(path):
    with Path(path).open(encoding="utf-8") as source:
        settings = json.load(source)
    required = {"marker_dictionary", "marker_id", "marker_size", "base_from_reference_tag"}
    if not isinstance(settings, dict) or set(settings) != required:
        raise ValueError("reference config requires exactly: " + ", ".join(sorted(required)))
    if settings["marker_dictionary"] != "DICT_4X4_50":
        raise ValueError("reference marker dictionary must be DICT_4X4_50")
    if type(settings["marker_id"]) is not int or not 0 <= settings["marker_id"] < 50:
        raise ValueError("reference marker_id must be an integer in 0..49")
    size = settings["marker_size"]
    if isinstance(size, bool) or not isinstance(size, (int, float)) or not np.isfinite(size) or size <= 0:
        raise ValueError("reference marker_size must be positive meters")
    if settings["base_from_reference_tag"] is None:
        raise ValueError("measure and set base_from_reference_tag before reference calibration")
    _pose(settings["base_from_reference_tag"], "base_from_reference_tag")
    return settings


def solve_reference(meta, reference, observations):
    if len(observations) < 12:
        raise ValueError("reference calibration needs at least 12 observations")
    world_from_tag = _pose(meta["world_from_base"], "world_from_base") @ _pose(
        reference["base_from_reference_tag"], "base_from_reference_tag")
    poses = [world_from_tag @ np.linalg.inv(_pose(p, "camera_from_reference_tag")) for p in observations]
    camera = mean_pose(poses)
    metrics = errors([np.eye(4)] * len(poses), poses, camera, np.eye(4))
    return {"version": 1, "mode": "reference", "metadata": meta, "reference": reference,
            "pose_count": len(poses), "repeatability_errors": metrics,
            "acceptance_limits": dict(ACCEPTANCE_LIMITS["fit"]),
            "passed": acceptable(metrics, "fit"),
            "calibration": {"world_from_camera": camera.tolist()},
            "accuracy_note": "Repeatability only; does not validate the measured reference mounting or camera intrinsics."}


def reference_calibration(args):
    if args.frames < 12 or not np.isfinite(args.timeout) or args.timeout <= 0:
        raise ValueError("reference mode requires --frames >= 12 and a finite positive --timeout")
    if args.output.resolve() in (args.config.resolve(), args.reference_config.resolve(), DEFAULT_SAMPLES.resolve()):
        raise ValueError("reference output must differ from configuration and sample files")
    config = HardwareConfig.load(args.config)
    reference = load_reference(args.reference_config)
    if reference["marker_id"] in {config.perception["tool_marker_id"],
                                  *(int(i) for i in config.perception["cube_from_markers"])}:
        raise ValueError("reference marker ID must differ from tool and cube IDs")
    meta = metadata(config)
    processor = CameraProcessor(marker_id=reference["marker_id"], marker_size=reference["marker_size"],
                                cube_size=max(.1, reference["marker_size"]))
    convert = rectifier(config.camera)
    observations, timestamps = [], []
    print("Camera-only calibration: keep camera and reference tag fixed. No robot connection.")
    print("Reported errors measure repeatability, not accuracy of the measured base-to-tag mounting.")
    with ExitStack() as stack:
        camera = RealSenseD435i(config.camera)
        stack.callback(camera.close)
        if not args.headless:
            cv2.namedWindow("Reference calibration", cv2.WINDOW_AUTOSIZE)
            stack.callback(cv2.destroyAllWindows)
        started = time.monotonic()
        status = "no frames"
        while len(observations) < args.frames:
            if time.monotonic() - started > args.timeout:
                raise RuntimeError(f"reference capture timed out: {len(observations)}/{args.frames}; {status}")
            frame = convert(capture_fresh(camera))
            observation = processor.process_marker(frame)
            fresh = 0 <= time.monotonic() - frame.timestamp <= config.control["frame_max_age"]
            fresh = fresh and (not timestamps or frame.timestamp > timestamps[-1])
            status = processor.last_status if fresh else "stale or repeated frame"
            if observation is not None and fresh:
                observations.append(observation.camera_from_marker.tolist())
                timestamps.append(frame.timestamp)
                print(f"Reference samples: {len(observations)}/{args.frames}", flush=True)
            if not args.headless:
                view = cv2.cvtColor(frame.rgb, cv2.COLOR_RGB2BGR)
                if observation is not None:
                    cv2.aruco.drawDetectedMarkers(view, [observation.corners.astype(np.float32).reshape(1, 4, 2)],
                                                 np.array([[reference["marker_id"]]], dtype=np.int32))
                cv2.putText(view, f"{len(observations)}/{args.frames}: {status}", (8, 22),
                            cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 180, 255), 1)
                cv2.imshow("Reference calibration", view)
                if cv2.waitKey(1) & 255 in (27, ord('q')) or cv2.getWindowProperty(
                        "Reference calibration", cv2.WND_PROP_VISIBLE) < 1:
                    raise RuntimeError("reference calibration cancelled; configuration unchanged")
        serial = camera.serial
    result = solve_reference(meta, reference, observations)
    result.update(device_serial=serial, timestamps=timestamps, camera_from_reference_tags=observations,
                  solved_at=datetime.now(timezone.utc).isoformat())
    atomic_json(args.output, result)
    print(json.dumps({k: result[k] for k in ("pose_count", "repeatability_errors", "passed")}, indent=2))
    print(f"Result: {args.output}")
    if args.write_config:
        if load_reference(args.reference_config) != reference:
            raise ValueError("reference mounting settings changed during capture; configuration unchanged")
        print(f"Configuration updated; backup: {apply_result(args.config, result)}")
    return 0 if result["passed"] else 2


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="mode", required=True)
    reference = commands.add_parser("reference", help="camera-only calibration from a measured fixed reference tag")
    reference.add_argument("--config", type=Path, default=DEFAULT_HARDWARE_CONFIG_PATH)
    reference.add_argument("--reference-config", type=Path,
                           default=Path(__file__).parent / "config" / "reference_calibration_config.json")
    reference.add_argument("--output", type=Path, default=Path(__file__).with_name("reference_calibration_result.json"))
    reference.add_argument("--frames", type=int, default=30)
    reference.add_argument("--timeout", type=float, default=30.)
    reference.add_argument("--headless", action="store_true")
    reference.add_argument("--write-config", action="store_true")
    for mode in ("collect", "solve"):
        command = commands.add_parser(mode)
        command.add_argument("--config", type=Path, default=DEFAULT_HARDWARE_CONFIG_PATH)
        command.add_argument("--samples", type=Path, default=DEFAULT_SAMPLES)
        if mode == "collect":
            command.add_argument("--skip-tag-stability-check", action="store_true",
                                 help="skip only the across-frame tool-tag pose stability check; record deviations")
            command.add_argument("--append", action="store_true", help="resume an existing compatible samples file")
            command.add_argument("--robot-ip", help="read-only connection address override")
        else:
            command.add_argument("--output", type=Path, default=DEFAULT_RESULT)
            command.add_argument("--write-config", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.mode == "reference":
            return reference_calibration(args)
        if args.mode == "collect":
            collect(args)
        else:
            if args.output.resolve() in (args.config.resolve(), args.samples.resolve()):
                raise ValueError("result output must differ from config and samples")
            with args.samples.open(encoding="utf-8") as source:
                result = solve(json.load(source))
            result["solved_at"] = datetime.now(timezone.utc).isoformat()
            atomic_json(args.output, result)
            print(json.dumps({k: result[k] for k in ("pose_count", "validation_errors", "fit_errors", "passed")}, indent=2))
            print(f"Result: {args.output}")
            if args.write_config:
                print(f"Configuration updated; backup: {apply_result(args.config, result)}")
            return 0 if result["passed"] else 2
        return 0
    except KeyboardInterrupt:
        print("Stopped; completed captures remain in the samples file.")
        return 130
    except (OSError, ValueError, RuntimeError, ImportError, KeyError, TypeError, cv2.error) as error:
        print(f"Calibration failed: {error}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
