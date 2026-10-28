"""Camera-only ArUco check: no robot connection, commands, or world calibration.

Run normally for a live window (Q/Esc closes it), or use --headless --frames 60.
Detection uses DICT_4X4_50 and does not require the physical marker size.
Reported depth is optical-axis depth at the tag center, not a calibrated pose.
"""

import argparse
import json
from pathlib import Path
import time

import cv2
import numpy as np

from hardware_realsense import RealSenseD435i


LABELS = {0: "cube top", 1: "tool", 2: "cube +Y", 3: "cube +X",
          4: "cube -Y", 5: "cube -X", 6: "cube bottom"}


def annotate(sample, detector, depth_scale):
    view = cv2.cvtColor(sample.rgb, cv2.COLOR_RGB2BGR)
    corners, ids, _ = detector.detectMarkers(cv2.cvtColor(view, cv2.COLOR_BGR2GRAY))
    readings = {}
    if ids is not None:
        cv2.aruco.drawDetectedMarkers(view, corners, ids)
        height, width = sample.depth.shape
        for quad, marker_id in zip(corners, ids.reshape(-1)):
            marker_id = int(marker_id)
            x, y = np.rint(quad.reshape(4, 2).mean(axis=0)).astype(int)
            patch = sample.depth[max(0, y - 2):min(height, y + 3),
                                 max(0, x - 2):min(width, x + 3)]
            valid = patch[np.isfinite(patch) & (patch > 0)]
            depth = float(np.median(valid) * depth_scale) if valid.size else None
            readings[marker_id] = depth
            text = f"{LABELS.get(marker_id, 'other')} | " + (f"Z {depth:.3f} m" if depth else "no depth")
            text_width = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, .45, 1)[0][0]
            location = (max(0, min(x, width - text_width - 4)), max(15, min(y + 20, height - 8)))
            cv2.putText(view, text, location, cv2.FONT_HERSHEY_SIMPLEX, .45, (0, 0, 0), 3)
            cv2.putText(view, text, location, cv2.FONT_HERSHEY_SIMPLEX, .45, (255, 255, 255), 1)
    return view, readings


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).parent / "config" / "hardware_config.json")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--frames", type=int, default=0, help="stop after N frames; 0 runs until Q/Esc")
    args = parser.parse_args(argv)
    if args.frames < 0 or (args.headless and args.frames == 0):
        parser.error("use a positive --frames count with --headless; frame count cannot be negative")
    camera = None
    window_open = False
    window = "RealSense - ArUco camera test"
    try:
        with args.config.open(encoding="utf-8") as source:
            settings = json.load(source)["camera"]
        parameters = cv2.aruco.DetectorParameters()
        parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        detector = cv2.aruco.ArucoDetector(
            cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50), parameters)
        camera = RealSenseD435i(settings)
        print(f"Camera: {camera.device_name} | serial {camera.serial}", flush=True)
        print("Expected IDs: tool 1; cube 0, 2, 3, 4, 5, 6. Q/Esc exits.", flush=True)
        if not args.headless:
            cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE)
            window_open = True
        seen, count, last_report = set(), 0, -float("inf")
        while args.frames == 0 or count < args.frames:
            sample = camera.capture()
            view, readings = annotate(sample, detector, camera.depth_scale)
            count += 1
            seen.update(readings)
            now = time.monotonic()
            if now - last_report >= 1:
                details = ", ".join(f"ID {i} ({LABELS.get(i, 'other')}): "
                                    + (f"Z={z:.3f}m" if z is not None else "no depth")
                                    for i, z in sorted(readings.items()))
                print(details or "No markers detected in this frame.", flush=True)
                last_report = now
            if window_open:
                cv2.imshow(window, view)
                if cv2.waitKey(1) & 255 in (27, ord("q"), ord("Q")):
                    break
                if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                    break
        print(f"Read {count} RGB-D frames. IDs seen: {sorted(seen) or 'none'}.")
        return 0
    except KeyboardInterrupt:
        print("Camera test stopped.")
        return 130
    except (OSError, ValueError, RuntimeError, ImportError, KeyError, cv2.error) as error:
        print(f"Camera test failed: {error}")
        return 1
    finally:
        try:
            if camera is not None:
                camera.close()
        finally:
            if window_open:
                cv2.destroyAllWindows()


if __name__ == "__main__":
    raise SystemExit(main())
