"""Read RealSense factory calibration and update hardware config; no robot connection."""
import argparse
import copy
import json
from pathlib import Path
import time

import numpy as np

from hardware.calibrate_camera import atomic_json
from hardware.hardware_config import DEFAULT_HARDWARE_CONFIG_PATH
from hardware.hardware_realsense import RealSenseD435i


def select_serial(devices, configured, requested=None):
    serials = [device['serial'] for device in devices]
    if requested is not None:
        if requested not in serials:
            raise ValueError(f'Requested camera {requested} not connected; found {serials}')
        return requested
    if not serials:
        raise RuntimeError('No RealSense device connected')
    if len(serials) == 1:
        return serials[0]
    if configured in serials:
        return configured
    raise ValueError(f'Multiple cameras connected: {serials}; specify --serial')


def update_config(path, *, serial=None, dry_run=False, sdk=None, camera_factory=RealSenseD435i):
    path = Path(path).resolve()
    original = path.read_text(encoding='utf-8')
    values = json.loads(original)
    if values['camera'].get('factory') != 'hardware.hardware_realsense:create_camera':
        raise ValueError('This script requires the hardware_realsense camera adapter')
    if sdk is None:
        import pyrealsense2 as sdk
    context = sdk.context()
    devices = [{'name': d.get_info(sdk.camera_info.name),
                'serial': d.get_info(sdk.camera_info.serial_number)} for d in context.query_devices()]
    previous = values['camera']['options'].get('serial')
    selected = select_serial(devices, previous, serial)
    settings = copy.deepcopy(values['camera'])
    settings['options']['serial'] = selected
    settings['depth_scale'] = None
    # Intrinsics depend on stream profile, not exposure; don't apply the old
    # camera's exposure settings or wait for image timestamp convergence.
    settings['options']['warmup_frames'] = 0
    for key in ('rgb_auto_exposure', 'rgb_exposure', 'rgb_gain', 'rgb_brightness'):
        settings['options'][key] = None
    camera = camera_factory(settings, sdk=sdk)
    try:
        report = camera.factory_calibration()
    finally:
        camera.close()
    calibration = report['camera']
    if report['device']['serial'] != selected:
        raise ValueError('Camera serial changed during inspection; config unchanged')
    if (calibration['width'], calibration['height']) != (settings['width'], settings['height']):
        raise ValueError('Camera resolution differs from requested profile; config unchanged')
    if calibration['distortion'] is None:
        raise ValueError('Factory distortion model is not OpenCV-compatible; config unchanged')
    matrix = np.asarray(calibration['intrinsics'], dtype=float)
    distortion = np.asarray(calibration['distortion'], dtype=float)
    scale = calibration['depth_scale']
    if (matrix.shape != (3, 3) or not np.isfinite(matrix).all()
            or matrix[0, 0] <= 0 or matrix[1, 1] <= 0
            or not np.allclose(matrix[2], [0, 0, 1])
            or distortion.shape != (5,) or not np.isfinite(distortion).all()
            or not np.isfinite(scale) or scale <= 0):
        raise ValueError('Invalid factory calibration; config unchanged')
    backup = None
    if not dry_run:
        if path.read_text(encoding='utf-8') != original:
            raise RuntimeError('Config changed during inspection; refusing to overwrite it')
        archive = (path.parent.parent if path.parent.name == 'config' else path.parent) / 'old_version'
        archive.mkdir(parents=True, exist_ok=True)
        backup = archive / f'{path.name}.before-factory-intrinsics-{time.time_ns()}.bak'
        with backup.open('x', encoding='utf-8') as stream:
            stream.write(original)
        for key in ('intrinsics', 'distortion', 'depth_scale'):
            values['camera'][key] = calibration[key]
        values['camera']['options']['serial'] = selected
        atomic_json(path, values)
    return report, backup, previous


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=DEFAULT_HARDWARE_CONFIG_PATH)
    parser.add_argument('--serial', help='Select a connected camera explicitly')
    parser.add_argument('--dry-run', action='store_true', help='Inspect without saving or making a backup')
    args = parser.parse_args(argv)
    try:
        report, backup, previous = update_config(args.config, serial=args.serial, dry_run=args.dry_run)
    except (OSError, ValueError, RuntimeError, KeyError, ImportError) as error:
        parser.exit(2, f'Error: {error}\n')
    print(json.dumps(report, indent=2))
    print('Dry run; config unchanged.' if args.dry_run else f'Saved: {args.config.resolve()}\nBackup: {backup}')
    if previous != report['device']['serial']:
        print(f'Camera changed: {previous} -> {report["device"]["serial"]}')
    print('Robot transforms were NOT recalibrated. For a replacement or moved camera, '
          'collect new camera-to-base calibration before servo; do not append old-camera samples.')


if __name__ == '__main__':
    main()
