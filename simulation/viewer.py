"""Shared passive-viewer settings and best-effort native window placement."""
import math
import time
import warnings

DEFAULT_VIEWER = dict(show_left_ui=False, show_right_ui=False, resize_window=True,
                      screen_fraction=.9, width=None, height=None, x=None, y=None,
                      show_estimates=False)


def viewer_settings(values):
    if not isinstance(values, dict) or set(values) - set(DEFAULT_VIEWER):
        raise ValueError('viewer must contain only supported window settings')
    result = dict(DEFAULT_VIEWER, **values)
    for key in ('show_left_ui', 'show_right_ui', 'resize_window', 'show_estimates'):
        if type(result[key]) is not bool:
            raise ValueError(f'viewer.{key} must be boolean')
    fraction = result['screen_fraction']
    if (isinstance(fraction, bool) or not isinstance(fraction, (int, float))
            or not math.isfinite(fraction) or not 0 < fraction <= 1):
        raise ValueError('viewer.screen_fraction must be in (0, 1]')
    for key in ('width', 'height', 'x', 'y'):
        if result[key] is not None and (type(result[key]) is not int
                or abs(result[key]) > 100000 or (key in ('width', 'height') and result[key] < 100)):
            raise ValueError(f'viewer.{key} must be a valid pixel integer or null')
    return result


def window_rectangle(settings, workarea):
    left, top, available_width, available_height = workarea
    width = min(settings['width'] or int(available_width * settings['screen_fraction']), available_width)
    height = min(settings['height'] or int(available_height * settings['screen_fraction']), available_height)
    x = settings['x'] if settings['x'] is not None else left + (available_width - width) // 2
    y = settings['y'] if settings['y'] is not None else top + (available_height - height) // 2
    return x, y, max(1, width), max(1, height)


def launch_viewer(model, data, settings):
    import mujoco.viewer
    from simulation.window_layout import place_window
    settings = viewer_settings(settings)
    viewer = mujoco.viewer.launch_passive(model, data,
        show_left_ui=settings['show_left_ui'], show_right_ui=settings['show_right_ui'])
    try:
        with viewer.lock():
            viewer.cam.lookat[:] = (0., .16, .23)
            viewer.cam.distance = 1.1
            viewer.cam.azimuth = 225.
            viewer.cam.elevation = -18.
    except BaseException:
        viewer.close()
        raise
    if settings['resize_window']:
        try:
            deadline = time.monotonic() + 1.
            while not place_window(settings):
                if time.monotonic() >= deadline:
                    raise RuntimeError('MuJoCo window was not registered with the window manager')
                time.sleep(.05)
        except (OSError, RuntimeError, AttributeError) as exc:
            warnings.warn(f'Viewer window geometry unchanged: {exc}. Resize manually or set '
                          'viewer.resize_window=false.', RuntimeWarning)
        except BaseException:
            viewer.close()
            raise
    return viewer
