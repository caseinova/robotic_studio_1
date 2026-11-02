"""Resize only the current process's MuJoCo window, never other applications."""
import ctypes as c
from ctypes.util import find_library
import os
import sys

from simulation.viewer import window_rectangle


def place_window(settings):
    if sys.platform == 'win32':
        return _windows(settings)
    if sys.platform.startswith('linux') and os.environ.get('DISPLAY'):
        return _x11(settings)
    raise RuntimeError('automatic placement requires Windows or Linux X11/XWayland')


def _windows(settings):
    from ctypes import wintypes as w
    user = c.WinDLL('user32', use_last_error=True)
    callback_type = c.WINFUNCTYPE(w.BOOL, w.HWND, w.LPARAM)
    user.EnumWindows.argtypes = [callback_type, w.LPARAM]
    user.GetWindowThreadProcessId.argtypes = [w.HWND, c.POINTER(w.DWORD)]
    user.GetWindowTextW.argtypes = [w.HWND, w.LPWSTR, c.c_int]
    user.ShowWindow.argtypes = [w.HWND, c.c_int]
    user.SetWindowPos.argtypes = [w.HWND, w.HWND, c.c_int, c.c_int, c.c_int, c.c_int, w.UINT]
    user.SystemParametersInfoW.argtypes = [w.UINT, w.UINT, c.c_void_p, w.UINT]
    found = []

    @callback_type
    def visit(handle, _):
        pid = w.DWORD()
        user.GetWindowThreadProcessId(handle, c.byref(pid))
        title = c.create_unicode_buffer(512)
        user.GetWindowTextW(handle, title, len(title))
        if pid.value == os.getpid() and 'MuJoCo' in title.value:
            found.append(handle)
        return True

    user.EnumWindows(visit, 0)
    if not found:
        return False
    area = w.RECT()
    if not user.SystemParametersInfoW(48, 0, c.byref(area), 0):  # SPI_GETWORKAREA
        raise OSError('cannot read desktop work area')
    rect = window_rectangle(settings, (area.left, area.top, area.right-area.left, area.bottom-area.top))
    for handle in found:
        user.ShowWindow(handle, 9)  # SW_RESTORE
        if not user.SetWindowPos(handle, None, *rect, 0x14):  # preserve Z-order and activation
            raise OSError('cannot set MuJoCo window geometry')
    return True


def _x11(settings):
    library = find_library('X11')
    if not library:
        raise OSError('libX11 unavailable')
    x = c.CDLL(library)
    display_t, window_t, atom_t = c.c_void_p, c.c_ulong, c.c_ulong

    def function(name, result, *args):
        fn = getattr(x, name)
        fn.restype, fn.argtypes = result, list(args)
        return fn

    open_display = function('XOpenDisplay', display_t, c.c_char_p)
    close_display = function('XCloseDisplay', c.c_int, display_t)
    root_window = function('XDefaultRootWindow', window_t, display_t)
    atom = function('XInternAtom', atom_t, display_t, c.c_char_p, c.c_int)
    get_property = function('XGetWindowProperty', c.c_int, display_t, window_t, atom_t,
        c.c_long, c.c_long, c.c_int, atom_t, c.POINTER(atom_t), c.POINTER(c.c_int),
        c.POINTER(c.c_ulong), c.POINTER(c.c_ulong), c.POINTER(c.POINTER(c.c_ubyte)))
    free = function('XFree', c.c_int, c.c_void_p)
    fetch_name = function('XFetchName', c.c_int, display_t, window_t, c.POINTER(c.c_char_p))
    move_resize = function('XMoveResizeWindow', c.c_int, display_t, window_t, c.c_int, c.c_int, c.c_uint, c.c_uint)
    flush = function('XFlush', c.c_int, display_t)

    class ClientMessage(c.Structure):
        _fields_ = [('type', c.c_int), ('serial', c.c_ulong), ('send_event', c.c_int),
                    ('display', display_t), ('window', window_t), ('message_type', atom_t),
                    ('format', c.c_int), ('data', c.c_long * 5)]

    class Event(c.Union):
        _fields_ = [('client', ClientMessage), ('padding', c.c_long * 24)]

    send = function('XSendEvent', c.c_int, display_t, window_t, c.c_int, c.c_long, c.POINTER(Event))
    display = open_display(None)
    if not display:
        raise OSError('cannot open X display')
    try:
        root = root_window(display)

        def prop(window, name):
            actual, fmt, count, remaining = atom_t(), c.c_int(), c.c_ulong(), c.c_ulong()
            data = c.POINTER(c.c_ubyte)()
            result = get_property(display, window, atom(display, name.encode(), 0), 0, 65536,
                0, 0, c.byref(actual), c.byref(fmt), c.byref(count), c.byref(remaining), c.byref(data))
            try:
                if result or fmt.value != 32 or not data:
                    return []
                return list(c.cast(data, c.POINTER(c.c_ulong))[:count.value])
            finally:
                if data:
                    free(data)

        windows = []
        for window in prop(root, '_NET_CLIENT_LIST'):
            if prop(window, '_NET_WM_PID') != [os.getpid()]:
                continue
            name = c.c_char_p()
            try:
                if fetch_name(display, window, c.byref(name)) and name.value and b'MuJoCo' in name.value:
                    windows.append(window)
            finally:
                if name:
                    free(name)
        if not windows:
            return False
        areas = prop(root, '_NET_WORKAREA')
        desktop = (prop(root, '_NET_CURRENT_DESKTOP') or [0])[0]
        area = areas[desktop*4:desktop*4+4]
        if len(area) != 4:
            raise RuntimeError('window manager does not expose desktop work area')
        rect = window_rectangle(settings, area)
        for window in windows:
            # Remove fullscreen and maximization using the window-manager protocol.
            for names in (('_NET_WM_STATE_FULLSCREEN',),
                          ('_NET_WM_STATE_MAXIMIZED_VERT', '_NET_WM_STATE_MAXIMIZED_HORZ')):
                event = Event()
                event.client.type = 33  # ClientMessage
                event.client.display = display
                event.client.window = window
                event.client.message_type = atom(display, b'_NET_WM_STATE', 0)
                event.client.format = 32
                event.client.data[0] = 0  # _NET_WM_STATE_REMOVE
                for i, name in enumerate(names, 1):
                    event.client.data[i] = atom(display, name.encode(), 0)
                event.client.data[3] = 1  # source: application
                send(display, root, 0, (1 << 20) | (1 << 19), c.byref(event))
            move_resize(display, window, *rect)
        flush(display)
        return True
    finally:
        close_display(display)
