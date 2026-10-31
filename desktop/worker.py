"""Run existing entry points with a cross-platform cooperative stop request."""
import argparse
import _thread
import importlib
from pathlib import Path
import threading

from desktop.profiles import arguments


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--validate', action='store_true')
    args = parser.parse_args(argv)
    stop_file = args.directory / 'stop.request'
    finished = threading.Event()

    def monitor():
        while not finished.wait(.1):
            if stop_file.exists():
                _thread.interrupt_main()
                return

    watcher = threading.Thread(target=monitor, daemon=True)
    try:
        if stop_file.exists():
            return 130
        watcher.start()
        module, command = arguments(args.directory, args.validate)
        return importlib.import_module(module).main(command)
    except KeyboardInterrupt:
        print('Stop requested; experiment cleanup completed.', flush=True)
        return 130
    finally:
        finished.set()
        if watcher.is_alive():
            watcher.join(timeout=1.)


if __name__ == '__main__':
    raise SystemExit(main())
