"""Launch the desktop experiment manager."""


def main():
    import argparse
    argparse.ArgumentParser(description=__doc__).parse_args()
    try:
        from desktop.window import main as launch
    except ModuleNotFoundError as error:
        if error.name == 'PySide6':
            raise SystemExit('Install desktop dependencies: python -m pip install -r requirements-gui.txt') from error
        raise
    return launch()


if __name__ == '__main__':
    raise SystemExit(main())
