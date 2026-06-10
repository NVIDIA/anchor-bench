from __future__ import annotations

import argparse
from typing import Any


def launch_isaac_sim(headless: bool, enable_cameras: bool = False) -> tuple[Any, Any]:
    """Launch the Isaac Sim AppLauncher with a controlled flag set.

    Returns ``(app_launcher, simulation_app)``. Imports Isaac Lab on call so
    the helper stays host-safe at module load time.
    """
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(add_help=False)
    AppLauncher.add_app_launcher_args(parser)
    launcher_args = ["--headless"] if headless else []
    app_args = parser.parse_args(launcher_args)
    app_args.enable_cameras = enable_cameras
    app_launcher = AppLauncher(app_args)
    return app_launcher, app_launcher.app
