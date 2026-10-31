from pathlib import Path
import sys

import numpy as np


# ============================================================
# Project root
# ============================================================

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[1]
)

if str(PROJECT_ROOT) not in sys.path:

    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


# ============================================================
# Project import
# ============================================================

from robot.real_ur_backend import (
    RealUR3e
)


# ============================================================
# Robot IP
# ============================================================

ROBOT_IP = "192.168.0.191"


# ============================================================
# Connect
# ============================================================

robot = RealUR3e(
    robot_ip=ROBOT_IP
)


try:

    q = (
        robot.get_joint_positions()
    )

    qd = (
        robot.get_joint_velocities()
    )

    tcp = (
        robot.get_tcp_pose()
    )


    print()
    print(
        "=========================================="
    )

    print(
        "REAL UR3e BACKEND TEST"
    )

    print(
        "=========================================="
    )


    print()
    print(
        "Joint positions [rad]:"
    )

    print(
        np.round(
            q,
            6,
        )
    )


    print()
    print(
        "Joint positions [deg]:"
    )

    print(
        np.round(
            np.rad2deg(q),
            3,
        )
    )


    print()
    print(
        "Joint velocities [rad/s]:"
    )

    print(
        np.round(
            qd,
            6,
        )
    )


    print()
    print(
        "Actual TCP pose:"
    )

    print(
        np.round(
            tcp,
            6,
        )
    )


finally:

    robot.close()
    