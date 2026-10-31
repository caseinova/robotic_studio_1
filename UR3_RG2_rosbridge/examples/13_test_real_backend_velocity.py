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


from robot.real_ur_backend import (
    RealUR3e
)


# ============================================================
# Settings
# ============================================================

ROBOT_IP = "192.168.0.191"

CONTROL_FREQUENCY = 100.0

TEST_DURATION = 10.0

TEST_SPEED = 0.02


# ============================================================
# Create real backend
# ============================================================

robot = RealUR3e(
    robot_ip=ROBOT_IP,

    control_frequency=(
        CONTROL_FREQUENCY
    ),

    enable_control=True,

    # Backend absolute software cap
    max_joint_speed=0.05,

    acceleration=0.10,

    stop_acceleration=0.50,
)


try:

    # ========================================================
    # Initial state
    # ========================================================

    q_initial = (
        robot.get_joint_positions()
    )

    qd_initial = (
        robot.get_joint_velocities()
    )


    print()
    print(
        "=========================================="
    )

    print(
        "REAL UR3e BACKEND VELOCITY TEST"
    )

    print(
        "=========================================="
    )


    print()
    print(
        "Initial q [deg]:"
    )

    print(
        np.round(
            np.rad2deg(q_initial),
            4,
        )
    )


    print()
    print(
        "Initial qd [rad/s]:"
    )

    print(
        np.round(
            qd_initial,
            6,
        )
    )


    # ========================================================
    # Confirm robot is initially stationary
    # ========================================================

    if (
        np.max(
            np.abs(qd_initial)
        )
        > 0.01
    ):

        raise RuntimeError(
            "Robot is already moving. "
            "Test cancelled."
        )


    # ========================================================
    # Command
    #
    # wrist_3 only
    # ========================================================

    qd_command = np.zeros(
        6,
        dtype=float,
    )

    qd_command[2] = (
        TEST_SPEED
    )


    cycles = int(
        CONTROL_FREQUENCY
        * TEST_DURATION
    )


    print()
    print(
        "Commanding wrist_3:"
    )

    print(
        f"{TEST_SPEED:.3f} rad/s"
    )

    print(
        f"for {TEST_DURATION:.3f} s"
    )


    # ========================================================
    # Same interface used by MuJoCo
    # ========================================================

    for cycle in range(
        cycles
    ):

        robot.set_joint_velocity(
            qd_command
        )


        if cycle % 10 == 0:

            q = (
                robot.get_joint_positions()
            )

            qd_actual = (
                robot.get_joint_velocities()
            )


            print(
                f"t="
                f"{cycle / CONTROL_FREQUENCY:.2f}s "
                f"| q6="
                f"{np.rad2deg(q[5]):.3f} deg "
                f"| qd6="
                f"{qd_actual[5]:.4f} rad/s"
            )


    # ========================================================
    # Stop
    # ========================================================

    robot.stop()


    # ========================================================
    # Final state
    # ========================================================

    q_final = (
        robot.get_joint_positions()
    )


    qd_final = (
        robot.get_joint_velocities()
    )


    delta_q = (
        q_final
        - q_initial
    )


    print()
    print(
        "=========================================="
    )

    print(
        "RESULT"
    )

    print(
        "=========================================="
    )


    print()
    print(
        "Joint displacement [deg]:"
    )

    print(
        np.round(
            np.rad2deg(delta_q),
            4,
        )
    )


    print()
    print(
        "Final qd [rad/s]:"
    )

    print(
        np.round(
            qd_final,
            6,
        )
    )


    print()
    print(
        "wrist_3 displacement:"
    )

    print(
        f"{np.rad2deg(delta_q[5]):.4f} deg"
    )


except KeyboardInterrupt:

    print()
    print(
        "Operator stop requested."
    )


finally:

    robot.stop()
    robot.close()