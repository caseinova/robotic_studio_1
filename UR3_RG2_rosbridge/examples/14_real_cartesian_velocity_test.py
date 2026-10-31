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
# Imports
# ============================================================

from roboticstoolbox.models.URDF.URDFRobot import (
    URDFRobot
)

from robot.real_ur_backend import (
    RealUR3e
)

from control.cartesian_velocity import (
    damped_least_squares
)

from control.safety import (
    limit_cartesian_velocity,
    adaptive_damping,
    apply_joint_limit_protection,
    limit_joint_velocity,
)


# ============================================================
# Robot
# ============================================================

ROBOT_IP = "192.168.0.191"


# ============================================================
# First real Cartesian test
# ============================================================

CONTROL_FREQUENCY = 100.0
TEST_DURATION = 50.0

# PolyScope Base +X
# 1 mm/s
XD_COMMAND = np.array(
    [
        1000.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ],
    dtype=float,
)


# Application-level limits for FIRST test
MAX_LINEAR_SPEED = 0.002       # 2 mm/s
MAX_ANGULAR_SPEED = 0.05       # rad/s

MAX_JOINT_SPEED = 0.03         # rad/s
JOINT_LIMIT_MARGIN = np.deg2rad(10.0)

MIN_SIGMA_ALLOWED = 0.04


# Monitoring thresholds
MAX_TCP_DISPLACEMENT = 5   # 5 mm absolute abort threshold
MAX_ACTUAL_JOINT_SPEED = 0.05  # rad/s


# ============================================================
# Load official nominal UR3e model
# ============================================================

print()
print("Loading official UR3e model...")

kinematic_model = URDFRobot(
    "ur3e"
)

qlim = np.asarray(
    kinematic_model.qlim,
    dtype=float,
)


# ============================================================
# Create real backend
# ============================================================

robot = RealUR3e(
    robot_ip=ROBOT_IP,
    control_frequency=CONTROL_FREQUENCY,
    enable_control=True,

    # Hard backend limit
    max_joint_speed=MAX_JOINT_SPEED,

    acceleration=0.05,
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

    tcp_initial = (
        robot.get_tcp_pose()
    )


    print()
    print("==========================================")
    print("FIRST REAL CARTESIAN VELOCITY TEST")
    print("==========================================")

    print()
    print("Initial q [deg]:")

    print(
        np.round(
            np.rad2deg(q_initial),
            3,
        )
    )


    print()
    print("Initial actual TCP [m]:")

    print(
        np.round(
            tcp_initial[:3],
            6,
        )
    )


    # ========================================================
    # Robot must initially be stationary
    # ========================================================

    if (
        np.max(
            np.abs(qd_initial)
        )
        > 0.01
    ):

        raise RuntimeError(
            "Robot was already moving. "
            "Test aborted."
        )


    # ========================================================
    # Clamp Cartesian request
    # ========================================================

    xdot_safe = limit_cartesian_velocity(
        XD_COMMAND,
        max_linear_speed=MAX_LINEAR_SPEED,
        max_angular_speed=MAX_ANGULAR_SPEED,
    )


    print()
    print("Requested Cartesian velocity:")

    print(
        xdot_safe
    )

    print()
    print(
        "Command: PolyScope Base +X "
        "at 1 mm/s for 1 second"
    )


    # ========================================================
    # Main loop
    # ========================================================

    cycles = int(
        CONTROL_FREQUENCY
        * TEST_DURATION
    )


    for cycle in range(cycles):

        # ----------------------------------------------------
        # Actual feedback
        # ----------------------------------------------------

        q = (
            robot.get_joint_positions()
        )

        qd_actual = (
            robot.get_joint_velocities()
        )

        tcp_actual = (
            robot.get_tcp_pose()
        )


        # ----------------------------------------------------
        # Watch actual robot motion
        # ----------------------------------------------------

        tcp_delta = (
            tcp_actual[:3]
            - tcp_initial[:3]
        )

        tcp_distance = float(
            np.linalg.norm(
                tcp_delta
            )
        )


        if (
            tcp_distance
            > MAX_TCP_DISPLACEMENT
        ):

            raise RuntimeError(
                "TCP displacement exceeded "
                "5 mm software test limit."
            )


        if (
            np.max(
                np.abs(qd_actual)
            )
            > MAX_ACTUAL_JOINT_SPEED
        ):

            raise RuntimeError(
                "Actual joint velocity exceeded "
                "software test limit."
            )


        # ----------------------------------------------------
        # IMPORTANT:
        #
        # Use UR controller Base frame,
        # NOT default base_link.
        # ----------------------------------------------------

        J = (
            kinematic_model.jacob0(
                q,
                start="base",
                end="tool0",
            )
        )


        # ----------------------------------------------------
        # Singularity monitoring / adaptive damping
        # ----------------------------------------------------

        (
            damping,
            sigma_min,
        ) = adaptive_damping(
            J,
            minimum_damping=0.05,
            maximum_damping=0.20,
            singularity_threshold=0.08,
        )


        if sigma_min < MIN_SIGMA_ALLOWED:

            raise RuntimeError(
                "Robot configuration is too close "
                "to the software singularity limit."
            )


        # ----------------------------------------------------
        # Cartesian velocity -> qdot
        # ----------------------------------------------------

        qdot = (
            damped_least_squares(
                J=J,
                xdot=xdot_safe,
                damping=damping,
            )
        )


        # ----------------------------------------------------
        # Joint-limit protection
        # ----------------------------------------------------

        qdot = (
            apply_joint_limit_protection(
                q=q,
                qdot=qdot,
                qlim=qlim,
                margin=JOINT_LIMIT_MARGIN,
            )
        )


        # ----------------------------------------------------
        # qdot limit
        # ----------------------------------------------------

        qdot = (
            limit_joint_velocity(
                qdot,
                max_speed=MAX_JOINT_SPEED,
            )
        )


        # ----------------------------------------------------
        # Send to real robot
        # ----------------------------------------------------

        robot.set_joint_velocity(
            qdot
        )


        # ----------------------------------------------------
        # Diagnostics
        # ----------------------------------------------------

        if cycle % 10 == 0:

            print(
                f"t={cycle / CONTROL_FREQUENCY:.2f}s "
                f"| TCP dx="
                f"{tcp_delta[0] * 1000.0: .3f} mm "
                f"| dy="
                f"{tcp_delta[1] * 1000.0: .3f} mm "
                f"| dz="
                f"{tcp_delta[2] * 1000.0: .3f} mm "
                f"| sigma_min={sigma_min:.4f} "
                f"| max|qd|="
                f"{np.max(np.abs(qdot)):.4f}"
            )


    # ========================================================
    # Stop
    # ========================================================

    robot.stop()


    # ========================================================
    # Final state
    # ========================================================

    tcp_final = (
        robot.get_tcp_pose()
    )

    q_final = (
        robot.get_joint_positions()
    )


    displacement = (
        tcp_final[:3]
        - tcp_initial[:3]
    )


    print()
    print("==========================================")
    print("RESULT")
    print("==========================================")

    print()
    print("Actual TCP displacement [mm]:")

    print(
        np.round(
            displacement * 1000.0,
            4,
        )
    )


    print()
    print("Requested ideal displacement [mm]:")

    print(
        np.array(
            [1.0, 0.0, 0.0]
        )
    )


    print()
    print("Joint displacement [deg]:")

    print(
        np.round(
            np.rad2deg(
                q_final
                - q_initial
            ),
            4,
        )
    )


except KeyboardInterrupt:

    print()
    print("Operator stop requested.")


except Exception as exc:

    print()
    print("TEST ABORTED:")
    print(exc)


finally:

    # Always terminate speed mode
    robot.stop()
    robot.close()