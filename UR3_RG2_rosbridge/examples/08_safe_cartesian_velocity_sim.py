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

from robot.mujoco_backend import (
    MuJoCoUR3e
)

from control.cartesian_velocity import (
    damped_least_squares,
)

from control.safety import (
    limit_cartesian_velocity,
    adaptive_damping,
    apply_joint_limit_protection,
    limit_joint_velocity,
)


# ============================================================
# Model
# ============================================================

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "ur3e_official"
    / "ur3e_velocity.xml"
)


# ============================================================
# Control settings
# ============================================================

CONTROL_FREQUENCY = 100.0

TEST_DURATION = 2.0


# ------------------------------------------------------------
# Desired Cartesian velocity
#
# +X 5 mm/s
# ------------------------------------------------------------

XD_COMMAND = np.array(
    [
        0.005,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ],
    dtype=float,
)


# ------------------------------------------------------------
# Application-level safety settings
# ------------------------------------------------------------

MAX_CARTESIAN_LINEAR_SPEED = 0.020
MAX_CARTESIAN_ANGULAR_SPEED = 0.20

MAX_JOINT_SPEED = 0.10

JOINT_LIMIT_MARGIN = np.deg2rad(
    10.0
)


# ============================================================
# Initial configuration
# ============================================================

Q_START = np.deg2rad(
    [
        0.0,
        -90.0,
        90.0,
        -90.0,
        -90.0,
        0.0,
    ]
)


# ============================================================
# Load official UR3e kinematic model
# ============================================================

print()
print(
    "Loading official UR3e kinematic model..."
)

kinematic_model = URDFRobot(
    "ur3e"
)


# ============================================================
# Get official joint limits from model
# ============================================================

qlim = np.asarray(
    kinematic_model.qlim,
    dtype=float,
)


print()
print("Joint limits from RTB model [deg]:")

print(
    np.round(
        np.rad2deg(qlim),
        2,
    )
)


# ============================================================
# MuJoCo backend
# ============================================================

print()
print(
    "Loading MuJoCo backend..."
)

robot = MuJoCoUR3e(
    model_path=MODEL_PATH,
    control_frequency=CONTROL_FREQUENCY,
    show_viewer=True,
    realtime=True,
)


try:

    # ========================================================
    # Reset to test pose
    # ========================================================

    robot.reset(
        q=Q_START
    )


    q_initial = (
        robot.get_joint_positions()
    )


    T_initial = (
        kinematic_model.fkine(
            q_initial,
            end="tool0",
        )
    )


    p_initial = np.array(
        T_initial.t,
        dtype=float,
    )


    print()
    print(
        "=========================================="
    )

    print(
        "SAFE CARTESIAN VELOCITY TEST"
    )

    print(
        "=========================================="
    )


    print()
    print(
        "Requested Cartesian velocity:"
    )

    print(
        XD_COMMAND
    )


    # ========================================================
    # Apply Cartesian speed limit once
    # ========================================================

    xdot_safe = (
        limit_cartesian_velocity(
            XD_COMMAND,
            max_linear_speed=(
                MAX_CARTESIAN_LINEAR_SPEED
            ),
            max_angular_speed=(
                MAX_CARTESIAN_ANGULAR_SPEED
            ),
        )
    )


    print()
    print(
        "Safe Cartesian velocity:"
    )

    print(
        xdot_safe
    )


    # ========================================================
    # Control loop
    # ========================================================

    cycles = int(
        CONTROL_FREQUENCY
        * TEST_DURATION
    )


    for cycle in range(
        cycles
    ):

        # ----------------------------------------------------
        # 1. Actual joint feedback
        # ----------------------------------------------------

        q = (
            robot.get_joint_positions()
        )


        # ----------------------------------------------------
        # 2. Current Jacobian
        # ----------------------------------------------------

        J = (
            kinematic_model.jacob0(
                q,
                end="tool0",
            )
        )


        # ----------------------------------------------------
        # 3. Adaptive damping
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


        # ----------------------------------------------------
        # 4. Cartesian -> qdot
        # ----------------------------------------------------

        qdot = (
            damped_least_squares(
                J=J,
                xdot=xdot_safe,
                damping=damping,
            )
        )


        # ----------------------------------------------------
        # 5. Protect joint limits
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
        # 6. Joint velocity limit
        # ----------------------------------------------------

        qdot = (
            limit_joint_velocity(
                qdot,
                max_speed=MAX_JOINT_SPEED,
            )
        )


        # ----------------------------------------------------
        # 7. Send command
        # ----------------------------------------------------

        robot.set_joint_velocity(
            qdot
        )


        # ----------------------------------------------------
        # Diagnostics
        # ----------------------------------------------------

        if cycle % 20 == 0:

            T_now = (
                kinematic_model.fkine(
                    q,
                    end="tool0",
                )
            )

            p_now = np.asarray(
                T_now.t,
                dtype=float,
            )


            print(
                f"t={cycle / CONTROL_FREQUENCY:4.2f}s "
                f"| X={p_now[0]:.5f} "
                f"| sigma_min={sigma_min:.5f} "
                f"| damping={damping:.4f} "
                f"| max|qd|={np.max(np.abs(qdot)):.4f}"
            )


    # ========================================================
    # Stop
    # ========================================================

    robot.stop()


    # Hold position with zero velocity command
    for _ in range(
        int(CONTROL_FREQUENCY)
    ):

        robot.set_joint_velocity(
            np.zeros(6)
        )


    # ========================================================
    # Final result
    # ========================================================

    q_final = (
        robot.get_joint_positions()
    )


    T_final = (
        kinematic_model.fkine(
            q_final,
            end="tool0",
        )
    )


    p_final = np.asarray(
        T_final.t,
        dtype=float,
    )


    displacement = (
        p_final
        - p_initial
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
        "TCP displacement [mm]:"
    )

    print(
        np.round(
            displacement * 1000.0,
            3,
        )
    )


    expected = (
        XD_COMMAND[:3]
        * TEST_DURATION
        * 1000.0
    )


    print()
    print(
        "Ideal requested displacement [mm]:"
    )

    print(
        expected
    )


    print()
    print(
        "Final q [deg]:"
    )

    print(
        np.round(
            np.rad2deg(q_final),
            3,
        )
    )


except KeyboardInterrupt:

    print()
    print(
        "Operator requested stop."
    )


finally:

    robot.stop()
    robot.close()