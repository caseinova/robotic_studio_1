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
# Project imports
# ============================================================

from roboticstoolbox.models.URDF.URDFRobot import (
    URDFRobot
)

from robot.mujoco_backend import (
    MuJoCoUR3e
)

from control.cartesian_velocity import (
    damped_least_squares,
    limit_joint_velocity,
)


# ============================================================
# MuJoCo model
# ============================================================

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "ur3e_official"
    / "ur3e_velocity.xml"
)


# ============================================================
# Test settings
# ============================================================

CONTROL_FREQUENCY = 100.0

TEST_DURATION = 2.0


# Cartesian velocity
#
# Base-frame +X direction
# 5 mm/s
#
# [vx, vy, vz, wx, wy, wz]

XD_DESIRED = np.array(
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


DAMPING = 0.05


# Conservative simulation joint velocity limit
MAX_JOINT_SPEED = 0.20


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
# Load official Robotics Toolbox UR3e model
# ============================================================

print()
print("Loading Robotics Toolbox UR3e...")

kinematic_model = URDFRobot(
    "ur3e"
)


# ============================================================
# Create MuJoCo backend
# ============================================================

print(
    "Loading MuJoCo UR3e..."
)

robot = MuJoCoUR3e(
    model_path=MODEL_PATH,
    control_frequency=CONTROL_FREQUENCY,
    show_viewer=True,
    realtime=True,
)


try:

    # ========================================================
    # Set test start configuration
    # ========================================================

    robot.reset(
        q=Q_START
    )


    q = (
        robot.get_joint_positions()
    )


    # ========================================================
    # Initial TCP pose
    # ========================================================

    T_initial = (
        kinematic_model.fkine(
            q,
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
        "Cartesian Velocity Control Test"
    )

    print(
        "=========================================="
    )


    print()
    print(
        "Initial joint configuration [deg]:"
    )

    print(
        np.round(
            np.rad2deg(q),
            3,
        )
    )


    print()
    print(
        "Initial tool0 position [m]:"
    )

    print(
        np.round(
            p_initial,
            6,
        )
    )


    print()
    print(
        "Desired Cartesian velocity:"
    )

    print(
        XD_DESIRED
    )


    print()
    print(
        "Command:"
    )

    print(
        "Base +X at 5 mm/s"
    )


    # ========================================================
    # Main control loop
    # ========================================================

    number_of_cycles = int(
        CONTROL_FREQUENCY
        * TEST_DURATION
    )


    for cycle in range(
        number_of_cycles
    ):

        # ----------------------------------------------------
        # 1. Read actual simulated joint configuration
        # ----------------------------------------------------

        q = (
            robot.get_joint_positions()
        )


        # ----------------------------------------------------
        # 2. Compute current Jacobian
        #
        # tool0 velocity expressed in base frame
        # ----------------------------------------------------

        J = (
            kinematic_model.jacob0(
                q,
                end="tool0",
            )
        )


        # ----------------------------------------------------
        # 3. Compute qdot with Damped Least Squares
        # ----------------------------------------------------

        qdot = damped_least_squares(
            J=J,
            xdot=XD_DESIRED,
            damping=DAMPING,
        )


        # ----------------------------------------------------
        # 4. Apply application-level velocity limit
        # ----------------------------------------------------

        qdot = limit_joint_velocity(
            qdot=qdot,
            max_speed=MAX_JOINT_SPEED,
        )


        # ----------------------------------------------------
        # 5. Send exactly that qdot to MuJoCo backend
        # ----------------------------------------------------

        robot.set_joint_velocity(
            qdot
        )


        # ----------------------------------------------------
        # Diagnostics every ~0.2 s
        # ----------------------------------------------------

        if cycle % 20 == 0:

            # Current TCP pose
            T_now = (
                kinematic_model.fkine(
                    q,
                    end="tool0",
                )
            )

            p_now = np.array(
                T_now.t,
                dtype=float,
            )


            # Singular values are useful diagnostics
            singular_values = (
                np.linalg.svd(
                    J,
                    compute_uv=False,
                )
            )


            minimum_sv = float(
                np.min(
                    singular_values
                )
            )


            print(
                f"t={cycle / CONTROL_FREQUENCY:4.2f}s "
                f"| X={p_now[0]: .4f} m "
                f"| max|qd|={np.max(np.abs(qdot)): .4f} rad/s "
                f"| sigma_min={minimum_sv:.5f}"
            )


    # ========================================================
    # Stop
    # ========================================================

    robot.stop()


    # Hold for one second
    for _ in range(
        int(CONTROL_FREQUENCY)
    ):

        robot.set_joint_velocity(
            np.zeros(6)
        )


    # ========================================================
    # Final position
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


    p_final = np.array(
        T_final.t,
        dtype=float,
    )


    displacement = (
        p_final
        - p_initial
    )


    # ========================================================
    # Result
    # ========================================================

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
        "Initial tool0 [m]:"
    )

    print(
        np.round(
            p_initial,
            6,
        )
    )


    print()
    print(
        "Final tool0 [m]:"
    )

    print(
        np.round(
            p_final,
            6,
        )
    )


    print()
    print(
        "Displacement [mm]:"
    )

    print(
        np.round(
            displacement * 1000.0,
            3,
        )
    )


    print()
    print(
        "Expected ideal +X displacement:"
    )

    expected_x = (
        XD_DESIRED[0]
        * TEST_DURATION
    )


    print(
        f"{expected_x * 1000.0:.3f} mm"
    )


    print()
    print(
        "Final joint configuration [deg]:"
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