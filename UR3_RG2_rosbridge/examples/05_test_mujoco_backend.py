from pathlib import Path
import sys

import numpy as np


# ============================================================
# Add project root to Python import path
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


from robot.mujoco_backend import (
    MuJoCoUR3e
)

# ============================================================
# Model path
# ============================================================

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "ur3e_official"
    / "ur3e_velocity.xml"
)


# ============================================================
# Create robot backend
# ============================================================

robot = MuJoCoUR3e(
    model_path=MODEL_PATH,
    control_frequency=100.0,
    show_viewer=True,
    realtime=True,
)


try:

    # ========================================================
    # Initial feedback
    # ========================================================

    q_initial = (
        robot.get_joint_positions()
    )


    print()
    print("==========================================")
    print("MuJoCo UR3e Backend Test")
    print("==========================================")


    print()
    print("Initial q [rad]:")

    print(
        np.round(
            q_initial,
            5,
        )
    )


    # ========================================================
    # Test command
    #
    # Wrist 3 only
    # +0.05 rad/s
    # for 2 seconds
    # ========================================================

    CONTROL_FREQUENCY = 100.0
    TEST_DURATION = 2.0


    number_of_cycles = int(
        CONTROL_FREQUENCY
        * TEST_DURATION
    )


    print()
    print(
        "Commanding wrist_3:"
    )

    print(
        "+0.05 rad/s for 2 seconds"
    )


    for cycle in range(
        number_of_cycles
    ):

        qd = np.zeros(
            6,
            dtype=float,
        )

        qd[5] = 0.05


        robot.set_joint_velocity(
            qd
        )


        if cycle % 20 == 0:

            q = (
                robot
                .get_joint_positions()
            )

            qd_actual = (
                robot
                .get_joint_velocities()
            )


            print(
                f"t={cycle / CONTROL_FREQUENCY:4.2f}s "
                f"q6={q[5]: .4f} "
                f"qd6={qd_actual[5]: .4f}"
            )


    # ========================================================
    # Stop
    # ========================================================

    robot.stop()


    # Keep simulator running for 1 second
    # with zero velocity command

    for _ in range(100):

        robot.set_joint_velocity(
            np.zeros(6)
        )


    # ========================================================
    # Final state
    # ========================================================

    q_final = (
        robot.get_joint_positions()
    )


    delta_q = (
        q_final
        - q_initial
    )


    print()
    print("Initial q:")
    print(
        np.round(
            q_initial,
            5,
        )
    )


    print()
    print("Final q:")
    print(
        np.round(
            q_final,
            5,
        )
    )


    print()
    print("Delta q:")
    print(
        np.round(
            delta_q,
            5,
        )
    )


    print()
    print(
        "Wrist 3 delta:"
    )

    print(
        f"{delta_q[5]:.5f} rad"
    )

    print(
        f"{np.rad2deg(delta_q[5]):.3f} deg"
    )


except KeyboardInterrupt:

    print()
    print(
        "Operator requested stop."
    )


finally:

    print()
    print(
        "Stopping and closing simulator..."
    )

    robot.stop()

    robot.close()