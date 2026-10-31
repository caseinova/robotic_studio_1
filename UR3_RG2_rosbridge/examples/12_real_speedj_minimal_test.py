import time

import numpy as np

from rtde_control import (
    RTDEControlInterface as RTDEControl
)

from rtde_receive import (
    RTDEReceiveInterface as RTDEReceive
)


# ============================================================
# Robot
# ============================================================

ROBOT_IP = "192.168.0.191"


# ============================================================
# Very conservative first-motion test
# ============================================================

CONTROL_FREQUENCY = 500.0
DT = 1.0 / CONTROL_FREQUENCY

TEST_DURATION = 5.0

TEST_SPEED = 0.02          # rad/s
ACCELERATION = 0.10        # rad/s^2


# ============================================================
# Command
#
# Joint order:
#
# 0 shoulder_pan
# 1 shoulder_lift
# 2 elbow
# 3 wrist_1
# 4 wrist_2
# 5 wrist_3
# ============================================================

qd_command = np.zeros(
    6,
    dtype=float,
)

qd_command[5] = TEST_SPEED


# ============================================================
# Connect
# ============================================================

print()
print("==========================================")
print("UR3e MINIMAL speedJ TEST")
print("==========================================")

print()
print(f"Robot IP: {ROBOT_IP}")

print()
print("Connecting RTDE receive...")

rtde_r = RTDEReceive(
    ROBOT_IP
)


rtde_c = None


try:

    # ========================================================
    # Read state BEFORE control connection
    # ========================================================

    q_initial = np.asarray(
        rtde_r.getActualQ(),
        dtype=float,
    )

    qd_initial = np.asarray(
        rtde_r.getActualQd(),
        dtype=float,
    )


    print()
    print("Initial q [deg]:")

    print(
        np.round(
            np.rad2deg(q_initial),
            4,
        )
    )


    print()
    print("Initial qd [rad/s]:")

    print(
        np.round(
            qd_initial,
            6,
        )
    )


    # ========================================================
    # Basic sanity check
    # ========================================================

    if not np.all(
        np.isfinite(q_initial)
    ):

        raise RuntimeError(
            "Invalid initial joint positions."
        )


    if np.max(
        np.abs(qd_initial)
    ) > 0.01:

        raise RuntimeError(
            "Robot is already moving. "
            "Test aborted."
        )


    # ========================================================
    # Connect control interface
    # ========================================================

    print()
    print("Connecting RTDE control...")

    rtde_c = RTDEControl(
        ROBOT_IP
    )


    print(
        "RTDE control connected."
    )


    # ========================================================
    # Expected movement
    # ========================================================

    expected_delta = (
        TEST_SPEED
        * TEST_DURATION
    )


    print()
    print("Command:")

    print(
        f"wrist_3 = +{TEST_SPEED:.3f} rad/s"
    )

    print(
        f"duration = {TEST_DURATION:.3f} s"
    )

    print(
        f"expected change = "
        f"{expected_delta:.5f} rad "
        f"({np.rad2deg(expected_delta):.3f} deg)"
    )


    # ========================================================
    # speedJ loop
    # ========================================================

    cycles = int(
        CONTROL_FREQUENCY
        * TEST_DURATION
    )


    print()
    print("Starting motion...")


    for _ in range(cycles):

        t_start = (
            rtde_c.initPeriod()
        )


        rtde_c.speedJ(
            qd_command.tolist(),
            ACCELERATION,
            DT,
        )


        rtde_c.waitPeriod(
            t_start
        )


    # ========================================================
    # Stop velocity mode
    # ========================================================

    rtde_c.speedStop()


    # Give feedback a moment to settle
    time.sleep(
        0.2
    )


    # ========================================================
    # Read final state
    # ========================================================

    q_final = np.asarray(
        rtde_r.getActualQ(),
        dtype=float,
    )


    qd_final = np.asarray(
        rtde_r.getActualQd(),
        dtype=float,
    )


    delta_q = (
        q_final
        - q_initial
    )


    print()
    print("==========================================")
    print("RESULT")
    print("==========================================")


    print()
    print("Final q [deg]:")

    print(
        np.round(
            np.rad2deg(q_final),
            4,
        )
    )


    print()
    print("Joint change [deg]:")

    print(
        np.round(
            np.rad2deg(delta_q),
            4,
        )
    )


    print()
    print("Final qd [rad/s]:")

    print(
        np.round(
            qd_final,
            6,
        )
    )


    print()
    print(
        "Expected wrist_3 change:"
    )

    print(
        f"{np.rad2deg(expected_delta):.3f} deg"
    )


    print()
    print(
        "Actual wrist_3 change:"
    )

    print(
        f"{np.rad2deg(delta_q[5]):.3f} deg"
    )


    print()
    print(
        "MINIMAL MOTION TEST COMPLETED"
    )


except KeyboardInterrupt:

    print()
    print(
        "Keyboard interrupt received."
    )


finally:

    # ========================================================
    # Always attempt to stop motion
    # ========================================================

    if rtde_c is not None:

        try:

            rtde_c.speedStop()

        except Exception:

            pass


        try:

            rtde_c.stopScript()

        except Exception:

            pass


    try:

        rtde_r.disconnect()

    except Exception:

        pass


    print()
    print(
        "RTDE connections closed."
    )