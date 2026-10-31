import argparse
import time

import numpy as np

from rtde_receive import RTDEReceiveInterface as RTDEReceive


# ============================================================
# UR3e joint names
# ============================================================

JOINT_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow",
    "wrist_1",
    "wrist_2",
    "wrist_3",
]


# ============================================================
# Command-line arguments
# ============================================================

parser = argparse.ArgumentParser(
    description="Read-only UR3e RTDE connection test"
)

parser.add_argument(
    "--ip",
    required=True,
    help="UR3e robot IP address",
)

args = parser.parse_args()

ROBOT_IP = args.ip


# ============================================================
# Connect
#
# IMPORTANT:
# Only RTDEReceiveInterface is created.
#
# No RTDEControlInterface.
# No motion commands.
# ============================================================

print()
print("==========================================")
print("UR3e RTDE READ-ONLY TEST")
print("==========================================")

print()
print("Robot IP:")
print(ROBOT_IP)

print()
print("Connecting...")


rtde_r = RTDEReceive(
    ROBOT_IP
)


print()
print("Connected.")


try:

    # ========================================================
    # Read robot state
    # ========================================================

    q = np.asarray(
        rtde_r.getActualQ(),
        dtype=float,
    )

    qd = np.asarray(
        rtde_r.getActualQd(),
        dtype=float,
    )

    tcp = np.asarray(
        rtde_r.getActualTCPPose(),
        dtype=float,
    )


    # ========================================================
    # Validate returned sizes
    # ========================================================

    if q.shape != (6,):

        raise RuntimeError(
            f"Unexpected q shape: {q.shape}"
        )


    if qd.shape != (6,):

        raise RuntimeError(
            f"Unexpected qd shape: {qd.shape}"
        )


    if tcp.shape != (6,):

        raise RuntimeError(
            f"Unexpected TCP pose shape: {tcp.shape}"
        )


    # ========================================================
    # Joint positions
    # ========================================================

    print()
    print("==========================================")
    print("ACTUAL JOINT POSITIONS")
    print("==========================================")

    for i in range(6):

        print(
            f"{i}: {JOINT_NAMES[i]:14s} "
            f"{q[i]: .6f} rad   "
            f"{np.rad2deg(q[i]): .3f} deg"
        )


    # ========================================================
    # Joint velocities
    # ========================================================

    print()
    print("==========================================")
    print("ACTUAL JOINT VELOCITIES")
    print("==========================================")

    for i in range(6):

        print(
            f"{i}: {JOINT_NAMES[i]:14s} "
            f"{qd[i]: .6f} rad/s"
        )


    # ========================================================
    # TCP pose
    #
    # UR convention:
    #
    # [x, y, z, rx, ry, rz]
    #
    # x,y,z      -> metres
    # rx,ry,rz   -> rotation vector / axis-angle representation
    # ========================================================

    print()
    print("==========================================")
    print("ACTUAL TCP POSE")
    print("==========================================")

    print()
    print("Position [m]:")

    print(
        np.round(
            tcp[:3],
            6,
        )
    )

    print()
    print("Rotation vector [rad]:")

    print(
        np.round(
            tcp[3:],
            6,
        )
    )


    # ========================================================
    # Read continuously for 5 seconds
    #
    # Still READ ONLY.
    # ========================================================

    print()
    print("==========================================")
    print("LIVE READ TEST")
    print("==========================================")

    print()
    print("Reading for 5 seconds...")
    print("No motion commands are being sent.")
    print()


    start = time.perf_counter()

    last_print = -1.0


    while (
        time.perf_counter()
        - start
        < 5.0
    ):

        q_live = np.asarray(
            rtde_r.getActualQ(),
            dtype=float,
        )

        tcp_live = np.asarray(
            rtde_r.getActualTCPPose(),
            dtype=float,
        )


        elapsed = (
            time.perf_counter()
            - start
        )


        # Print about twice per second
        if (
            elapsed
            - last_print
            >= 0.5
        ):

            print(
                f"t={elapsed:4.1f}s "
                f"| q1={np.rad2deg(q_live[0]):8.3f} deg "
                f"| TCP="
                f"[{tcp_live[0]: .4f}, "
                f"{tcp_live[1]: .4f}, "
                f"{tcp_live[2]: .4f}] m"
            )

            last_print = elapsed


        time.sleep(
            0.01
        )


    print()
    print("==========================================")
    print("READ-ONLY TEST PASSED")
    print("==========================================")


except KeyboardInterrupt:

    print()
    print("Operator stopped test.")


finally:

    # Disconnect receive interface.
    #
    # No stop command is necessary because this program
    # never commanded robot motion.

    try:

        rtde_r.disconnect()

    except Exception:

        pass


    print()
    print("RTDE receive connection closed.")