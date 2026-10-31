from pathlib import Path
import sys

import numpy as np

from scipy.spatial.transform import Rotation

from roboticstoolbox.models.URDF.URDFRobot import (
    URDFRobot
)


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
# Robot IP
# ============================================================

ROBOT_IP = "192.168.0.191"


# ============================================================
# Helper:
# UR pose [x,y,z,rx,ry,rz] -> 4x4 homogeneous transform
# ============================================================

def ur_pose_to_matrix(pose):

    pose = np.asarray(
        pose,
        dtype=float,
    )

    if pose.shape != (6,):
        raise ValueError(
            "UR pose must contain 6 values."
        )

    T = np.eye(4)

    T[:3, 3] = pose[:3]

    T[:3, :3] = (
        Rotation.from_rotvec(
            pose[3:]
        )
        .as_matrix()
    )

    return T


# ============================================================
# Helper:
# rotation error
# ============================================================

def rotation_error_deg(
    R_reference,
    R_measured,
):

    R_error = (
        R_reference.T
        @ R_measured
    )

    rotvec = (
        Rotation.from_matrix(
            R_error
        )
        .as_rotvec()
    )

    return np.rad2deg(
        np.linalg.norm(rotvec)
    )


# ============================================================
# Load official UR3e model
# ============================================================

print()
print(
    "Loading official Robotics Toolbox UR3e..."
)

model = URDFRobot(
    "ur3e"
)


# ============================================================
# Connect to real robot
#
# READ ONLY
# ============================================================

robot = RealUR3e(
    robot_ip=ROBOT_IP
)


try:

    # ========================================================
    # Read actual robot state
    # ========================================================

    q = (
        robot.get_joint_positions()
    )

    actual_tcp = (
        robot.get_tcp_pose()
    )


    print()
    print(
        "=========================================="
    )

    print(
        "REAL UR3e <-> RTB MODEL COMPARISON"
    )

    print(
        "=========================================="
    )


    print()
    print(
        "Actual q [deg]:"
    )

    print(
        np.round(
            np.rad2deg(q),
            4,
        )
    )


    # ========================================================
    # RTB forward kinematics
    # ========================================================

    T_tool0_rtb = (
        model.fkine(
            q,
            end="tool0",
        )
        .A
    )


    T_flange_rtb = (
        model.fkine(
            q,
            end="flange",
        )
        .A
    )


    # ========================================================
    # Actual controller TCP
    # ========================================================

    T_tcp_actual = (
        ur_pose_to_matrix(
            actual_tcp
        )
    )


    # ========================================================
    # Positions
    # ========================================================

    p_tool0 = (
        T_tool0_rtb[:3, 3]
    )

    p_flange = (
        T_flange_rtb[:3, 3]
    )

    p_tcp = (
        T_tcp_actual[:3, 3]
    )


    print()
    print(
        "=========================================="
    )

    print(
        "POSITIONS"
    )

    print(
        "=========================================="
    )


    print()
    print(
        "RTB tool0 [m]:"
    )

    print(
        np.round(
            p_tool0,
            6,
        )
    )


    print()
    print(
        "RTB flange [m]:"
    )

    print(
        np.round(
            p_flange,
            6,
        )
    )


    print()
    print(
        "UR actual active TCP [m]:"
    )

    print(
        np.round(
            p_tcp,
            6,
        )
    )


    # ========================================================
    # Direct tool0 <-> actual TCP difference
    # ========================================================

    position_difference = (
        p_tcp
        - p_tool0
    )


    position_difference_mm = (
        position_difference
        * 1000.0
    )


    position_error_mm = (
        np.linalg.norm(
            position_difference_mm
        )
    )


    orientation_error = (
        rotation_error_deg(
            T_tool0_rtb[:3, :3],
            T_tcp_actual[:3, :3],
        )
    )


    print()
    print(
        "=========================================="
    )

    print(
        "DIRECT tool0 -> ACTIVE TCP DIFFERENCE"
    )

    print(
        "=========================================="
    )


    print()
    print(
        "Position difference [mm]:"
    )

    print(
        np.round(
            position_difference_mm,
            3,
        )
    )


    print(
        f"Position difference magnitude: "
        f"{position_error_mm:.3f} mm"
    )


    print()
    print(
        f"Orientation difference: "
        f"{orientation_error:.4f} deg"
    )


    # ========================================================
    # Relative transform:
    #
    # RTB tool0 -> actual active TCP
    #
    # If Pendant active TCP is zero and nominal kinematics
    # were exact, this would be approximately identity.
    #
    # If an RG2 TCP is active, this will also contain
    # that configured TCP offset.
    # ========================================================

    T_tool0_to_tcp = (
        np.linalg.inv(
            T_tool0_rtb
        )
        @ T_tcp_actual
    )


    relative_translation = (
        T_tool0_to_tcp[
            :3,
            3
        ]
    )


    relative_rotvec = (
        Rotation.from_matrix(
            T_tool0_to_tcp[
                :3,
                :3
            ]
        )
        .as_rotvec()
    )


    print()
    print(
        "=========================================="
    )

    print(
        "ESTIMATED tool0 -> ACTIVE TCP TRANSFORM"
    )

    print(
        "=========================================="
    )


    print()
    print(
        "Translation [mm]:"
    )

    print(
        np.round(
            relative_translation
            * 1000.0,
            3,
        )
    )


    print()
    print(
        "Rotation vector [rad]:"
    )

    print(
        np.round(
            relative_rotvec,
            6,
        )
    )


    print()
    print(
        "Rotation magnitude [deg]:"
    )

    print(
        f"{np.rad2deg(np.linalg.norm(relative_rotvec)):.4f}"
    )


    # ========================================================
    # Interpretation
    # ========================================================

    print()
    print(
        "=========================================="
    )

    print(
        "INTERPRETATION"
    )

    print(
        "=========================================="
    )


    if position_error_mm < 3.0:

        print()
        print(
            "RTB tool0 and active TCP positions "
            "are very close."
        )

        print(
            "This is consistent with a near-zero "
            "active TCP and reasonably close kinematics."
        )

    else:

        print()
        print(
            "A noticeable tool0 / active TCP "
            "difference is present."
        )

        print(
            "This does NOT automatically mean "
            "the UR3e model is wrong."
        )

        print()
        print(
            "Possible contributors:"
        )

        print(
            "  1. A non-zero active TCP "
            "(for example RG2)"
        )

        print(
            "  2. Nominal RTB kinematics versus "
            "this robot's factory calibration"
        )


finally:

    robot.close()