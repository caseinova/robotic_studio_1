from pathlib import Path
import sys

import numpy as np
import mujoco


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
# Imports after project path is available
# ============================================================

from roboticstoolbox.models.URDF.URDFRobot import URDFRobot


# ============================================================
# Model paths
# ============================================================

MUJOCO_MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "ur3e_official"
    / "ur3e_velocity.xml"
)


# ============================================================
# UR3e joint order
# ============================================================

JOINT_NAMES = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]


# ============================================================
# Test configuration
#
# Avoid q = all zeros because it is not a useful general
# Cartesian-control test configuration.
# ============================================================

q_test = np.deg2rad(
    [
        0.0,
        -90.0,
        90.0,
        -90.0,
        -90.0,
        0.0,
    ]
)


print()
print("==========================================")
print("UR3e RTB <-> MuJoCo Alignment Test")
print("==========================================")

print()
print("Test joint configuration [deg]:")
print(
    np.round(
        np.rad2deg(q_test),
        3,
    )
)


# ============================================================
# 1. Robotics Toolbox model
# ============================================================

print()
print("Loading Robotics Toolbox UR3e...")

rtb_robot = URDFRobot(
    "ur3e"
)


# ============================================================
# 2. MuJoCo model
# ============================================================

print("Loading MuJoCo UR3e...")

mj_model = mujoco.MjModel.from_xml_path(
    str(MUJOCO_MODEL_PATH)
)

mj_data = mujoco.MjData(
    mj_model
)


# ============================================================
# 3. Verify and set MuJoCo joint positions
# ============================================================

print()
print("MuJoCo joint mapping:")

for i, joint_name in enumerate(
    JOINT_NAMES
):

    joint_id = mujoco.mj_name2id(
        mj_model,
        mujoco.mjtObj.mjOBJ_JOINT,
        joint_name,
    )

    if joint_id < 0:

        raise RuntimeError(
            f"Joint not found in MuJoCo: "
            f"{joint_name}"
        )

    qpos_address = int(
        mj_model.jnt_qposadr[
            joint_id
        ]
    )

    mj_data.qpos[
        qpos_address
    ] = q_test[i]

    print(
        f"  q[{i}] -> {joint_name}"
    )


# ============================================================
# 4. Initialise intvelocity activation targets
#
# Prevent actuators from targeting zero while testing
# a non-zero configuration.
# ============================================================

for actuator_id in range(
    mj_model.nu
):

    act_address = int(
        mj_model.actuator_actadr[
            actuator_id
        ]
    )

    if act_address < 0:
        continue

    joint_id = int(
        mj_model.actuator_trnid[
            actuator_id,
            0,
        ]
    )

    if joint_id < 0:
        continue

    qpos_address = int(
        mj_model.jnt_qposadr[
            joint_id
        ]
    )

    mj_data.act[
        act_address
    ] = mj_data.qpos[
        qpos_address
    ]


mj_data.ctrl[:] = 0.0


# ============================================================
# 5. Forward MuJoCo kinematics
# ============================================================

mujoco.mj_forward(
    mj_model,
    mj_data,
)


# ============================================================
# 6. Find matching end frame
#
# Prefer tool0.
# If fixed-body fusion removed it, try flange.
# ============================================================

END_CANDIDATES = [
    "tool0",
    "flange",
    "wrist_3_link",
]


selected_end = None
mj_body_id = None


for candidate in END_CANDIDATES:

    body_id = mujoco.mj_name2id(
        mj_model,
        mujoco.mjtObj.mjOBJ_BODY,
        candidate,
    )

    if body_id >= 0:

        selected_end = candidate
        mj_body_id = body_id
        break


if selected_end is None:

    print()
    print("Available MuJoCo bodies:")
    print()

    for body_id in range(
        mj_model.nbody
    ):

        name = mujoco.mj_id2name(
            mj_model,
            mujoco.mjtObj.mjOBJ_BODY,
            body_id,
        )

        print(
            body_id,
            name,
        )

    raise RuntimeError(
        "Could not find tool0, flange, "
        "or wrist_3_link in MuJoCo."
    )


print()
print(
    "Comparison end frame:",
    selected_end,
)


# ============================================================
# 7. RTB forward kinematics
# ============================================================

T_rtb = rtb_robot.fkine(
    q_test,
    end=selected_end,
)

T_rtb_matrix = np.asarray(
    T_rtb.A,
    dtype=float,
)

p_rtb = T_rtb_matrix[
    0:3,
    3
]

R_rtb = T_rtb_matrix[
    0:3,
    0:3
]


# ============================================================
# 8. MuJoCo end-frame pose
# ============================================================

p_mj = np.array(
    mj_data.xpos[
        mj_body_id
    ],
    dtype=float,
)

R_mj = np.array(
    mj_data.xmat[
        mj_body_id
    ],
    dtype=float,
).reshape(
    3,
    3,
)


# ============================================================
# 9. Compare position
# ============================================================

position_error = (
    p_mj
    - p_rtb
)

position_error_mm = (
    np.linalg.norm(
        position_error
    )
    * 1000.0
)


# ============================================================
# 10. Compare orientation
# ============================================================

R_error = (
    R_rtb.T
    @ R_mj
)

cos_angle = (
    np.trace(R_error)
    - 1.0
) / 2.0

cos_angle = np.clip(
    cos_angle,
    -1.0,
    1.0,
)

orientation_error_rad = np.arccos(
    cos_angle
)

orientation_error_deg = np.rad2deg(
    orientation_error_rad
)


# ============================================================
# 11. Print results
# ============================================================

print()
print("==========================================")
print("RTB pose")
print("==========================================")

print()
print("Position [m]:")
print(
    np.round(
        p_rtb,
        6,
    )
)

print()
print("Rotation:")
print(
    np.round(
        R_rtb,
        6,
    )
)


print()
print("==========================================")
print("MuJoCo pose")
print("==========================================")

print()
print("Position [m]:")
print(
    np.round(
        p_mj,
        6,
    )
)

print()
print("Rotation:")
print(
    np.round(
        R_mj,
        6,
    )
)


print()
print("==========================================")
print("Alignment error")
print("==========================================")

print()
print(
    f"Position error: "
    f"{position_error_mm:.6f} mm"
)

print(
    f"Orientation error: "
    f"{orientation_error_deg:.6f} deg"
)


# ============================================================
# 12. Jacobian sanity check
# ============================================================

J = rtb_robot.jacob0(
    q_test,
    end=selected_end,
)


print()
print("==========================================")
print("RTB Jacobian")
print("==========================================")

print()
print(
    np.round(
        J,
        5,
    )
)

print()
print(
    "Jacobian shape:",
    J.shape,
)


# ============================================================
# 13. Singular values / conditioning
# ============================================================

singular_values = np.linalg.svd(
    J,
    compute_uv=False,
)

print()
print("Jacobian singular values:")
print(
    np.round(
        singular_values,
        6,
    )
)

smallest_sv = float(
    np.min(
        singular_values
    )
)

print()
print(
    "Smallest singular value:",
    f"{smallest_sv:.8f}"
)


print()
print("==========================================")

if (
    position_error_mm < 0.1
    and orientation_error_deg < 0.1
):

    print(
        "ALIGNMENT PASSED"
    )

else:

    print(
        "ALIGNMENT NEEDS REVIEW"
    )

print("==========================================")