from pathlib import Path
import time

import numpy as np

import mujoco
import mujoco.viewer


# ============================================================
# Project / model path
# ============================================================

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[1]
)

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "ur3e_official"
    / "ur3e_velocity.xml"
)


# ============================================================
# UR3e joint order
#
# Keep this exact order.
# Later this will also match RTDE getActualQ()/speedJ().
# ============================================================

JOINT_NAMES = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]


ACTUATOR_NAMES = [
    "ur3e_qd_shoulder_pan_joint",
    "ur3e_qd_shoulder_lift_joint",
    "ur3e_qd_elbow_joint",
    "ur3e_qd_wrist_1_joint",
    "ur3e_qd_wrist_2_joint",
    "ur3e_qd_wrist_3_joint",
]


# ============================================================
# Control settings
# ============================================================

CONTROL_FREQUENCY = 100.0     # Hz
CONTROL_DT = 1.0 / CONTROL_FREQUENCY

TEST_SPEED = 0.05             # rad/s
MOVE_DURATION = 2.0           # seconds
STOP_DURATION = 1.0           # seconds


# ============================================================
# Load MuJoCo model
# ============================================================

print()
print("==========================================")
print("Loading UR3e velocity model")
print("==========================================")

print(MODEL_PATH)


model = mujoco.MjModel.from_xml_path(
    str(MODEL_PATH)
)

data = mujoco.MjData(
    model
)


# ============================================================
# Find joint IDs and qpos addresses
# ============================================================

joint_ids = []
qpos_addresses = []

for joint_name in JOINT_NAMES:

    joint_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_JOINT,
        joint_name,
    )

    if joint_id < 0:
        raise RuntimeError(
            f"Joint not found: {joint_name}"
        )

    qpos_address = int(
        model.jnt_qposadr[joint_id]
    )

    joint_ids.append(
        joint_id
    )

    qpos_addresses.append(
        qpos_address
    )


# ============================================================
# Find actuator IDs
# ============================================================

actuator_ids = []

for actuator_name in ACTUATOR_NAMES:

    actuator_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_ACTUATOR,
        actuator_name,
    )

    if actuator_id < 0:
        raise RuntimeError(
            f"Actuator not found: {actuator_name}"
        )

    actuator_ids.append(
        actuator_id
    )


# ============================================================
# Helper: read only the six UR3e joints
# ============================================================

def get_q():

    return np.array(
        [
            data.qpos[address]
            for address in qpos_addresses
        ],
        dtype=float,
    )


def get_qd():

    result = []

    for joint_id in joint_ids:

        dof_address = int(
            model.jnt_dofadr[joint_id]
        )

        result.append(
            data.qvel[dof_address]
        )

    return np.array(
        result,
        dtype=float,
    )


# ============================================================
# Reset simulation
# ============================================================

mujoco.mj_resetData(
    model,
    data,
)


# ============================================================
# IMPORTANT:
# Initialise intvelocity activation states to current q.
#
# intvelocity internally stores a position setpoint in data.act.
# If the robot later starts from a non-zero configuration,
# leaving act = 0 would make the servo try to return to zero.
# ============================================================

for joint_id, actuator_id in zip(
    joint_ids,
    actuator_ids,
):

    qpos_address = int(
        model.jnt_qposadr[joint_id]
    )

    act_address = int(
        model.actuator_actadr[actuator_id]
    )

    if act_address >= 0:

        data.act[
            act_address
        ] = data.qpos[
            qpos_address
        ]


# Start with zero velocity command
data.ctrl[:] = 0.0


mujoco.mj_forward(
    model,
    data,
)


# ============================================================
# Determine physics substeps
# ============================================================

PHYSICS_DT = float(
    model.opt.timestep
)


SUBSTEPS = max(
    1,
    int(
        round(
            CONTROL_DT
            / PHYSICS_DT
        )
    ),
)


print()
print("Control frequency:")
print(
    f"{CONTROL_FREQUENCY:.1f} Hz"
)

print()
print("Control period:")
print(
    f"{CONTROL_DT:.4f} s"
)

print()
print("MuJoCo physics timestep:")
print(
    f"{PHYSICS_DT:.6f} s"
)

print()
print("Physics substeps per control cycle:")
print(SUBSTEPS)


# ============================================================
# Initial state
# ============================================================

q_initial = get_q()

print()
print("Initial q [rad]:")
print(
    np.round(
        q_initial,
        5,
    )
)


# ============================================================
# Viewer
# ============================================================

with mujoco.viewer.launch_passive(
    model,
    data,
) as viewer:

    # Give viewer one update
    viewer.sync()

    try:

        # ====================================================
        # PHASE 1:
        # Move only wrist_3
        # ====================================================

        print()
        print("==========================================")
        print("PHASE 1: Move wrist_3")
        print("==========================================")

        print(
            f"Command = +{TEST_SPEED:.3f} rad/s"
        )

        print(
            f"Duration = {MOVE_DURATION:.1f} s"
        )


        total_cycles = int(
            MOVE_DURATION
            * CONTROL_FREQUENCY
        )


        for cycle in range(
            total_cycles
        ):

            cycle_start = (
                time.perf_counter()
            )


            # ----------------------------------------------
            # Student-style qdot command
            # ----------------------------------------------

            qd_command = np.zeros(
                6,
                dtype=float,
            )

            qd_command[5] = (
                TEST_SPEED
            )


            # ----------------------------------------------
            # Send qdot directly to six actuators
            # ----------------------------------------------

            for actuator_id, command in zip(
                actuator_ids,
                qd_command,
            ):

                ctrl_address = int(
                    model.actuator_ctrladr[
                        actuator_id
                    ]
                )

                data.ctrl[
                    ctrl_address
                ] = command


            # ----------------------------------------------
            # Advance MuJoCo physics
            # ----------------------------------------------

            for _ in range(
                SUBSTEPS
            ):

                mujoco.mj_step(
                    model,
                    data,
                )


            viewer.sync()


            # ----------------------------------------------
            # Print approximately every 0.2 s
            # ----------------------------------------------

            if cycle % 20 == 0:

                q_now = get_q()
                qd_now = get_qd()

                print(
                    f"t = {cycle * CONTROL_DT:5.2f} s "
                    f"| q6 = {q_now[5]: .4f} rad "
                    f"| qd6 = {qd_now[5]: .4f} rad/s"
                )


            # ----------------------------------------------
            # Real-time pacing
            # ----------------------------------------------

            elapsed = (
                time.perf_counter()
                - cycle_start
            )

            remaining = (
                CONTROL_DT
                - elapsed
            )

            if remaining > 0:

                time.sleep(
                    remaining
                )


        # ====================================================
        # PHASE 2:
        # Zero velocity command
        # ====================================================

        print()
        print("==========================================")
        print("PHASE 2: Stop")
        print("==========================================")


        total_cycles = int(
            STOP_DURATION
            * CONTROL_FREQUENCY
        )


        for cycle in range(
            total_cycles
        ):

            cycle_start = (
                time.perf_counter()
            )


            # Zero qdot command
            data.ctrl[:] = 0.0


            for _ in range(
                SUBSTEPS
            ):

                mujoco.mj_step(
                    model,
                    data,
                )


            viewer.sync()


            elapsed = (
                time.perf_counter()
                - cycle_start
            )

            remaining = (
                CONTROL_DT
                - elapsed
            )

            if remaining > 0:

                time.sleep(
                    remaining
                )


    except KeyboardInterrupt:

        print()
        print(
            "Operator requested stop."
        )


    finally:

        # Absolutely make sure command becomes zero
        data.ctrl[:] = 0.0


# ============================================================
# Final results
# ============================================================

q_final = get_q()

delta_q = (
    q_final
    - q_initial
)


print()
print("==========================================")
print("RESULT")
print("==========================================")

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
    "Expected wrist_3 change "
    "(ideal command integration):"
)

expected_change = (
    TEST_SPEED
    * MOVE_DURATION
)

print(
    f"{expected_change:.5f} rad"
)

print(
    f"{np.rad2deg(expected_change):.3f} deg"
)


print()
print(
    "Actual wrist_3 change:"
)

print(
    f"{delta_q[5]:.5f} rad"
)

print(
    f"{np.rad2deg(delta_q[5]):.3f} deg"
)


print()
print("Simulation finished.")