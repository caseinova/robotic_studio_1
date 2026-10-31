from pathlib import Path
import time

import mujoco
import mujoco.viewer


# ============================================================
# Project paths
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
    / "ur3e_official.xml"
)


print()
print("==============================================")
print("Loading UR3e MuJoCo model")
print("==============================================")

print()
print(MODEL_PATH)


# ============================================================
# Load MuJoCo model
# ============================================================

model = mujoco.MjModel.from_xml_path(
    str(MODEL_PATH)
)

data = mujoco.MjData(
    model
)


# ============================================================
# Model information
# ============================================================

print()
print("Model loaded successfully.")

print()
print("nq:")
print(model.nq)

print()
print("nv:")
print(model.nv)

print()
print("nu:")
print(model.nu)

print()
print("Number of joints:")
print(model.njnt)


# ============================================================
# Print all joint names
# ============================================================

print()
print("==============================================")
print("MuJoCo joints")
print("==============================================")

for joint_id in range(model.njnt):

    name = mujoco.mj_id2name(
        model,
        mujoco.mjtObj.mjOBJ_JOINT,
        joint_id,
    )

    print(
        joint_id,
        name,
    )


# ============================================================
# Forward initial model
# ============================================================

mujoco.mj_forward(
    model,
    data,
)


# ============================================================
# Viewer
# ============================================================

with mujoco.viewer.launch_passive(
    model,
    data,
) as viewer:

    while viewer.is_running():

        mujoco.mj_step(
            model,
            data,
        )

        viewer.sync()

        # Avoid running the Python loop unnecessarily fast
        time.sleep(
            model.opt.timestep
        )