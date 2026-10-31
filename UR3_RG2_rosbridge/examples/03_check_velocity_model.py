from pathlib import Path

import mujoco


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


model = mujoco.MjModel.from_xml_path(
    str(MODEL_PATH)
)


print()
print("==========================================")
print("UR3e MuJoCo Velocity Model")
print("==========================================")


print()
print("nq =", model.nq)
print("nv =", model.nv)
print("nu =", model.nu)
print("na =", model.na)


print()
print("Actuators:")
print()


for actuator_id in range(
    model.nu
):

    name = mujoco.mj_id2name(
        model,
        mujoco.mjtObj.mjOBJ_ACTUATOR,
        actuator_id,
    )

    ctrl_range = (
        model.actuator_ctrlrange[
            actuator_id
        ]
    )

    print(
        actuator_id,
        name,
        "ctrlrange =",
        ctrl_range,
    )