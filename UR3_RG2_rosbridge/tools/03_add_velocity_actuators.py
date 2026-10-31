from pathlib import Path
import xml.etree.ElementTree as ET


# ============================================================
# Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_XML = (
    PROJECT_ROOT
    / "models"
    / "ur3e_official"
    / "ur3e_official.xml"
)

OUTPUT_XML = (
    PROJECT_ROOT
    / "models"
    / "ur3e_official"
    / "ur3e_velocity.xml"
)


# ============================================================
# UR3e joints
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
# Simulation actuator settings
#
# IMPORTANT:
# These are simulation controller gains.
# They are NOT manufacturer UR3e servo gains.
# ============================================================

MAX_SIM_JOINT_SPEED = 0.20      # rad/s

KP = 100.0
KV = 20.0


# ============================================================
# Load MJCF
# ============================================================

if not INPUT_XML.exists():

    raise FileNotFoundError(
        f"Input model not found:\n{INPUT_XML}"
    )


tree = ET.parse(INPUT_XML)
root = tree.getroot()


print()
print("==========================================")
print("Adding UR3e integrated-velocity actuators")
print("==========================================")

print()
print("Input:")
print(INPUT_XML)


# ============================================================
# Verify all six joints exist
# ============================================================

available_joints = {
    joint.get("name")
    for joint in root.iter("joint")
    if joint.get("name") is not None
}


print()
print("Detected joints:")

for name in sorted(available_joints):
    print("  ", name)


print()
print("Checking required UR3e joints...")


for joint_name in JOINT_NAMES:

    if joint_name not in available_joints:

        raise RuntimeError(
            f"Required UR3e joint not found: "
            f"{joint_name}"
        )

    print(
        "  OK:",
        joint_name,
    )


# ============================================================
# Use implicitfast integrator
#
# Recommended by MuJoCo for actuators using damping / kv.
# ============================================================

option = root.find("option")

if option is None:

    option = ET.Element("option")

    # Put near beginning of model
    root.insert(
        0,
        option,
    )


option.set(
    "integrator",
    "implicitfast",
)


# ============================================================
# Find or create <actuator>
# ============================================================

actuator_section = root.find(
    "actuator"
)

if actuator_section is None:

    actuator_section = ET.SubElement(
        root,
        "actuator",
    )


# ============================================================
# Remove our old actuators if script is run again
# ============================================================

for actuator in list(
    actuator_section
):

    actuator_name = actuator.get(
        "name",
        "",
    )

    if actuator_name.startswith(
        "ur3e_qd_"
    ):

        actuator_section.remove(
            actuator
        )


# ============================================================
# Add six integrated-velocity servos
# ============================================================

for joint_name in JOINT_NAMES:

    actuator_name = (
        "ur3e_qd_"
        + joint_name
    )

    ET.SubElement(
        actuator_section,
        "intvelocity",
        {
            "name":
                actuator_name,

            "joint":
                joint_name,

            # qdot command range [rad/s]
            "ctrllimited":
                "true",

            "ctrlrange":
                (
                    f"{-MAX_SIM_JOINT_SPEED} "
                    f"{MAX_SIM_JOINT_SPEED}"
                ),

            # Simulation servo gains
            "kp":
                str(KP),

            "kv":
                str(KV),
        },
    )


# ============================================================
# Pretty formatting
# ============================================================

try:
    ET.indent(
        tree,
        space="    ",
    )
except AttributeError:
    pass


# ============================================================
# Save
# ============================================================

tree.write(
    OUTPUT_XML,
    encoding="utf-8",
    xml_declaration=True,
)


print()
print("==========================================")
print("SUCCESS")
print("==========================================")

print()
print("Output:")
print(OUTPUT_XML)

print()
print("Added actuators:")

for joint_name in JOINT_NAMES:

    print(
        "  ur3e_qd_"
        + joint_name
    )